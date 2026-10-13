"""LLM provider abstraction — Anthropic, plus a no-cost mock for tests.

  LLM_PROVIDER / LLM_PROVIDER_CHEAP / LLM_PROVIDER_STRONG   anthropic | mock

The pipeline has no embeddings (removed 30-Sep-2026, BUILD_NOTES D15): S3 clusters with the
LLM and falls back to singleton events, so no embedding provider exists here.
OpenAI, Gemini and Ollama providers were removed at the same time.

JSON output is enforced via structured output / tool-use, never regex parsing.
"""
from __future__ import annotations

from typing import Any


# ====================================================================== LLM


class LLMResult:
    __slots__ = ("data", "model", "raw")

    def __init__(self, data: dict, model: str, raw: Any = None):
        self.data = data
        self.model = model
        self.raw = raw


class TruncatedOutputError(RuntimeError):
    """The model hit max_tokens. Its structured output is incomplete and must not be used.

    BUILD_NOTES F27: Anthropic returns a *partial* tool_use input when it stops at
    max_tokens (observed: only `event_id`, no `facts`), which parsed as a valid empty
    extraction and was recorded as success. Raising makes truncation a visible failure.
    """


class BaseLLM:
    name = "base"

    def complete_json(
        self,
        system: str,
        user: str,
        json_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8000,
    ) -> LLMResult:
        raise NotImplementedError




class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, api_key: str, default_model: str):
        import anthropic

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = default_model
        # Newer models reject some request fields. Each rejection is learned ONCE (first 400)
        # and remembered for the rest of the run, so only one call is wasted per field:
        #   temperature           -> deprecated on Sonnet 5 / Opus 5.x
        #   forced tool_choice    -> 400 on Opus 5.5 / Fable 5.1 ("tool_choice ... not supported")
        self._omit_temperature = False
        self._auto_tool_choice = False

    def complete_json(self, system, user, json_schema, schema_name="response",
                      model=None, temperature=0.0, max_tokens=8000) -> LLMResult:
        mdl = model or self._default_model
        # Structured output via tool use — the reliable way to get JSON from Claude.
        tool = {"name": schema_name, "description": "Return the structured result.",
                "input_schema": json_schema}

        def _kwargs():
            kw = dict(model=mdl, max_tokens=max_tokens, tools=[tool],
                      messages=[{"role": "user", "content": user}])
            if self._auto_tool_choice:
                kw["tool_choice"] = {"type": "auto"}
                kw["system"] = (system + "\n\nReturn your answer ONLY by calling the "
                                f"`{schema_name}` tool exactly once.")
            else:
                kw["tool_choice"] = {"type": "tool", "name": schema_name}
                kw["system"] = system
            if not self._omit_temperature:
                kw["temperature"] = temperature
            return kw

        for _attempt in range(3):
            try:
                resp = self._client.messages.create(**_kwargs())
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "temperature" in msg and not self._omit_temperature:
                    self._omit_temperature = True
                elif "tool_choice" in msg and not self._auto_tool_choice:
                    self._auto_tool_choice = True
                else:
                    raise
        else:  # pragma: no cover - both fields learned and it still failed
            raise RuntimeError(f"{mdl}: request rejected after adapting temperature/tool_choice")

        if resp.stop_reason == "max_tokens":
            raise TruncatedOutputError(f"{mdl} stopped at max_tokens={max_tokens}")
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"{mdl} refused the request (stop_details={resp.stop_details})")
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == schema_name:
                return LLMResult(block.input, mdl, resp)
        raise RuntimeError(f"Anthropic returned no `{schema_name}` tool call "
                           f"(stop_reason={resp.stop_reason})")


class MockLLM(BaseLLM):
    """No API key, no network, no cost. Deterministic, schema-valid responses.

    Purpose: run the WHOLE pipeline end-to-end before spending anything, so plumbing,
    DB writes, clustering and the UI can be validated independently of model quality.

    It is deliberately dumb and it says so: every card it produces is stamped
    model_version 'mock' so mock output can never be mistaken for real analysis.
    """

    name = "mock"

    def __init__(self, tier: str = "cheap"):
        self._tier = tier

    def complete_json(self, system, user, json_schema, schema_name="response",
                      model=None, temperature=0.0, max_tokens=8000) -> LLMResult:
        handler = {
            "triage": self._triage,
            "extraction": self._extraction,
            "interpretation": self._interpretation,
        }.get(schema_name, lambda _u: {})
        return LLMResult(handler(user), f"mock/{self._tier}", None)

    # -- S2: keyword heuristic, precision-first (unknown -> low score)
    #
    # NOTE: this must emit the same shape as the real two-dimensional triage
    # (TRIAGE_SCHEMA in s2_filter.py: portfolio_relevance, kb_value, event_type) or
    # s2_filter._pass_b silently reads 0/0 for every article via r.get(...) defaults —
    # a real bug found during live smoke-testing (22-Jul-2026): mock mode scored every
    # article 0 and dropped 100% of input, because this handler still returned the
    # OLD single-dimension "score"/"event_key" shape from before the two-dimensional
    # redesign. Fixed here; see BUILD_NOTES.
    @staticmethod
    def _triage(user: str) -> dict:
        import re as _re

        # words that argue for portfolio_relevance (moves client money now)
        portfolio_words = (
            "rbi", "repo", "inflation", "cpi", "wpi", "gdp", "crude", "brent", "opec",
            "rupee", "forex", "fii", "fpi", "nifty", "sensex", "yield", "gsec", "bond",
            "earnings", "profit", "results", "tariff", "dividend", "ipo", "buyback",
            "stock market", "s&p", "nasdaq", "dow jones",
        )
        # words that argue for kb_value (durable/structural knowledge)
        kb_words = (
            "import", "export", "capacity", "contract", "structural", "policy",
            "regulation", "duty", "curb", "supply", "reserves", "fdi", "investment",
            "infrastructure", "transmission", "coking coal", "lng", "record", "highest",
            "lowest", "multi-month", "multi-year", "consumption", "trade deal",
        )
        # noise: single-stock trivia, filings, market chatter with no structural content
        weak = ("cricket", "movie", "actor", "recipe", "travel", "celebrity", "sport",
                "quote of the day", "appeal no", "gainers & losers", "gainers &amp; losers",
                "52-week", "multibagger", "gmp signals")

        # Topic buckets -> a REAL controlled-taxonomy slug (dictionaries/event_taxonomy.yaml).
        # event_key itself is derived downstream by build_event_key(event_type, taxonomy).
        topics = [
            ("crude-price-move", ("crude", "oil", "brent", "opec", "petrol")),
            ("inflation-print", ("inflation", "wpi", "cpi", "wholesale", "consumer price")),
            ("rbi-policy", ("rbi", "repo", "monetary policy", "mpc", "vrr", "reserve money",
                            "forward book", "swap facility")),
            ("rupee-move", ("rupee", "forex", "dollar")),
            ("fpi-flows", ("fpi", "fii", "foreign investor", "outflow", "inflow")),
            ("market-daily-wrap", ("sensex", "nifty", "stock market today", "gainers",
                                   "losers", "points")),
            ("bond-yields", ("yield", "gsec", "g-sec", "t-bill", "treasury bill")),
            ("india-earnings-season", ("earnings", "q1 results", "q1 fy", "results live",
                                       "profit soars", "profit jump", "net profit")),
            ("trade-and-external", ("trade deal", "import bill", "export", "fdi",
                                    "current account", "trade deficit")),
            ("growth-data", ("gdp", "iip", "pmi", "economic activity", "consumption")),
            ("sector-development", ("transmission", "capacity", "supply chain",
                                    "import duty", "lng", "coking coal")),
            ("corporate-action", ("ipo", "buyback", "dividend", "rights issue",
                                  "board meeting", "fundraise")),
            ("regulatory-change", ("sebi", "circular", "appeal no", "certification",
                                   "research analyst")),
            ("global-markets-move", ("s&p", "nasdaq", "dow jones", "wall street",
                                     "global market", "airbus")),
        ]
        results = []
        for m in _re.finditer(r'"article_id":\s*(\d+).*?"title":\s*"([^"]*)"', user, _re.S):
            aid, title = int(m.group(1)), m.group(2).lower()

            pr = 4.0 + 2.0 * sum(1 for k in portfolio_words if k in title)
            kb = 3.0 + 2.0 * sum(1 for k in kb_words if k in title)
            noise = sum(1 for k in weak if k in title)
            pr -= 4.0 * noise
            kb -= 4.0 * noise
            pr, kb = max(0.0, min(10.0, pr)), max(0.0, min(10.0, kb))

            etype = "uncategorised"
            for slug, words in topics:
                if any(w in title for w in words):
                    etype = slug
                    break

            results.append({
                "article_id": aid,
                "portfolio_relevance": pr,
                "kb_value": kb,
                "event_type": etype,
                "portfolio_reason": "mock triage — keyword heuristic, not a real judgement",
                "kb_reason": "mock triage — keyword heuristic, not a real judgement",
            })
        return {"results": results}

    # -- S4: pull real numbers out of the text so downstream logic gets real shapes
    @staticmethod
    def _extraction(user: str) -> dict:
        import re as _re

        facts = []
        for m in _re.finditer(r'"article_id":\s*(\d+),.*?"body_or_summary":\s*"(.*?)"(?=,\n|\s*\})',
                              user, _re.S):
            aid, body = int(m.group(1)), m.group(2)
            for n in list(_re.finditer(r"(\d+\.\d+)\s*(per cent|%|per barrel)", body))[:4]:
                facts.append({
                    "article_id": aid,
                    "metric": "mock_metric",
                    "value": n.group(1),
                    "unit": "%" if "cent" in n.group(2) or "%" in n.group(2) else "USD/barrel",
                    "as_of_period": "mock period",
                    "claim_context": body[max(0, n.start() - 60):n.end() + 40],
                    "category": "macro_economics",
                    "entities": ["india"],
                    "fact_type": "point_in_time",
                    "persistence_tier": "event_context",
                    "verification_status": "single_source",
                    "confidence": "low",
                    "new_metric": True,
                    "new_entity": False,
                })
        return {"facts": facts, "transient_context": []}

    # -- S6: a schema-valid card, honestly labelled, citing nothing it cannot support
    @staticmethod
    def _interpretation(user: str) -> dict:
        import re as _re

        fact_ids = [int(x) for x in _re.findall(r'"fact_id":\s*(\d+)', user)][:3]
        return {
            "event_title": "MOCK CARD — not real analysis",
            "event_type": "macro_economics",
            "is_update": False,
            "what_changed_since_last_version": None,
            "summary": ("This card was generated by the mock provider to exercise the "
                        "pipeline end to end. It contains no real interpretation."),
            "source_agreement": {"consistent_facts": [], "discrepancies": []},
            "impact_chains": [{
                "chain": [{
                    "from": "mock_driver", "to": "mock_asset",
                    "mechanism": "Mock mechanism generated without a language model",
                    "fact_ids": fact_ids, "evidence": [],
                }],
                "terminal_impact": {"asset": "mock_asset", "direction": "unclear",
                                    "magnitude": "unclear",
                                    "conditionality": "mock output — no real condition"},
            }],
            "historical_patterns_cited": [],
            "investor_actions_conditional": [],
            "relevant_fund_categories": [],
            "rm_one_liner": "Mock output — do not use with clients.",
            "data_gaps_for_kb": ["mock provider produced this card; no real gaps identified"],
            "footnotes": {"*": "No official records found supporting this data"},
        }




def get_llm(config: dict, tier: str = "cheap") -> BaseLLM:
    """tier: 'cheap' (S2 triage, S3 clustering, S4 extract) or 'strong' (S6 interpret)."""
    provider = (config.get(f"llm_provider_{tier}") or config.get("llm_provider")
                or "anthropic").lower()
    if provider == "mock":
        return MockLLM(tier)
    if provider == "anthropic":
        return AnthropicLLM(config.get("anthropic_api_key"), config[f"model_{tier}_anthropic"])
    raise ValueError(
        f"Unknown provider '{provider}' for tier '{tier}'. Supported: anthropic, mock."
    )

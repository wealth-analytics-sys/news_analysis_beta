"""Mock provider + S6 citation validator tests. No keys, no DB, no network."""
import pytest

from src.providers import AnthropicLLM, MockLLM, get_llm
from src.s6_interpret import validate_citations


# ---------------------------------------------------------------- provider wiring
def test_mock_providers_need_no_api_key():
    cfg = {"llm_provider": "mock"}
    assert get_llm(cfg, "cheap").name == "mock"
    assert get_llm(cfg, "strong").name == "mock"


def test_per_tier_provider_resolution():
    """Each tier resolves on its own and falls back to LLM_PROVIDER when unset."""
    cfg = {"llm_provider": "anthropic", "llm_provider_cheap": "mock", "llm_provider_strong": None}
    assert get_llm(cfg, "cheap").name == "mock"
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        get_llm({**cfg, "anthropic_api_key": None, "model_strong_anthropic": "m"}, "strong")


def test_default_provider_is_anthropic():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        get_llm({"anthropic_api_key": None, "model_cheap_anthropic": "m"}, "cheap")


def test_unknown_provider_lists_the_supported_ones():
    """OpenAI, Gemini and Ollama were removed (BUILD_NOTES D15)."""
    for gone in ("openai", "gemini", "ollama", "banana"):
        with pytest.raises(ValueError, match="Supported: anthropic, mock"):
            get_llm({"llm_provider": gone}, "cheap")


def test_anthropic_falls_back_from_forced_tool_choice_once():
    """Opus 5.5 rejects forced tool_choice; the provider retries with auto and remembers."""
    import types

    calls = []

    def create(**kw):
        calls.append(kw)
        if kw["tool_choice"]["type"] == "tool":
            raise RuntimeError("400: tool_choice type 'tool' is not supported for this model")
        if "temperature" in kw:
            raise RuntimeError("400: `temperature` is deprecated for this model")
        return types.SimpleNamespace(stop_reason="tool_use", content=[
            types.SimpleNamespace(type="tool_use", name="extraction", input={"facts": []})])

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._default_model, llm._omit_temperature, llm._auto_tool_choice = "m", False, False
    llm._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    assert llm.complete_json("sys", "u", {}, "extraction").data == {"facts": []}
    assert llm._auto_tool_choice and llm._omit_temperature
    assert "extraction" in calls[-1]["system"]            # told to call the tool
    n = len(calls)
    llm.complete_json("sys", "u", {}, "extraction")
    assert len(calls) == n + 1                            # learned: no wasted calls


# ---------------------------------------------------------------- mock llm
#
# NOTE: these two tests were rewritten 22-Jul-2026 after live smoke-testing found that
# MockLLM._triage still returned the OLD single-dimension shape ("score"/"event_key")
# from before the two-dimensional triage redesign (portfolio_relevance/kb_value/
# event_type — see s2_filter.TRIAGE_SCHEMA). s2_filter._pass_b reads the new field names
# via r.get(..., 0), so every mock-mode article silently scored 0 and the whole pipeline
# dropped 100% of input. These tests were pinning the stale contract, which is exactly
# why the bug went unnoticed. Fixed in providers.py; tests below assert the corrected
# shape that _pass_b actually consumes.
def test_mock_triage_drops_irrelevant_and_keeps_market_news():
    user = ('[{"article_id": 1, "title": "RBI holds repo rate as inflation cools"},'
            ' {"article_id": 2, "title": "Kohli scores century in third test"}]')
    res = MockLLM("cheap").complete_json("s", user, {}, "triage")
    by = {r["article_id"]: r for r in res.data["results"]}
    assert max(by[1]["portfolio_relevance"], by[1]["kb_value"]) >= 7      # kept
    assert max(by[2]["portfolio_relevance"], by[2]["kb_value"]) < 7       # dropped


def test_mock_triage_groups_same_topic_under_one_event_key():
    """event_type (-> event_key) is what buckets articles for S3, so the mock must emit a
    real controlled-taxonomy slug and group same-topic headlines under it."""
    user = ('[{"article_id": 1, "title": "Sensex drops as crude oil surges"},'
            ' {"article_id": 2, "title": "Brent crude climbs on West Asia tensions"},'
            ' {"article_id": 3, "title": "WPI inflation rises to 9.87% in June"}]')
    res = MockLLM("cheap").complete_json("s", user, {}, "triage")
    types = {r["article_id"]: r["event_type"] for r in res.data["results"]}
    assert types[1] == types[2] == "crude-price-move"
    assert types[3] != types[1]


def test_mock_interpretation_is_schema_valid_and_self_labelled():
    card = MockLLM("strong").complete_json("s", '"fact_id": 101', {}, "interpretation").data
    for field in ("event_title", "event_type", "summary", "impact_chains"):
        assert field in card
    assert "MOCK" in card["event_title"]          # can never pass as real analysis
    # confidence_overall / time_horizon / chain_confidence were removed in payload v1.4
    assert "confidence_overall" not in card
    assert "time_horizon" not in card
    assert all("chain_confidence" not in c for c in card["impact_chains"])


# ---------------------------------------------------------------- validator
def _facts():
    return {
        101: {"fact_id": 101, "metric": "brent_crude_price", "entities": ["brent_crude"]},
        102: {"fact_id": 102, "metric": "us_10y_yield", "entities": ["us_treasury_10y", "usa"]},
        119: {"fact_id": 119, "metric": "strait_of_hormuz_global_oil_supply_share",
              "entities": ["strait_of_hormuz", "global_crude_supply"]},
    }


def _card(fact_ids, frm="strait_of_hormuz_disruption_risk",
          to="global_crude_supply_availability", mech="About a fifth of global oil supplies transit the Strait of Hormuz"):
    return {"impact_chains": [{"chain": [
        {"from": frm, "to": to, "mechanism": mech, "fact_ids": fact_ids, "evidence": []}]}]}


def test_validator_keeps_a_correct_citation():
    card, warns = validate_citations(_card([119]), _facts())
    assert card["impact_chains"][0]["chain"][0]["fact_ids"] == [119]
    assert warns == []


def test_validator_strips_grossly_unrelated_citation():
    """A US bond yield cited on a Hormuz shipping hop must be stripped."""
    card, warns = validate_citations(_card([102]), _facts())
    assert card["impact_chains"][0]["chain"][0]["fact_ids"] == []
    assert len(warns) == 1
    assert warns[0]["fact_id"] == 102


def test_validator_warns_on_unknown_fact_id():
    card, warns = validate_citations(_card([999]), _facts())
    assert card["impact_chains"][0]["chain"][0]["fact_ids"] == []
    assert "not found" in warns[0]["reason"]


def test_validator_known_blind_spot_is_documented_not_fixed():
    """DECISION (c), BUILD_NOTES F7 — adjacent-topic citations PASS.

    A Brent PRICE fact cited on an import-dependency claim is a real compliance risk and
    this validator does NOT catch it, because the hop's `from` node is the fact's own
    topic. This test pins that known behaviour so nobody mistakes the validator for a
    complete control. Prompt discipline + human review are the primary safeguard.
    """
    card, warns = validate_citations(
        _card([101], frm="brent_crude_price", to="india_crude_import_bill",
              mech="India imports approximately 85% of its crude"),
        _facts())
    assert card["impact_chains"][0]["chain"][0]["fact_ids"] == [101]   # NOT stripped
    assert warns == []


# ------------------------------------------------- .env loading (regression)
def test_dotenv_parses_quoted_values_and_strips_quotes(tmp_path):
    """REGRESSION: config.py originally never read .env at all, so pasted keys were
    silently ignored. It also must strip the quotes people naturally add."""
    import os

    from src.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        'QUOTED_KEY="quoted-key"\n'
        "SINGLE_QUOTED_KEY='single-quoted'\n"
        "export EXPORTED_KEY=exported-value\n"
        "PLAIN_KEY=plain-value  # trailing comment\n"
        "# a full line comment\n"
        "\n",
        encoding="utf-8",
    )
    for k in ("QUOTED_KEY", "SINGLE_QUOTED_KEY", "EXPORTED_KEY", "PLAIN_KEY"):
        os.environ.pop(k, None)

    n = load_dotenv(env, override=True)
    assert n == 4
    assert os.environ["QUOTED_KEY"] == "quoted-key"
    assert os.environ["SINGLE_QUOTED_KEY"] == "single-quoted"
    assert os.environ["EXPORTED_KEY"] == "exported-value"
    assert os.environ["PLAIN_KEY"] == "plain-value"
    for k in ("QUOTED_KEY", "SINGLE_QUOTED_KEY", "EXPORTED_KEY", "PLAIN_KEY"):
        os.environ.pop(k, None)


def test_dotenv_does_not_shadow_real_environment(tmp_path):
    """CI secrets must win over a stray local .env."""
    import os

    from src.config import load_dotenv

    os.environ["CI_WINS"] = "from-ci"
    env = tmp_path / ".env"
    env.write_text("CI_WINS=from-file\n", encoding="utf-8")
    load_dotenv(env, override=False)
    assert os.environ["CI_WINS"] == "from-ci"
    os.environ.pop("CI_WINS", None)


def test_dotenv_missing_file_is_not_an_error(tmp_path):
    from src.config import load_dotenv

    assert load_dotenv(tmp_path / "nope.env") == 0

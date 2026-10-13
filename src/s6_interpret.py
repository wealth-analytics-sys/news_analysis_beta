"""S6 — Interpret + deterministic citation validator. Two modes, one write gate.

  agent mode  Claude Code writes each card (no API credits): run_s6.py pending / export /
              write, driven by .claude/commands/s6-interpret.md.
  API mode    the strong-tier model writes each card (run_s6.py api -> run()).

Both modes use the same pending_events(), build_inputs() and write_card(), so a card
reaches event_interpretations only through the same gates: citation validator, JSON
schema (INTERPRET_SCHEMA, payload v1.4), wording check, versioning.

Inputs per event: article summaries, this event's facts (with statuses and tiers), the
event's transient context (S5 keeps it on news_events; never a fact, D1/D15), top-K KB
facts (persistence_tier='kb_durable'), and the prior version if this is an update.

Versioning: a new interpretation is always version N+1. Prior versions are NEVER
overwritten. Events flagged needs_reinterpretation get is_update=true.

CITATION VALIDATOR — scope per decision (c), BUILD_NOTES F7:
This is a GROSS-MISMATCH BACKSTOP ONLY. It reliably catches citations to facts on an
unrelated topic. It does NOT catch adjacent-topic citations (a price print cited on a
structural claim). Prompt discipline plus human review are the primary control for that
class. Cards with stripped citations are flagged for review via citation_warnings.
"""
from __future__ import annotations

import json
import logging
import re

from . import db
from .config import load_prompt
from .providers import get_llm

log = logging.getLogger(__name__)

KB_RETRIEVAL_LIMIT = 10

INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "event_title": {"type": "string"},
        "event_type": {"type": "string"},
        "primary_category": {"type": "string",
                             "enum": ["global_economy", "global_equity", "indian_economy",
                                      "indian_equity_corporate", "commodity", "regulatory",
                                      "other"]},
        "is_update": {"type": "boolean"},
        "what_changed_since_last_version": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "source_agreement": {
            "type": "object",
            "properties": {
                "consistent_facts": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"claim": {"type": "string"},
                                   "article_ids": {"type": "array", "items": {"type": "integer"}}}}},
                "discrepancies": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"claim": {"type": "string"},
                                   "values": {"type": "array", "items": {"type": "object"}}}}},
            },
        },
        "impact_chains": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "chain": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"}, "to": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "fact_ids": {"type": "array", "items": {"type": "integer"}},
                        "evidence": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["from", "to", "mechanism", "fact_ids"]}},
                "plain_explanation": {"type": "string"},
                "terminal_impact": {"type": "object", "properties": {
                    "asset": {"type": "string"},
                    "direction": {"type": "string",
                                  "enum": ["negative", "positive", "mixed", "unclear"]},
                    "magnitude": {"type": "string",
                                  "enum": ["small", "moderate", "large", "unclear"]},
                    "conditionality": {"type": "string"}}},
            },
            "required": ["chain", "terminal_impact"]}},
        "historical_patterns_cited": {"type": "array", "items": {"type": "object"}},
        "investor_actions_conditional": {"type": "array", "items": {"type": "object"}},
        "relevant_fund_categories": {"type": "array", "items": {"type": "string"}},
        "rm_one_liner": {"type": "string"},
        "data_gaps_for_kb": {"type": "array", "items": {"type": "string"}},
        "footnotes": {"type": "object"},
    },
    "required": ["event_title", "event_type", "summary", "impact_chains"],
}


# ---------------------------------------------------------------- validator
# Tokens that appear in almost every metric, entity and hop, so an overlap on them proves
# nothing. BUILD_NOTES F30: with entity `india` on most facts and "India" in most hops, the
# old validator passed nearly every citation. A match must now come from a specific token.
UBIQUITOUS_TOKENS = frozenset({
    "india", "indian", "indias", "global", "world", "us", "usa", "the", "and", "of", "in",
    "pct", "percent", "yoy", "mom", "qoq", "rate", "change", "growth", "level", "total",
    "value", "share", "price", "index", "market", "data", "monthly", "annual", "fy",
})


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", str(text).lower().replace("_", " ")).split())


def _specific(tokens: set[str]) -> set[str]:
    return {t for t in tokens if t not in UBIQUITOUS_TOKENS and len(t) > 1}


def validate_citations(payload: dict, facts_by_id: dict[int, dict],
                       aliases: dict[str, list[str]] | None = None) -> tuple[dict, list[dict]]:
    """Strip citations whose fact is topically unrelated to the hop. Returns (payload, warnings).

    Intentionally strict-but-dumb (CLAUDE.md §5.3). A citation passes if a SPECIFIC token of
    the fact's metric or entity slugs — or of their dictionary aliases (`aliases`:
    canonical name -> [alias, ...]) — appears in the hop's from/to/mechanism.
    Ubiquitous tokens ("india", "rate", "pct" ...) never count (F30).
    Known blind spot: adjacent-topic citations pass. See BUILD_NOTES F7 — accepted risk,
    human review is the backstop.
    """
    aliases = aliases or {}
    warnings: list[dict] = []
    for ci, chain in enumerate(payload.get("impact_chains", [])):
        for hi, hop in enumerate(chain.get("chain", [])):
            ctx = _specific(_tokens(
                f"{hop.get('from','')} {hop.get('to','')} {hop.get('mechanism','')}"))
            kept = []
            for fid in hop.get("fact_ids", []) or []:
                fact = facts_by_id.get(fid)
                if fact is None:
                    warnings.append({"chain_idx": ci, "hop_idx": hi, "fact_id": fid,
                                     "reason": "fact_id not found for this event or its KB facts"})
                    continue
                metric = fact.get("metric", "")
                overlap = any(_specific(_tokens(n)) & ctx
                              for n in [metric] + list(aliases.get(metric, []) or []))
                if not overlap:
                    for e in (fact.get("entities") or []):
                        names = [e] + list(aliases.get(e, []) or [])
                        if any(_specific(_tokens(n)) & ctx for n in names):
                            overlap = True
                            break
                if overlap:
                    kept.append(fid)
                else:
                    warnings.append({
                        "chain_idx": ci, "hop_idx": hi, "fact_id": fid,
                        "reason": (f"metric '{fact.get('metric')}' and entities "
                                   f"{fact.get('entities')} do not overlap the hop"),
                    })
            hop["fact_ids"] = kept
    return payload, warnings


# ---------------------------------------------------------------- inputs
def _load_events(config: dict, event_ids: list[int] | None) -> list[dict]:
    """Open events that need a card: none yet, or flagged needs_reinterpretation.

    BUILD_NOTES F38: this used to return EVERY open event, so each scheduled run wrote a
    fresh version for every open event whether or not anything had changed.
    """
    where, params = "", ()
    if event_ids:
        where, params = "AND e.id = ANY(%s)", (event_ids,)
    return db.fetch_all(config, f"""
        SELECT e.id, e.event_key, e.title, e.status, e.needs_reinterpretation
        FROM news_events e
        WHERE e.status = 'open'
          AND (e.needs_reinterpretation
               OR NOT EXISTS (SELECT 1 FROM event_interpretations i WHERE i.event_id = e.id))
          -- never card an event whose articles S4/S5 have not finished extracting: its
          -- card would have no facts to cite (the 17 zero-fact events, F27)
          AND NOT EXISTS (SELECT 1 FROM news_articles a
                          WHERE a.event_id = e.id AND a.status = 'kept'
                            AND a.facts_extracted_at IS NULL)
          {where}
        ORDER BY e.last_updated DESC
    """, params)


def _load_facts(config: dict, event_id: int) -> list[dict]:
    # claim_context (the source sentence) lets the model judge whether a fact really
    # evidences a hop; conflicting_values carries every per-article value of a
    # discrepancy_flagged fact (F28, F34). Without them the model saw only the first value.
    rows = db.fetch_all(config, """
        SELECT id AS fact_id, metric, value, unit, as_of_period, claim_context, category,
               entities, fact_type, persistence_tier, verification_status, confidence,
               supporting_article_ids, conflicting_values
        FROM extracted_facts WHERE event_id = %s ORDER BY id
    """, (event_id,))
    for r in rows:
        if r.get("conflicting_values") is None:
            r.pop("conflicting_values", None)
    return rows


# Entity slugs too common to rank KB relevance by (see UBIQUITOUS_TOKENS / F30, F32).
_UBIQUITOUS_ENTITIES = frozenset({"india", "global", "usa", "us", "world"})


def _load_entity_aliases(config: dict) -> dict[str, list[str]]:
    """canonical name -> aliases, for BOTH entity slugs and metric names (§5.3: a match
    may be "a dictionary-alias of" the hop's nodes). Metric aliases matter as much as
    entity ones: `import_duty_rate` cited on a "tariff" hop is a valid citation."""
    out: dict[str, list[str]] = {}
    for sql in ("SELECT canonical_slug AS name, aliases FROM entity_dictionary",
                "SELECT canonical_name AS name, aliases FROM metrics_dictionary"):
        try:
            for r in db.fetch_all(config, sql):
                out.setdefault(r["name"], []).extend(r.get("aliases") or [])
        except Exception as exc:  # noqa: BLE001 - dictionary may not be seeded
            log.warning("S6 could not load dictionary aliases (%s) — continuing without", exc)
    return out


def _load_articles(config: dict, event_id: int) -> list[dict]:
    return db.fetch_all(config, """
        SELECT id AS article_id, source, published_at, title,
               LEFT(COALESCE(raw_text, rss_summary, ''), 900) AS summary
        FROM news_articles WHERE event_id = %s AND status='kept'
        ORDER BY published_at NULLS LAST, id
    """, (event_id,))


def _retrieve_kb_facts(config: dict, event_facts: list[dict],
                       event_id: int | None = None) -> list[dict]:
    """KB retrieval MUST filter to kb_durable — ephemeral prints are never retrieved.

    Ranked by overlap on SPECIFIC entities (not "india"), then recency, and excluding the
    event's own facts (already supplied as EVENT_FACTS). BUILD_NOTES F32: the previous
    `ORDER BY created_at` with an `entities && ...` filter matched on "india" alone, so
    the ten newest KB facts — unrelated pharma-tariff items — were retrieved for almost
    every event. (No embedding re-rank: the pipeline has no embeddings, D15.)
    """
    entities = sorted({e for f in event_facts for e in (f.get("entities") or [])}
                      - _UBIQUITOUS_ENTITIES)
    categories = sorted({f.get("category") for f in event_facts if f.get("category")})
    if not entities and not categories:
        return []
    try:
        return db.fetch_all(config, """
            SELECT id AS fact_id, metric, value, unit, as_of_period, claim_context, category,
                   entities, fact_type, verification_status, confidence
            FROM (
              SELECT *, cardinality(ARRAY(
                       SELECT unnest(entities) INTERSECT SELECT unnest(%s::text[]))) AS overlap
              FROM extracted_facts
              WHERE persistence_tier = 'kb_durable'
                AND event_id IS DISTINCT FROM %s
                AND (entities && %s::text[] OR category = ANY(%s))
            ) kb
            ORDER BY overlap DESC, created_at DESC
            LIMIT %s
        """, (entities, event_id, entities, categories, KB_RETRIEVAL_LIMIT))
    except Exception as exc:  # noqa: BLE001
        log.warning("S6 KB retrieval failed (%s) — proceeding with no KB facts", exc)
        return []


def _prior_version(config: dict, event_id: int) -> dict | None:
    return db.fetch_one(config, """
        SELECT version, payload FROM event_interpretations
        WHERE event_id = %s ORDER BY version DESC LIMIT 1
    """, (event_id,))


def _load_transient(config: dict, event_id: int) -> list[dict]:
    """Inline-evidence values S5 kept on the event (index closes, session moves). Never
    facts, never cited by fact_id (decision D1); stored only for this handoff (D15)."""
    row = db.fetch_one(config, "SELECT transient_context FROM news_events WHERE id = %s",
                       (event_id,))
    return list((row or {}).get("transient_context") or [])


# ---------------------------------------------------------------- inputs bundle
def pending_events(config: dict, event_ids: list[int] | None = None) -> list[dict]:
    """Open events that need a card (none yet, or needs_reinterpretation), and whose
    articles are fully extracted. Same selection for agent mode and API mode."""
    return _load_events(config, event_ids)


def build_inputs(config: dict, event: dict) -> dict:
    """Everything the interpreter sees for ONE event — identical in agent and API mode.

    Keys mirror the RUNTIME INPUTS placeholders of prompts/interpret_v1.txt.
    """
    from .s2_filter import primary_category_for_event_key

    facts = _load_facts(config, event["id"])
    prior = _prior_version(config, event["id"])
    return {
        "EVENT_META": {
            "event_id": event["id"], "event_key": event["event_key"], "title": event["title"],
            # Deterministic default from the event_type; the interpreter may override it
            # with the geopolitical rule but should prefer it unless content contradicts.
            "suggested_category": primary_category_for_event_key(event["event_key"]),
        },
        "ARTICLE_SUMMARIES": _load_articles(config, event["id"]),
        "EVENT_FACTS": facts,
        "KB_FACTS": _retrieve_kb_facts(config, facts, event_id=event["id"]),
        "PRIOR_INTERPRETATION": prior["payload"] if prior else None,
        "TRANSIENT_CONTEXT": _load_transient(config, event["id"]),
    }


def render_prompt(inputs: dict) -> tuple[str, str]:
    """(system, user) for API mode, from build_inputs()."""
    template = load_prompt("interpret_v1.txt")
    marker = "RUNTIME INPUTS\n======================================================================\n"
    system = template.split("======================================================================\nRUNTIME INPUTS")[0].strip()
    user = template.split(marker, 1)[-1]

    def j(o):
        return json.dumps(o, ensure_ascii=False, indent=2, default=str)

    user = (user
            .replace("{{EVENT_META}}", j(inputs["EVENT_META"]))
            .replace("{{ARTICLE_SUMMARIES}}", j(inputs["ARTICLE_SUMMARIES"]))
            .replace("{{EVENT_FACTS}}", j(inputs["EVENT_FACTS"]))
            .replace("{{KB_FACTS}}", j(inputs["KB_FACTS"]))
            .replace("{{PRIOR_INTERPRETATION}}",
                     j(inputs["PRIOR_INTERPRETATION"]) if inputs["PRIOR_INTERPRETATION"] else "null"))
    if inputs.get("TRANSIENT_CONTEXT"):
        user += ("\n\nTRANSIENT CONTEXT (never stored as facts, has no fact_id — cite inline "
                 "via \"evidence\", never via fact_ids):\n" + j(inputs["TRANSIENT_CONTEXT"]))
    return system, user.strip()


# ---------------------------------------------------------------- wording check
# Cards are internal conditional analysis, NOT advice (prompt rules 6c and the header).
# Found in production (30-Sep-2026): one-liners such as "it may be worth reviewing how much
# of your portfolio..." and reader-relative time ("so far", "this year"). Descriptive uses
# ("companies that sell to the US", "the RBI sold dollars") are NOT flagged.
_ADVICE = re.compile(
    r"\b(you should|we (would )?recommend|we suggest|consider (buying|selling|adding|"
    r"reducing|trimming|switching|exiting|increasing)|(it )?(may be|is|would be) worth "
    r"(reviewing|checking|considering|looking)|it makes sense to|time to (buy|sell|exit|"
    r"book|switch)|we can (now )?look at)\b", re.I)
_RELATIVE_TIME = re.compile(
    r"\b(today|yesterday|tomorrow|this week|last week|next week|this month|this year|"
    r"recently|so far|at present|as of now|right now)\b", re.I)


def lint_wording(payload: dict) -> list[dict]:
    """Advisory phrasing and reader-relative time references, with the field path."""
    issues: list[dict] = []

    def walk(o, path):
        if isinstance(o, str):
            for kind, rx in (("advice", _ADVICE), ("relative_time", _RELATIVE_TIME)):
                for m in rx.finditer(o):
                    issues.append({"kind": kind, "path": path, "text": m.group(0)})
        elif isinstance(o, dict):
            for k, v in o.items():
                if k != "footnotes":
                    walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(payload, "")
    return issues


# ---------------------------------------------------------------- the single write gate
class CardRejected(ValueError):
    """A card failed a gate and was NOT written. `.problems` says why."""

    def __init__(self, event_id: int, problems: list):
        super().__init__(f"event {event_id}: card rejected — {problems}")
        self.event_id, self.problems = event_id, problems


def prepare_card(event_id: int, event_key: str, payload: dict, inputs: dict,
                 aliases: dict, has_prior: bool, strict: bool = True) -> tuple[dict, list, list]:
    """Every content gate, with NO database access — the same code runs locally (DB mode)
    and in the Claude cloud routine (gateway mode, BUILD_NOTES D17).

    Gates, in order: is_update, category safety net, citation validator (event + KB facts,
    dictionary aliases; failing citations are stripped and recorded), JSON schema, wording
    check. strict=True (agent mode): schema errors or wording issues REJECT the card so the
    author fixes it. strict=False (API mode): schema errors reject; wording issues are
    recorded in citation_warnings (kind 'wording') for human review.
    Returns (payload, warnings, wording_issues).
    """
    import jsonschema

    from .s2_filter import PRIMARY_CATEGORIES, primary_category_for_event_key

    payload = json.loads(json.dumps(payload))            # never mutate the caller's copy
    if has_prior:
        payload["is_update"] = True

    # Category safety net: omitted / invalid / 'other' -> deterministic event_type map.
    pc = (payload.get("primary_category") or "").strip().lower()
    if pc not in PRIMARY_CATEGORIES or pc == "other":
        derived = primary_category_for_event_key(event_key or "")
        if derived != "other" or pc not in PRIMARY_CATEGORIES:
            payload["primary_category"] = derived

    # KB facts are citable per the prompt, so the validator must see them too (F26).
    facts_by_id = {f["fact_id"]: f for f in inputs.get("KB_FACTS") or []}
    facts_by_id.update({f["fact_id"]: f for f in inputs.get("EVENT_FACTS") or []})
    payload, warnings = validate_citations(payload, facts_by_id, aliases=aliases or {})

    schema_errors = [e.message[:200] for e in
                     jsonschema.Draft7Validator(INTERPRET_SCHEMA).iter_errors(payload)]
    if schema_errors:
        raise CardRejected(event_id, [{"kind": "schema", "error": e} for e in schema_errors])
    wording = lint_wording(payload)
    if wording and strict:
        raise CardRejected(event_id, wording)
    warnings += [{"kind": "wording", **w} for w in wording]
    return payload, warnings, wording


def _summary(event_id, version, payload, warnings, wording, dry) -> dict:
    return {"event_id": event_id, "version": version,
            "citations_stripped": sum(1 for w in warnings if "fact_id" in w),
            "wording_flags": len(wording), "chains": len(payload.get("impact_chains", [])),
            "dry_run": bool(dry)}


def write_card(config: dict, event_id: int, payload: dict, model_version: str,
               inputs: dict | None = None, strict: bool = True) -> dict:
    """DB mode: the ONLY path by which a card reaches event_interpretations locally / in CI.
    Content gates are prepare_card(); this adds the event-state checks and the insert.
    Never overwrites: always inserts version N+1 and clears needs_reinterpretation.
    """
    event = db.fetch_one(config, "SELECT id, event_key, title, status FROM news_events "
                                 "WHERE id = %s", (event_id,))
    if not event:
        raise CardRejected(event_id, ["event not found"])
    if event["status"] != "open":
        raise CardRejected(event_id, [f"event is {event['status']}, not open"])
    inputs = inputs or build_inputs(config, event)
    prior = db.fetch_one(config, "SELECT max(version) AS v FROM event_interpretations "
                                 "WHERE event_id = %s", (event_id,))
    prior_v = (prior or {}).get("v") or 0
    payload, warnings, wording = prepare_card(event_id, event["event_key"], payload, inputs,
                                              _load_entity_aliases(config), bool(prior_v), strict)
    version = prior_v + 1
    if not config.get("dry_run"):
        def _tx(cur):
            cur.execute("""
                INSERT INTO event_interpretations
                  (event_id, version, schema_version, payload, model_version, citation_warnings)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (event_id, version, "1.4", json.dumps(payload), model_version,
                  json.dumps(warnings) if warnings else None))
            cur.execute("UPDATE news_events SET needs_reinterpretation=false WHERE id=%s",
                        (event_id,))
        db.run_transaction(config, _tx)
    return _summary(event_id, version, payload, warnings, wording, config.get("dry_run"))


def write_card_via_gateway(gateway, event_id: int, payload: dict, model_version: str,
                           inputs: dict, aliases: dict, dry_run: bool = False) -> dict:
    """Gateway mode (Claude cloud routine): the SAME content gates as write_card, then the
    insert through s6_write_card over HTTPS, which re-checks the hard rules (event state,
    cited fact_ids, removed fields, URLs, wording) before inserting version N+1."""
    has_prior = inputs.get("PRIOR_INTERPRETATION") is not None
    payload, warnings, wording = prepare_card(
        event_id, (inputs.get("EVENT_META") or {}).get("event_key", ""), payload, inputs,
        aliases, has_prior, strict=True)
    version = None
    if not dry_run:
        from .s6_gateway import GatewayError
        try:
            version = gateway.write_card(event_id, payload, model_version, warnings)["version"]
        except GatewayError as exc:
            raise CardRejected(event_id, [{"kind": "gateway", "error": str(exc)}]) from None
    return _summary(event_id, version, payload, warnings, wording, dry_run)


# ---------------------------------------------------------------- API mode
def run(config: dict, event_ids: list[int] | None = None, **_inputs) -> dict:
    """API mode: the strong-tier model writes each pending card (spends API credits).

    Agent mode (Claude Code writes the cards, no API credits) uses the same
    pending_events / build_inputs / write_card functions via run_s6.py.
    """
    events = pending_events(config, event_ids)
    if not events:
        return {"stage": "s6_interpret", "mode": "api", "events": 0, "cards": 0}

    llm = get_llm(config, tier="strong")
    cards = failures = stripped_total = flagged = 0
    details = []
    for event in events:
        inputs = build_inputs(config, event)
        if not inputs["ARTICLE_SUMMARIES"]:
            continue
        system, user = render_prompt(inputs)
        try:
            res = llm.complete_json(system=system, user=user, json_schema=INTERPRET_SCHEMA,
                                    schema_name="interpretation", temperature=0.2,
                                    max_tokens=16000)
            out = write_card(config, event["id"], res.data, res.model, inputs, strict=False)
        except Exception as exc:  # noqa: BLE001 — one bad event must not kill the run
            failures += 1
            log.warning("S6 interpretation failed for event %s: %s", event["id"], exc)
            details.append({"event_id": event["id"], "error": str(exc)[:300]})
            continue
        cards += 1
        stripped_total += out["citations_stripped"]
        flagged += 1 if (out["citations_stripped"] or out["wording_flags"]) else 0
        details.append(out)

    return {
        "stage": "s6_interpret", "mode": "api",
        "events": len(events),
        "cards": cards,
        "failures": failures,
        "citations_stripped": stripped_total,
        "cards_flagged_for_review": flagged,
        "details": details,
    }

"""S4 — Extract (LLM call #1, per event).

Temperature 0, cheap model tier. Prompt is loaded from prompts/extract_v1.txt at runtime,
never hardcoded. JSON is enforced structurally (json_schema / tool-use), never by regex.

Emits TWO arrays per CLAUDE.md §5.1 rule 7 and decision D1:
  facts             -> persisted by S5
  transient_context -> index closes, index session % moves, single-stock daily % moves.
                       Used by S6 for narrative, NEVER written to extracted_facts.

Reprocessability (Prime Directive #6): S4 writes no rows itself — S5 persists. Because
news_articles.raw_text is stored and this stage runs at temperature 0, any event can be
re-extracted retroactively at any time.
"""
from __future__ import annotations

import json
import logging

from . import db
from .config import load_prompt
from .providers import get_llm

log = logging.getLogger(__name__)

FACT_PROPS = {
    "article_id": {"type": "integer"},
    "metric": {"type": "string"},
    "value": {"type": "string"},
    "unit": {"type": ["string", "null"]},
    "as_of_period": {"type": ["string", "null"]},
    "claim_context": {"type": "string"},
    "category": {
        "type": "string",
        "enum": ["macro_economics", "equity", "commodity", "forex",
                 "debt_rates", "policy_regulation", "sector_industry", "global_markets"],
    },
    "entities": {"type": "array", "items": {"type": "string"}},
    "fact_type": {"type": "string", "enum": ["point_in_time", "structural", "historical_pattern"]},
    "verification_status": {"type": "string", "enum": ["single_source"]},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "new_metric": {"type": "boolean"},
    "new_entity": {"type": "boolean"},
    "tier_override_reason": {"type": ["string", "null"]},
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": ["integer", "null"]},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **FACT_PROPS,
                    "persistence_tier": {"type": "string", "enum": ["event_context", "kb_durable"]},
                },
                "required": ["article_id", "metric", "value", "claim_context", "category",
                             "entities", "fact_type", "persistence_tier",
                             "verification_status", "confidence"],
            },
        },
        "transient_context": {
            "type": "array",
            "items": {"type": "object", "properties": FACT_PROPS,
                      "required": ["article_id", "metric", "value", "claim_context"]},
        },
    },
    "required": ["facts"],
}


def _load_dictionaries(config: dict) -> tuple[list[dict], list[dict]]:
    try:
        metrics = db.fetch_all(config, """
            SELECT canonical_name, aliases, category, default_persistence_tier
            FROM metrics_dictionary ORDER BY canonical_name
        """)
        entities = db.fetch_all(config, """
            SELECT canonical_slug, aliases FROM entity_dictionary ORDER BY canonical_slug
        """)
        return metrics, entities
    except Exception as exc:  # noqa: BLE001
        log.warning("S4 could not load dictionaries (%s) — proceeding with empty ones", exc)
        return [], []


def _load_event_bundles(config: dict, event_ids: list[int] | None,
                        reextract: bool = False) -> list[dict]:
    """Bundle contract (v1.1): article_id, source, url, published_at, title, body_or_summary.

    Scope (BUILD_NOTES F22): only OPEN events, and only kept articles not yet extracted
    (`facts_extracted_at IS NULL`, stamped by S5). Previously an uncapped call selected
    every event that had ever had a kept article — 1,069 vs 203 open in the Sep-2026
    backlog — re-extracting closed history and duplicating its facts. `reextract=True`
    deliberately re-processes all kept articles of the given events (Prime Directive #6).
    """
    clauses, params = ["a.status = 'kept'", "e.status = 'open'"], []
    if not reextract:
        clauses.append("a.facts_extracted_at IS NULL")
    if event_ids:
        clauses.append("e.id = ANY(%s)")
        params.append(list(event_ids))

    rows = db.fetch_all(config, f"""
        SELECT e.id AS event_id, e.event_key, e.title AS event_title,
               a.id AS article_id, a.source, a.url, a.published_at, a.title,
               COALESCE(a.raw_text, a.rss_summary, '') AS body_or_summary
        FROM news_events e
        JOIN news_articles a ON a.event_id = e.id
        WHERE {' AND '.join(clauses)}
        ORDER BY e.id, a.published_at NULLS LAST, a.id
    """, tuple(params))

    bundles: dict[int, dict] = {}
    for r in rows:
        b = bundles.setdefault(r["event_id"], {
            "event_id": r["event_id"], "event_key": r["event_key"],
            "title": r["event_title"], "articles": [],
        })
        if not r.get("url"):
            raise ValueError(
                f"Bundle contract violation: article {r['article_id']} has no url."
            )
        b["articles"].append({
            "article_id": r["article_id"],
            "source": r["source"],
            "url": r["url"],
            "published_at": r["published_at"].isoformat() if r.get("published_at") else None,
            "title": r["title"],
            "body_or_summary": r["body_or_summary"],
        })
    return list(bundles.values())


def _render_prompt(bundle: dict, metrics: list[dict], entities: list[dict]) -> tuple[str, str]:
    template = load_prompt("extract_v1.txt")
    system, _, _ = template.partition("======================================================================\nRUNTIME INPUTS")

    user = template.split("RUNTIME INPUTS\n======================================================================\n", 1)[-1]
    user = (user
            .replace("{{EVENT_META}}", json.dumps(
                {"event_id": bundle["event_id"], "event_key": bundle["event_key"],
                 "title": bundle["title"]}, ensure_ascii=False, indent=2))
            .replace("{{METRICS_DICTIONARY}}", json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
            .replace("{{ENTITY_DICTIONARY}}", json.dumps(entities, ensure_ascii=False, indent=2, default=str))
            .replace("{{ARTICLES}}", json.dumps(bundle["articles"], ensure_ascii=False, indent=2)))
    return system.strip(), user.strip()


# F27: 8000 output tokens truncated 17 of 38 events (a 3-article bundle already hit it).
# 16000 stays under the Anthropic SDK's non-streaming ceiling (~21k tokens), and bundles
# are chunked so no single call has to carry a 40-article event.
EXTRACT_MAX_TOKENS = 16000
DEFAULT_ARTICLES_PER_CALL = 5


def _coerce_items(data: dict, key: str) -> list[dict]:
    """Return data[key] as a list of dicts (BUILD_NOTES F23).

    The model sometimes returns the array JSON-*encoded* as a string inside the tool input
    (observed: a 21,584-char string), sometimes followed by trailing text. Iterating that
    string yielded characters and crashed with "'str' object does not support item
    assignment". Decode the leading JSON value; drop any non-dict items with a warning.
    """
    v = data.get(key) or []
    if isinstance(v, str):
        try:
            v, _end = json.JSONDecoder().raw_decode(v.lstrip())
        except ValueError as exc:
            raise ValueError(f"{key} returned as an unparseable string: {exc}") from exc
    if not isinstance(v, list):
        raise ValueError(f"{key} is {type(v).__name__}, expected a list")
    items = [x for x in v if isinstance(x, dict)]
    if len(items) != len(v):
        log.warning("S4 dropped %d non-object %s item(s)", len(v) - len(items), key)
    return items


def _parse_day(text):
    """A calendar day from a canonical as_of_period ("2026-09-28", "2026-09-28 close",
    "asof:2026-09-17") or an ISO timestamp; None for months, quarters, FYs, ranges."""
    import re
    from datetime import date

    m = re.search(r"(?<![\d.])(\d{4})-(\d{2})-(\d{2})(?![\d.])", str(text or ""))
    if not m or ".." in str(text):
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _flag_post_publication_dates(facts: list[dict], articles: list[dict]) -> int:
    """A fact cannot be dated after its article was published (BUILD_NOTES F36: a
    "Friday" close resolved to 26-Sep, a Saturday after publication). Such facts are kept
    — extraction is source-only — but downgraded to confidence 'low' and marked."""
    published = {a["article_id"]: _parse_day(a.get("published_at")) for a in articles}
    flagged = 0
    for f in facts:
        day, pub = _parse_day(f.get("as_of_period")), published.get(f.get("article_id"))
        if day and pub and day > pub:
            f["confidence"] = "low"
            f["date_check"] = f"as_of_period {day} is after publication {pub}"
            flagged += 1
    if flagged:
        log.warning("S4 flagged %d fact(s) dated after their article's publication", flagged)
    return flagged


def _chunk_bundle(bundle: dict, size: int) -> list[dict]:
    arts = bundle["articles"]
    if size <= 0 or len(arts) <= size:
        return [bundle]
    return [{**bundle, "articles": arts[i:i + size]} for i in range(0, len(arts), size)]


def run(config: dict, event_ids: list[int] | None = None, bundles: list[dict] | None = None,
        dictionaries: tuple[list[dict], list[dict]] | None = None,
        reextract: bool = False, **_inputs) -> dict:
    """Extract facts for the given events.

    `bundles` may be injected (offline testing / the local no-DB pipeline).
    `dictionaries`, if given as (metrics, entities), is used INSTEAD of the DB lookup —
    this is what lets the local pipeline stay genuinely DB-free: without it, this stage
    would silently open a live DB connection just to read metrics_dictionary/
    entity_dictionary even when bundles were injected and no other DB access was wanted.
    """
    if bundles is None:
        bundles = _load_event_bundles(config, event_ids, reextract=reextract)
    if not bundles:
        # Must return the SAME shape as the normal path — a caller reading
        # result['transient_context'] should not crash just because there was no work.
        return {"stage": "s4_extract", "events": 0, "facts": 0,
                "transient_context": 0, "failures": 0,
                "new_metric_proposals": [], "new_entity_proposals": [],
                "events_out": []}

    metrics, entities = dictionaries if dictionaries is not None else _load_dictionaries(config)
    llm = get_llm(config, tier="cheap")

    per_call = int((config.get("extract") or {}).get("articles_per_call",
                                                      DEFAULT_ARTICLES_PER_CALL))
    results, total_facts, total_transient, failures = [], 0, 0, 0
    for bundle in bundles:
        # Large events are extracted in article chunks and merged back into ONE event
        # result, so S5 still consolidates across every article of the event (F27).
        facts: list[dict] = []
        transient: list[dict] = []
        model_version, error = None, None
        for chunk in _chunk_bundle(bundle, per_call):
            system, user = _render_prompt(chunk, metrics, entities)
            try:
                res = llm.complete_json(
                    system=system, user=user,
                    json_schema=EXTRACT_SCHEMA, schema_name="extraction",
                    temperature=0.0, max_tokens=EXTRACT_MAX_TOKENS,
                )
                facts.extend(_coerce_items(res.data, "facts"))
                transient.extend(_coerce_items(res.data, "transient_context"))
                model_version = res.model
            except Exception as exc:  # noqa: BLE001 - one bad event must not kill the run
                error = f"{type(exc).__name__}: {exc}"
                break

        article_ids = [a["article_id"] for a in bundle["articles"]]
        if error:
            # All-or-nothing per event: a partial extraction is NOT ingested, and its
            # articles stay un-stamped so the next run retries the whole event.
            failures += 1
            log.warning("S4 extraction failed for event %s: %s", bundle["event_id"], error)
            results.append({"event_id": bundle["event_id"], "article_ids": article_ids,
                            "facts": [], "transient_context": [], "error": error})
            continue

        # Extraction ALWAYS emits single_source; upgrades are S5's deterministic job.
        for f in facts:
            f["verification_status"] = "single_source"
        _flag_post_publication_dates(facts + transient, bundle["articles"])
        results.append({
            "event_id": bundle["event_id"],
            "article_ids": article_ids,
            "facts": facts,
            "transient_context": transient,
            "model_version": model_version,
        })
        total_facts += len(facts)
        total_transient += len(transient)

    new_metrics = sorted({f["metric"] for r in results for f in r["facts"] if f.get("new_metric")})
    new_entities = sorted({e for r in results for f in r["facts"]
                           if f.get("new_entity") for e in (f.get("entities") or [])})

    return {
        "stage": "s4_extract",
        "events": len(bundles),
        "facts": total_facts,
        "transient_context": total_transient,
        "failures": failures,
        # Surfaced for review, never auto-added to the dictionaries (Phase 2 DoD).
        "new_metric_proposals": new_metrics,
        "new_entity_proposals": new_entities,
        "events_out": results,
    }

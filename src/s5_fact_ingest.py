"""S5 — Fact-Ingest. PURE PYTHON. NO LLM.

Deterministic consolidation of the per-article facts S4 produced:

  1. Dedup key = canonical `metric` + EXACT `as_of_period`.
     Period match is evaluated BEFORE the value tolerance. Two observations of the same
     quantity on different days/bases are NOT the same fact and must never be merged.
     (Brent $86.85 on 15-Jul vs $85.6 over four sessions to 16-Jul — the regression case.)

  2. Within a dedup group, members are clustered by value (±2% after unit/scale
     normalisation, or exact string). Then:
       one value cluster, 2+ articles                 -> cross_source_consistent
       several clusters, each article ONE comparable
       numeric value, and they disagree               -> discrepancy_flagged, ALL values kept
       an article reports several values, or values
       are not comparable (text, different currency)  -> SEPARATE facts, never a discrepancy
     BUILD_NOTES F29: the metric+period key cannot tell subjects apart (every state's FDI,
     every company's revenue shared one key), so one article stating several values is
     describing several subjects — not contradicting itself.

  3. supporting_article_ids maintained on every fact.
  4. kb_candidate = (persistence_tier == 'kb_durable').
  5. Metrics whose tier is 'transient' (index closes, index session moves, single-stock
     daily moves) are NEVER written as facts — decision D1. They are kept on the event
     (news_events.transient_context) for S6's inline evidence only.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from . import db

log = logging.getLogger(__name__)

NUMERIC_TOLERANCE = 0.02  # ±2%
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- pure helpers
def parse_numeric(value: str) -> float | None:
    """Extract a comparable number. Handles '~85', '-3,062.27', '9.87%'. None if not numeric."""
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    m = _NUM.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def values_agree(a: str, b: str, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    """Exact string match, or numeric within tolerance (relative)."""
    if a is None or b is None:
        return False
    if str(a).strip().lower() == str(b).strip().lower():
        return True
    na, nb = parse_numeric(a), parse_numeric(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    denom = max(abs(na), abs(nb))
    if denom == 0:
        return False
    return abs(na - nb) / denom <= tolerance


def dedup_key(fact: dict) -> tuple[str, str]:
    """metric + EXACT as_of_period. Period is part of identity, never normalised away."""
    return (
        (fact.get("metric") or "").strip().lower(),
        (fact.get("as_of_period") or "").strip().lower(),
    )


# Scale words, longest first so "lakh crore" wins over "lakh"/"crore".
_SCALES = [("lakh crore", 1e12), ("trillion", 1e12), ("tn", 1e12), ("crore", 1e7),
           ("cr", 1e7), ("lakh", 1e5), ("billion", 1e9), ("bn", 1e9), ("million", 1e6),
           ("mn", 1e6), ("thousand", 1e3)]
_CURRENCIES = [("nz$", "nzd"), ("us$", "usd"), ("usd", "usd"), ("$", "usd"), ("₹", "inr"),
               ("rs", "inr"), ("inr", "inr"), ("€", "eur"), ("eur", "eur"), ("¥", "jpy"),
               ("jpy", "jpy"), ("aed", "aed"), ("nzd", "nzd")]


def normalise(value, unit=None) -> tuple[float | None, str | None]:
    """(number scaled to units, currency) from a value + unit. 24.51 billion and 2451
    crore both become 2.451e10; '$42.8 billion' carries currency 'usd'."""
    num = parse_numeric(value)
    if num is None:
        return None, None
    text = f" {value} {unit or ''} ".lower()
    for word, factor in _SCALES:
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text):
            num *= factor
            break
    currency = None
    for token, code in _CURRENCIES:
        pattern = re.escape(token) if not token.isalpha() else rf"(?<![a-z]){token}(?![a-z])"
        if re.search(pattern, text):
            currency = code
            break
    return num, currency


def _measure(unit) -> str:
    """What a unit measures once scale words and currency are removed: 'INR lakh crore'
    and 'INR crore' -> '' (both money); '%' -> '%'; 'bps' -> 'bps'; 'USD/bbl' -> '/bbl'."""
    text = f" {unit or ''} ".lower()
    for word, _ in _SCALES:
        text = re.sub(rf"(?<![a-z]){re.escape(word)}(?![a-z])", " ", text)
    for token, _ in _CURRENCIES:
        text = (text.replace(token, " ") if not token.isalpha()
                else re.sub(rf"(?<![a-z]){token}(?![a-z])", " ", text))
    text = text.replace("percent", "%").replace("per cent", "%")
    return re.sub(r"\s+", "", text)


def _comparable(a: dict, b: dict) -> bool:
    """Numbers are comparable only in the same currency AND the same measure — 25 bps (a
    hike size) and 5.5% (a rate level) are different quantities, not a conflict."""
    na, ca = normalise(a.get("value"), a.get("unit"))
    nb, cb = normalise(b.get("value"), b.get("unit"))
    if na is None or nb is None:
        return False
    if ca and cb and ca != cb:
        return False
    ma, mb = _measure(a.get("unit")), _measure(b.get("unit"))
    return not (a.get("unit") and b.get("unit") and ma != mb)


def _agree(a: dict, b: dict) -> bool:
    if str(a.get("value")).strip().lower() == str(b.get("value")).strip().lower():
        return True
    if not _comparable(a, b):
        return False
    na, _ = normalise(a.get("value"), a.get("unit"))
    nb, _ = normalise(b.get("value"), b.get("unit"))
    return values_agree(str(na), str(nb))


def _value_clusters(members: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for m in members:
        for c in clusters:
            if _agree(c[0], m):
                c.append(m)
                break
        else:
            clusters.append([m])
    return clusters


def _is_genuine_conflict(clusters: list[list[dict]]) -> bool:
    """Several clusters form ONE discrepant fact only if every article sits in exactly one
    cluster (each source states one value) and all cluster values are numerically
    comparable. Otherwise the group mixes subjects or incomparable statements."""
    seen: dict = {}
    for i, c in enumerate(clusters):
        for m in c:
            aid = m.get("article_id")
            if aid in seen and seen[aid] != i:
                return False
            seen[aid] = i
    reps = [c[0] for c in clusters]
    return all(_comparable(reps[0], r) for r in reps[1:])


def consolidate(facts: list[dict]) -> list[dict]:
    """Group per-article facts into stored facts with deterministic verification_status."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in facts:
        groups.setdefault(dedup_key(f), []).append(f)

    out: list[dict] = []
    for _key, group in groups.items():
        clusters = _value_clusters(group)
        if len(clusters) > 1 and not _is_genuine_conflict(clusters):
            units = clusters                  # distinct subjects -> one stored fact each
        else:
            units = [group]
        for members in units:
            out.append(_consolidated(members))
    return out


def _consolidated(members: list[dict]) -> dict:
    article_ids = sorted({m["article_id"] for m in members if m.get("article_id") is not None})
    first = members[0]
    if len(article_ids) >= 2:
        all_agree = all(_agree(first, m) for m in members[1:])
        status = "cross_source_consistent" if all_agree else "discrepancy_flagged"
    else:
        status = "single_source"

    tier = first.get("persistence_tier") or "event_context"
    consolidated = {
        "metric": first.get("metric"),
        "value": first.get("value"),
        "unit": first.get("unit"),
        "as_of_period": first.get("as_of_period"),
        "claim_context": first.get("claim_context"),
        "category": first.get("category"),
        "entities": first.get("entities") or [],
        "fact_type": first.get("fact_type"),
        "persistence_tier": tier,
        "verification_status": status,
        "confidence": first.get("confidence"),
        "supporting_article_ids": article_ids,
        "kb_candidate": tier == "kb_durable",
        "conflicting_values": None,
    }
    if status == "discrepancy_flagged":
        # Retain EVERY value with per-article provenance. Never silently pick one.
        consolidated["conflicting_values"] = [
            {"article_id": m.get("article_id"), "value": m.get("value"),
             "unit": m.get("unit")} for m in members
        ]
    return consolidated


def split_transient(facts: list[dict], transient_metrics: set[str]) -> tuple[list[dict], list[dict]]:
    """Separate never-stored metrics (D1) from persistable facts."""
    keep, transient = [], []
    for f in facts:
        if (f.get("metric") or "").strip().lower() in transient_metrics:
            transient.append(f)
        else:
            keep.append(f)
    return keep, transient


# ---------------------------------------------------------------- stage
INSERT_SQL = """
INSERT INTO extracted_facts
  (event_id, metric, value, unit, as_of_period, claim_context, category, entities,
   fact_type, persistence_tier, verification_status, confidence,
   supporting_article_ids, kb_candidate, conflicting_values)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

STAMP_SQL = "UPDATE news_articles SET facts_extracted_at = now() WHERE id = ANY(%s)"

# Transient context (index closes, index/stock session moves) is NEVER a fact (decision
# D1): it has no fact_id and can never be cited as one. It is kept on the EVENT only so a
# separately-run S6 can use it as inline hop evidence (BUILD_NOTES D15). Appended, so a
# later run that extracts only newly arrived articles keeps the earlier context.
TRANSIENT_SQL = """
UPDATE news_events
SET transient_context = COALESCE(transient_context, '[]'::jsonb) || %s::jsonb
WHERE id = %s
"""


def _write_event(config: dict, event_id, consolidated: list[dict], article_ids: list,
                 transient: list[dict] | None = None) -> None:
    """One transaction per event: its facts, its transient context, AND the extraction
    stamp on its articles (F31). A drop mid-way rolls everything back, so a retry cannot
    duplicate facts, and articles are only marked extracted once their facts are durably
    stored (F22)."""
    import json as _json

    import psycopg2.extras

    rows = [(event_id, f["metric"], f["value"], f["unit"], f["as_of_period"],
             f["claim_context"], f["category"], f["entities"], f["fact_type"],
             f["persistence_tier"], f["verification_status"], f["confidence"],
             f["supporting_article_ids"], f["kb_candidate"],
             _json.dumps(f["conflicting_values"]) if f.get("conflicting_values") else None)
            for f in consolidated]

    def _tx(cur):
        if rows:
            psycopg2.extras.execute_batch(cur, INSERT_SQL, rows, page_size=100)
        if transient:
            cur.execute(TRANSIENT_SQL, (_json.dumps(transient, default=str), event_id))
        if article_ids:
            cur.execute(STAMP_SQL, (list(article_ids),))

    db.run_transaction(config, _tx)


def _load_transient_metrics(config: dict) -> set[str]:
    try:
        rows = db.fetch_all(
            config,
            "SELECT canonical_name FROM metrics_dictionary WHERE default_persistence_tier='transient'",
        )
        return {r["canonical_name"].strip().lower() for r in rows}
    except Exception:  # noqa: BLE001 - dictionary may not be seeded yet
        return set()


def run(config: dict, extraction: dict | None = None, **_inputs) -> dict:
    """`extraction` is S4's output: {"events": [{event_id, facts, transient_context}]}."""
    if not extraction or not extraction.get("events"):
        # Same shape as the normal path (see s4_extract early-return note).
        return {"stage": "s5_fact_ingest", "events": 0, "facts_in": 0,
                "facts_consolidated": 0, "facts_written": 0,
                "cross_source_upgrades": 0, "discrepancies_flagged": 0,
                "kb_candidates": 0, "skipped_failed_extraction": [], "per_event": []}

    transient_metrics = _load_transient_metrics(config)
    total_in = total_out = written = 0
    upgrades = discrepancies = kb_candidates = 0
    per_event = []

    skipped_failed = []
    for ev in extraction["events"]:
        event_id = ev.get("event_id")
        if ev.get("error"):
            # S4 failed (or partially failed) this event: ingest nothing and leave its
            # articles un-stamped so the next run re-extracts the whole event.
            skipped_failed.append(event_id)
            continue
        raw_facts = ev.get("facts", [])
        total_in += len(raw_facts)

        persistable, dropped_transient = split_transient(raw_facts, transient_metrics)
        # Drop facts the extractor emitted without a value or metric. The schema marks both
        # required, but the model occasionally returns value=null (e.g. a "gst_rate" fact
        # with no number). A fact without a value is not a fact — and `value`/`metric` are
        # NOT NULL in extracted_facts, so inserting one crashes the whole event. Precision-
        # first: drop it rather than persist a hollow row or fail the run.
        valued = [f for f in persistable
                  if (f.get("value") is not None and str(f.get("value")).strip()
                      and (f.get("metric") or "").strip())]
        dropped_valueless = len(persistable) - len(valued)
        if dropped_valueless:
            log.info("S5 event %s: dropped %d fact(s) with null/empty value or metric",
                     event_id, dropped_valueless)
        consolidated = consolidate(valued)
        total_out += len(consolidated)

        upgrades += sum(1 for f in consolidated if f["verification_status"] == "cross_source_consistent")
        discrepancies += sum(1 for f in consolidated if f["verification_status"] == "discrepancy_flagged")
        kb_candidates += sum(1 for f in consolidated if f["kb_candidate"])

        transient = (ev.get("transient_context") or []) + dropped_transient
        if not config.get("dry_run"):
            _write_event(config, event_id, consolidated, ev.get("article_ids") or [],
                         transient)
            written += len(consolidated)

        per_event.append({
            "event_id": event_id,
            "facts_in": len(raw_facts),
            "facts_stored": len(consolidated),
            "transient_dropped": len(dropped_transient),
            # stored on news_events.transient_context for S6 — never as a fact (D1)
            "transient_context": transient,
        })

    return {
        "stage": "s5_fact_ingest",
        "events": len(extraction["events"]),
        "facts_in": total_in,
        "facts_consolidated": total_out,
        "facts_written": written,
        "cross_source_upgrades": upgrades,
        "discrepancies_flagged": discrepancies,
        "kb_candidates": kb_candidates,
        "skipped_failed_extraction": skipped_failed,
        "per_event": per_event,
    }

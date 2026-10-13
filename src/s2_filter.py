"""S2 — Filter. Two passes, precision-first.

Pass A (deterministic, free): source blacklist, section blacklist, near-duplicate titles.
Pass B (cheap LLM triage): score 0-10 on relevance to Indian wealth-management portfolios,
and propose an event_key slug.

Hard rule from CLAUDE.md §2: score < threshold (7) is DROPPED. Dropped articles keep their
row with status='dropped' and a drop_reason — nothing is deleted (Prime Directive #6).

Precision over recall: when the model is unsure, it is instructed to score low.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from . import db
from .providers import get_llm

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")

TRIAGE_SYSTEM = """You are the triage filter for an Indian private wealth management research
pipeline. The pipeline has TWO purposes, and you must score each article against BOTH.

Judge every article on two INDEPENDENT dimensions. Give a one-line reason BEFORE each
number — derive the score, do not guess it.

=====================================================================
DIMENSION 1 — portfolio_relevance (0-10): does this move client money NOW?
=====================================================================
 10  RBI cuts the repo rate; a major macro print lands far off expectations
  9  Crude spikes 14% on supply disruption; rupee breaks a psychological level
  8  A large-cap posts results with clear sector read-across
  7  Sector margin or input-cost news; global yields move enough to shift FPI flows;
     a NATIONAL macro print at a multi-month extreme (e.g. "activity hits a 32-month
     high"); a trade / tariff action by a major partner that hits a whole Indian sector
     (e.g. "US tariff threat drags pharma") — sector-wide read-across IS portfolio-moving
  5  A mid-cap moves on company-specific news with little read-across
  3  A foreign bank changes staffing; an overseas exchange changes trading hours
  1  Sport, entertainment, lifestyle, celebrity, local crime

CALIBRATION — do NOT under-score these (they are 7, not 6):
  - A national macro indicator printing at a multi-month high/low is a 7. It is not a
    routine company number; it shifts the read on growth and policy.
  - A tariff / trade-policy move by the US, EU, China or a big partner that names an
    Indian SECTOR (pharma, IT, autos, metals) is a 7 — the read-across moves a basket,
    not one stock. Score the single-stock reaction lower, but the sector action is a 7.
  - "Large private/PSU bank Q1 with a big profit swing and clear sector read" is an 8,
    not a 6 — a top-5 bank's results are read across the whole financials basket.
  A borderline call between 6 and 7 on a genuinely market-relevant, non-company-specific
  story should resolve to 7. Precision-first applies to NOISE (single stocks, corporate
  actions, market chatter), NOT to real macro / policy / sector signals.

=====================================================================
DIMENSION 2 — kb_value (0-10): does this contain DURABLE structural knowledge?
=====================================================================
Durable means it stays true, or stays useful, long after today. This is how the firm's
knowledge base is built, so undramatic facts can score very high here.
 10  "India imports ~85% of its crude"; "a fifth of global oil transits Hormuz"
  9  India signs long-term LNG supply contracts; a new import-duty structure
  8  Sector cost structures; index composition changes; regulatory frameworks
  7  A historical pattern with numbers ("in past crude spikes, Nifty fell ~2%")
  4  A quarterly earnings number for one company
  2  Today's index close; today's FII flow figure
  0  Pure market commentary with no structural content

CRITICAL: these two dimensions are genuinely different. An article can be a 5 on
portfolio_relevance and a 9 on kb_value — a long-term supply contract does not move
prices this week but is exactly the durable knowledge this system exists to accumulate.
Do NOT let one dimension drag the other toward the middle. Score them separately.

USE THE FULL RANGE. If every article you score lands on 7 or 8, you are not
discriminating. Genuine 10s and genuine 2s both exist.

PRECISION: for portfolio_relevance, when unsure score LOW — noise is worse than a miss.
For kb_value, ask only "would a research analyst want this fact on file in a year?"

=====================================================================
EVENT TYPE — pick from the controlled list, never invent one
=====================================================================
You will be given EVENT_TAXONOMY. Choose the single best-fitting `slug`.
Rules that matter:
  - ALL individual company results -> `india-earnings-season`. Do NOT create per-company
    types. TVS Q1, Adani Q1 and a results roundup all get the same type.
  - Index-close and session commentary -> `market-daily-wrap`, one per day.
  - Long-term supply contracts, capacity and infrastructure -> `sector-development`.
  - Use `uncategorised` only when nothing fits.
The event_key is assembled downstream from your event_type — you supply only the type."""

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "integer"},
                    "portfolio_reason": {"type": "string"},
                    "portfolio_relevance": {"type": "number"},
                    "kb_reason": {"type": "string"},
                    "kb_value": {"type": "number"},
                    "event_type": {"type": "string"},
                },
                "required": ["article_id", "portfolio_relevance", "kb_value", "event_type"],
            },
        }
    },
    "required": ["results"],
}


# ---------------------------------------------------------------- pure helpers
def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace — for hashing and fuzzy match."""
    t = _PUNCT.sub(" ", (title or "").lower())
    return _WS.sub(" ", t).strip()


def title_hash(title: str) -> str:
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def is_near_duplicate(title: str, seen_titles: list[str], threshold: float = 0.90) -> bool:
    """Exact normalized match, or fuzzy ratio >= threshold against anything already kept."""
    norm = normalize_title(title)
    for other in seen_titles:
        if norm == normalize_title(other):
            return True
        if title_similarity(title, other) >= threshold:
            return True
    return False


def matches_section_blacklist(url: str, title: str, patterns: list[str]) -> str | None:
    """Return the matching pattern, or None."""
    hay = f"{url or ''} {title or ''}".lower()
    for pat in patterns or []:
        if pat.lower().strip("/") in hay:
            return pat
    return None


# ---------------------------------------------------------------- stage
def _pass_a(articles: list[dict], fcfg: dict) -> tuple[list[dict], list[tuple[int, str]]]:
    """Returns (survivors, [(article_id, drop_reason), ...])."""
    src_blacklist = {s.lower() for s in fcfg.get("source_blacklist", [])}
    sections = fcfg.get("section_blacklist", [])
    dup_ratio = float(fcfg.get("duplicate_title_ratio", 0.90))

    survivors: list[dict] = []
    dropped: list[tuple[int, str]] = []
    kept_titles: list[str] = []

    for art in articles:
        if (art.get("source") or "").lower() in src_blacklist:
            dropped.append((art["id"], "pass_a:source_blacklist"))
            continue

        hit = matches_section_blacklist(art.get("url", ""), art.get("title", ""), sections)
        if hit:
            dropped.append((art["id"], f"pass_a:section_blacklist:{hit}"))
            continue

        if is_near_duplicate(art.get("title", ""), kept_titles, dup_ratio):
            dropped.append((art["id"], "pass_a:near_duplicate_title"))
            continue

        kept_titles.append(art.get("title", ""))
        survivors.append(art)

    return survivors, dropped


# ---------------------------------------------------------------- primary category index
# The SIX top-level buckets the UI segregates events by (round 7, Rushabh). Exactly ONE per
# event, assigned by the NATURE of the event (not its impact — impact is multi-category).
# Single source of truth = dictionaries/event_taxonomy.yaml (card_category_of_event_type,
# card_category_labels). S6 assigns primary_category on the card (so it can apply the
# "global_economy if geopolitically triggered" override with full context); this map is the
# deterministic DEFAULT/fallback rollup of the event_type slugs.
PRIMARY_CATEGORIES = ["global_economy", "global_equity", "indian_economy",
                      "indian_equity_corporate", "commodity", "regulatory", "other"]

PRIMARY_CATEGORY_LABELS = {
    "global_economy": "Global Economy",
    "global_equity": "Global Equity",
    "indian_economy": "Indian Economy",
    "indian_equity_corporate": "Indian Equity and Corporates",
    "commodity": "Commodity",
    "regulatory": "Regulatory",
    "other": "Other",
}

_CARD_CAT_CACHE: dict | None = None


def _load_card_category_map() -> dict:
    """Load event_type -> card_category from the taxonomy YAML (cached, single source)."""
    global _CARD_CAT_CACHE
    if _CARD_CAT_CACHE is None:
        import yaml

        from .config import REPO_ROOT

        path = REPO_ROOT / "dictionaries" / "event_taxonomy.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        _CARD_CAT_CACHE = {str(k).strip().lower(): v
                           for k, v in (data.get("card_category_of_event_type") or {}).items()}
    return _CARD_CAT_CACHE


def primary_category_for(event_type: str) -> str:
    """Deterministic default primary bucket for an event_type slug (fallback when S6 did
    not assign one, or returned 'other'/invalid). Unknown types -> 'other'."""
    return _load_card_category_map().get((event_type or "").strip().lower(), "other")


def primary_category_for_event_key(event_key: str) -> str:
    """Resolve the primary bucket from an event_key by longest-prefix matching the known
    event_type slugs (event_key = '<event_type>-<period>', so a raw rsplit is unreliable
    for multi-hyphen types like 'trade-and-external' or 'india-earnings-season')."""
    m = _load_card_category_map()
    ek = (event_key or "").strip().lower()
    best = ""
    for slug in m:
        if (ek == slug or ek.startswith(slug + "-")) and len(slug) > len(best):
            best = slug
    return m.get(best, "other")


def load_taxonomy() -> list[dict]:
    """Controlled event vocabulary. Triage picks a slug; it may not invent one."""
    import yaml

    from .config import REPO_ROOT

    path = REPO_ROOT / "dictionaries" / "event_taxonomy.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("event_types", [])


def reporting_quarter(d: datetime) -> tuple[int, int]:
    """Most-recently-COMPLETED Indian fiscal quarter as of date d -> (quarter, FY end year).

    Earnings for a quarter are announced in the following ~6-7 weeks, so an earnings season
    must be keyed by the quarter that just ENDED, not the calendar quarter of the run date.
    Indian FY runs Apr-Mar; FY27 = Apr 2026 - Mar 2027, with:
      Q1 Apr-Jun, Q2 Jul-Sep, Q3 Oct-Dec, Q4 Jan-Mar.
    Example: a run on 21-Jul-2026 must yield Q1FY27 (Apr-Jun 2026 just ended), NOT the
    calendar Q3. This groups the whole Jul-Aug Q1 announcement window under one key.
    """
    m, y = d.month, d.year
    if 4 <= m <= 6:     # Apr-Jun: last completed is Q4 of the FY that just ended in March
        return 4, y
    if 7 <= m <= 9:     # Jul-Sep: last completed is Q1 of the current FY (ends next March)
        return 1, y + 1
    if 10 <= m <= 12:   # Oct-Dec: last completed is Q2
        return 2, y + 1
    return 3, y         # Jan-Mar: last completed is Q3 (Oct-Dec of the prior calendar year)


def period_suffix(period: str, when: datetime | None = None) -> str:
    """Deterministic period token so the same event_type collapses to one key per period."""
    d = when or datetime.now(timezone.utc)
    if period == "daily":
        return d.strftime("%d%b%Y").lower()
    if period == "quarterly":
        q, fy = reporting_quarter(d)
        return f"q{q}fy{str(fy)[-2:]}"
    return d.strftime("%b%Y").lower()       # monthly and rolling


def build_event_key(event_type: str, taxonomy: list[dict], when: datetime | None = None) -> str:
    """event_key = <event_type>-<period_suffix>. Unknown types fall back to uncategorised."""
    by_slug = {t["slug"]: t for t in taxonomy}
    slug = (event_type or "").strip().lower()
    entry = by_slug.get(slug)
    if entry is None:
        slug, entry = "uncategorised", by_slug.get("uncategorised", {"period": "daily"})
    return f"{slug}-{period_suffix(entry.get('period', 'rolling'), when)}"


def _pass_b(config: dict, articles: list[dict]) -> tuple[dict[int, dict], list[int]]:
    """Two-dimensional LLM triage (decision: Option A, round 6).

    Returns (scores, failed_ids) on EVERY code path:
      scores     {article_id: {portfolio_relevance, kb_value, score, event_type, ...}}
                 where `score` = max(portfolio_relevance, kb_value) — an article is kept if
                 EITHER dimension clears the bar, so durable-but-undramatic facts survive.
      failed_ids articles that could not be scored due to an INFRASTRUCTURE error (TLS,
                 rate limit). These are held for retry, never treated as a relevance verdict.
    """
    if not articles:
        return {}, []

    llm = get_llm(config, tier="cheap")
    taxonomy = load_taxonomy()
    tax_brief = [{"slug": t["slug"], "description": t.get("description", "").strip()}
                 for t in taxonomy]

    out: dict[int, dict] = {}
    failed_ids: list[int] = []
    # Smaller batches return faster and fail smaller. With two-dimensional scoring each
    # article now emits two reasons plus two numbers, so a 20-article batch produces a
    # long response and a long wall-clock call. 10 keeps each call brisk and keeps a
    # truncated or failed batch small (it is held for retry, never dropped).
    batch_size = int((config.get("filter") or {}).get("triage_batch_size", 10))

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        payload = [
            {
                "article_id": a["id"],
                "source": a.get("source"),
                "title": a.get("title"),
                "summary": (a.get("rss_summary") or (a.get("raw_text") or "")[:600]),
            }
            for a in batch
        ]
        user = ("EVENT_TAXONOMY (choose event_type from these slugs only):\n"
                + _as_json(tax_brief)
                + "\n\nScore every article below on BOTH dimensions.\n\n"
                + _as_json(payload))
        try:
            res = llm.complete_json(
                system=TRIAGE_SYSTEM, user=user,
                json_schema=TRIAGE_SCHEMA, schema_name="triage",
                temperature=0.0, max_tokens=8000,
            )
            for r in res.data.get("results", []):
                pr = float(r.get("portfolio_relevance", 0) or 0)
                kb = float(r.get("kb_value", 0) or 0)
                etype = (r.get("event_type") or "uncategorised").strip().lower()
                out[int(r["article_id"])] = {
                    "portfolio_relevance": pr,
                    "kb_value": kb,
                    "score": max(pr, kb),
                    "event_type": etype,
                    "event_key": build_event_key(etype, taxonomy),
                    "portfolio_reason": r.get("portfolio_reason", ""),
                    "kb_reason": r.get("kb_reason", ""),
                }
        except Exception as exc:  # noqa: BLE001
            # An infrastructure failure is NOT a relevance judgement. These articles are
            # left unscored so run() can hold them at status='new' for the next run,
            # rather than discarding real news because of a TLS blip or rate limit.
            failed_ids.extend(a["id"] for a in batch)
            log.error(
                "S2 triage batch failed (%s: %s) — %d articles HELD for retry, not dropped",
                type(exc).__name__, str(exc)[:160], len(batch),
            )

    return out, failed_ids


def _as_json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def run(config: dict, limit: int | None = None, **_inputs) -> dict:
    """`limit`, if given, caps how many 'new' articles this call processes (oldest-first
    is NOT implied — ORDER BY below is unchanged). Optional and additive: default None
    preserves the original "process everything new" behaviour for the scheduled cron run.
    Added so a large one-off backlog (e.g. a bulk-pushed snapshot) can be triaged in
    bounded chunks across several calls instead of one long-running call — each chunk's
    articles flip out of status='new' as they're processed, so calling run() again with
    the same or no limit naturally resumes with whatever is left, never reprocessing.
    """
    fcfg = config.get("filter", {})
    threshold = float(fcfg.get("triage_threshold", 7))

    sql = """
        SELECT id, url, source, title, rss_summary, raw_text
        FROM news_articles
        WHERE status = 'new'
        ORDER BY published_at DESC NULLS LAST, id DESC
        """
    params: tuple = ()
    if limit:
        sql += " LIMIT %s"
        params = (int(limit),)
    articles = db.fetch_all(config, sql, params)
    if not articles:
        return {"stage": "s2_filter", "considered": 0, "kept": 0, "dropped": 0}

    survivors, dropped_a = _pass_a(articles, fcfg)
    scores, failed_ids = _pass_b(config, survivors)
    held = set(failed_ids)
    if survivors and len(held) == len(survivors):
        # BUILD_NOTES F33: an invalid API key failed EVERY batch from 18-Sep, each run
        # "succeeded" with everything held, and 1,385 articles piled up unnoticed for 12
        # days. Nothing is written here, so failing is safe; it turns CI red.
        raise RuntimeError(
            f"S2 triage failed for ALL {len(survivors)} article(s) — every batch errored "
            f"(check the LLM API key / provider). Articles left at status='new' for retry."
        )

    kept, dropped_b = [], []
    for art in survivors:
        info = scores.get(art["id"])
        if info is None:
            if art["id"] in held:
                # Infrastructure failure, not a judgement. Leave status='new' so the
                # next run retries it. Never discard real news over a transient fault.
                continue
            dropped_b.append((art["id"], "pass_b:unscored", None, None))
            continue
        # KEEP if EITHER dimension clears the bar (score = max of the two).
        if info["score"] < threshold:
            reason = (f"pass_b:portfolio_{info['portfolio_relevance']:g}"
                      f"_kb_{info['kb_value']:g}")
            dropped_b.append((art["id"], reason, info["score"], info["event_key"]))
        else:
            kept.append((art["id"], info["score"], info["event_key"]))

    if not config.get("dry_run"):
        db.execute_many(
            config,
            "UPDATE news_articles SET status='dropped', drop_reason=%s WHERE id=%s",
            [(reason, aid) for aid, reason in dropped_a],
        )
        db.execute_many(
            config,
            """UPDATE news_articles
               SET status='dropped', drop_reason=%s, triage_score=%s, event_key=%s
               WHERE id=%s""",
            [(reason, score, ekey, aid) for aid, reason, score, ekey in dropped_b],
        )
        db.execute_many(
            config,
            """UPDATE news_articles
               SET status='kept', triage_score=%s, event_key=%s
               WHERE id=%s""",
            [(score, ekey, aid) for aid, score, ekey in kept],
        )

    result = {
        "stage": "s2_filter",
        "considered": len(articles),
        "pass_a_dropped": len(dropped_a),
        "pass_b_dropped": len(dropped_b),
        "kept": len(kept),
        "dropped": len(dropped_a) + len(dropped_b),
        "held_for_retry": len(held),
        "threshold": threshold,
    }
    if held:
        # Surface loudly. A run that silently loses half its articles to a TLS error
        # must not look like a clean run.
        result["warning"] = (
            f"{len(held)} article(s) could not be scored due to an infrastructure error "
            f"and were HELD at status='new' for the next run. Investigate before trusting "
            f"this run's coverage."
        )
        log.error(result["warning"])
    return result

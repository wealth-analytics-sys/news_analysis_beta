"""S3 — Cluster. Articles -> events.

  - kept, unclustered articles are bucketed by their triage event_type
  - ROLLUP types (earnings, market wraps) become one event per bucket (deterministic)
  - other buckets are split into distinct stories by the LLM (prompts/cluster_v1.txt),
    which may also attach a story to an OPEN event from the lookback window
  - a catalyst-merge pass (prompts/catalyst_merge_v1.txt) fuses events sharing one driver
  - if a bucket's LLM call fails, its articles become singleton events (precision-first)
  - events whose newest NEWS is older than auto_close_days are auto-closed

No embeddings (removed 30-Sep-2026, BUILD_NOTES D15): they only ever served the failure
fallback, yet every run had to embed every article first.

Bundle contract (CLAUDE.md §3, v1.1): every article in the handoff bundle carries
article_id, source, url, published_at, title, body_or_summary. Omitting url is a
contract violation — build_bundle() asserts it.
"""
from __future__ import annotations

import json
import logging
import re

from . import db
from .config import load_prompt
from .providers import get_llm
from .s2_filter import build_event_key, load_taxonomy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- pure helpers
def build_bundle(event: dict, articles: list[dict]) -> dict:
    """The contract handed to S4/S6. Raises if url is missing (contract violation)."""
    out_articles = []
    for a in articles:
        if not a.get("url"):
            raise ValueError(
                f"Bundle contract violation: article {a.get('id')} has no url. "
                "url must propagate from news_articles through every downstream bundle."
            )
        out_articles.append({
            "article_id": a["id"],
            "source": a.get("source"),
            "url": a["url"],
            "published_at": a["published_at"].isoformat() if a.get("published_at") else None,
            "title": a.get("title"),
            "body_or_summary": a.get("raw_text") or a.get("rss_summary") or "",
        })
    return {
        "event_id": event.get("id"),
        "event_key": event.get("event_key"),
        "title": event.get("title"),
        "articles": out_articles,
    }


# ---------------------------------------------------------------- LLM clustering
def _slugify(text: str, max_words: int = 5) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
    return "-".join(words[:max_words])[:48] or "event"


CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assign_to_event_id": {"type": ["integer", "null"]},
                    "event_type": {"type": "string"},
                    "title": {"type": "string"},
                    "member_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["event_type", "title", "member_ids"],
            },
        }
    },
    "required": ["groups"],
}


def llm_cluster(config: dict, articles: list[dict], open_events: list[dict],
                rollup_types: set[str]) -> list[dict] | None:
    """Group kept articles into distinct STORIES via the cheap-tier LLM.

    Returns a list of groups, each: {assign_to_event_id, event_type, title, members:[art]}
    with every input article assigned to exactly one group (unassigned articles are
    recovered as singleton groups, so coverage is guaranteed). Returns None only if the
    LLM output is unusable — the caller then falls back to deterministic clustering.
    """
    if not articles:
        return []

    payload = [{
        "id": a["id"],
        "event_type": a.get("event_key", "").rsplit("-", 1)[0] if a.get("event_key") else "",
        "source": a.get("source"),
        "title": a.get("title"),
        "summary": (a.get("rss_summary") or (a.get("raw_text") or ""))[:400],
    } for a in articles]
    # event_type on articles is embedded in event_key as "<type>-<period>"; recover the
    # type portion for the model. If unavailable, leave blank and let the model infer.
    for p, a in zip(payload, articles):
        ek = a.get("event_key") or ""
        p["event_type"] = _event_type_from_key(ek)

    template = load_prompt("cluster_v1.txt")
    marker = "RUNTIME INPUTS\n======================================================================\n"
    system = template.split("======================================================================\nRUNTIME INPUTS")[0].strip()
    user = template.split(marker, 1)[-1]
    user = (user
            .replace("{{ROLLUP_TYPES}}", json.dumps(sorted(rollup_types)))
            .replace("{{OPEN_EVENTS}}", json.dumps(open_events, ensure_ascii=False, default=str))
            .replace("{{ARTICLES}}", json.dumps(payload, ensure_ascii=False)))

    llm = get_llm(config, tier="cheap")
    res = llm.complete_json(system=system.strip(), user=user.strip(),
                            json_schema=CLUSTER_SCHEMA, schema_name="clustering",
                            temperature=0.0, max_tokens=4000)
    raw_groups = (res.data or {}).get("groups")
    if not raw_groups:
        return None

    by_id = {a["id"]: a for a in articles}
    valid_open = {e["event_id"] for e in open_events}
    seen: set[int] = set()
    groups: list[dict] = []
    for g in raw_groups:
        members = [by_id[i] for i in g.get("member_ids", []) if i in by_id and i not in seen]
        if not members:
            continue
        seen.update(m["id"] for m in members)
        aid = g.get("assign_to_event_id")
        groups.append({
            "assign_to_event_id": aid if aid in valid_open else None,
            "event_type": (g.get("event_type") or "").strip().lower(),
            "title": (g.get("title") or members[0].get("title") or "").strip(),
            "members": members,
        })

    # Coverage guarantee: any article the model forgot becomes its own singleton event.
    missed = [a for a in articles if a["id"] not in seen]
    for a in missed:
        groups.append({"assign_to_event_id": None,
                       "event_type": _event_type_from_key(a.get("event_key") or ""),
                       "title": a.get("title") or "Untitled", "members": [a]})
    if missed:
        log.warning("S3 LLM clustering omitted %d article(s) — recovered as singletons",
                    len(missed))
    return groups


def _event_type_from_key(event_key: str) -> str:
    """event_key is '<event_type>-<period_suffix>'; recover the type portion.

    period_suffix is one trailing token (ddMMMyyyy / MMMyyyy / qNfyYY), so the type is
    everything before the last hyphen. 'trade-and-external-jul2026' -> 'trade-and-external'.
    """
    return event_key.rsplit("-", 1)[0] if "-" in event_key else event_key


def _load_open_events_brief(config: dict, lookback_days: int) -> list[dict]:
    """Open events from the lookback window as (event_id, event_type, title) for the
    clustering LLM to optionally attach ongoing stories to."""
    rows = db.fetch_all(config, """
        SELECT id AS event_id, event_key, title
        FROM news_events
        WHERE status='open' AND last_updated >= now() - (%s || ' days')::interval
        ORDER BY last_updated DESC
    """, (str(lookback_days),))
    return [{"event_id": r["event_id"], "event_type": _event_type_from_key(r["event_key"]),
             "title": r["title"]} for r in rows]


def _news_time(members: list[dict]):
    """Latest NEWS time of the members (published_at, else ingested_at via cluster_time).

    BUILD_NOTES F35: last_updated drives the 7-day auto-close. It used to be now() at
    processing time, so a backlog event built from 12-day-old articles stayed open a full
    week after it was processed. Keying it to news time closes stale news on schedule.
    Returns None when no member carries a time, so callers fall back to now().
    """
    times = [a.get("cluster_time") or a.get("published_at") for a in members]
    times = [t for t in times if t is not None]
    return max(times) if times else None


def _new_event(config: dict, event_key: str, title: str, members: list[dict]) -> dict | None:
    """Create one open event and assign its member articles. dry_run-safe (returns None)."""
    if config.get("dry_run"):
        return None
    first_seen = min((a["published_at"] for a in members if a.get("published_at")), default=None)
    row = db.fetch_one(
        config,
        """INSERT INTO news_events (event_key, title, status, first_seen, last_updated, article_count)
           VALUES (%s, %s, 'open', COALESCE(%s, now()), COALESCE(%s, now()), %s)
           RETURNING id, event_key, title""",
        (event_key, title, first_seen, _news_time(members), len(members)),
    )
    db.execute_many(config, "UPDATE news_articles SET event_id=%s WHERE id=%s",
                    [(row["id"], a["id"]) for a in members])
    return row


def _append_articles(config: dict, event_id: int, members: list[dict]) -> None:
    """Attach articles to an existing open event and flag it for re-interpretation."""
    if config.get("dry_run"):
        return
    db.execute_many(config, "UPDATE news_articles SET event_id=%s WHERE id=%s",
                    [(event_id, a["id"]) for a in members])
    db.execute(config, """
        UPDATE news_events
        SET last_updated = GREATEST(last_updated, COALESCE(%s, now())),
            article_count = article_count + %s,
            needs_reinterpretation = true
        WHERE id = %s
    """, (_news_time(members), len(members), event_id))


def _pick_title(members: list[dict]) -> str:
    """Representative title for an aggregated/rollup event: the highest-triage article's
    title (falls back to the first). Keeps a rollup card from being titled by a random member."""
    best = max(members, key=lambda a: (a.get("triage_score") or 0))
    return best.get("title") or members[0].get("title") or "Untitled event"


def _form_event_groups(config: dict, articles: list[dict], open_brief: list[dict],
                       taxonomy: list[dict], rollup_types: set[str]) -> list[dict]:
    """Form event groups, bucketed by triage event_type.

    - ROLLUP types (earnings, market-wrap): the whole bucket is ONE event (deterministic,
      never sent to the LLM — the rollup rule is too important to leave to a model).
    - DISTINCT types: a per-bucket LLM pass splits the bucket into individual stories.
      Running per-bucket (one topic at a time) rather than over all articles at once was
      decisively better in testing — the model over-split and skipped rollups when handed
      all 128 articles together. If the LLM call fails, every article in that bucket
      becomes its own singleton event — never one-event-per-bucket (the old over-merge).
      The catalyst-merge pass can still fuse singletons that share a driver.
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        buckets[_event_type_from_key(a.get("event_key") or "")].append(a)
    open_by_type: dict[str, list[dict]] = defaultdict(list)
    for e in open_brief:
        open_by_type[e["event_type"]].append(e)

    groups: list[dict] = []
    for etype, bucket in buckets.items():
        if etype in rollup_types:
            existing = open_by_type.get(etype)
            groups.append({
                "assign_to_event_id": existing[0]["event_id"] if existing else None,
                "event_type": etype, "title": _pick_title(bucket), "members": bucket,
            })
            continue
        if len(bucket) == 1:
            groups.append({"assign_to_event_id": None, "event_type": etype,
                           "title": bucket[0].get("title"), "members": bucket})
            continue
        sub = None
        try:
            sub = llm_cluster(config, bucket, open_by_type.get(etype, []), rollup_types)
        except Exception as exc:  # noqa: BLE001 — degrade to singletons, never crash
            log.warning("S3 per-bucket LLM failed for %s (%s) — singleton fallback",
                        etype, exc)
        if sub is None:
            sub = [{"assign_to_event_id": None, "event_type": etype,
                    "title": a.get("title"), "members": [a]} for a in bucket]
        groups.extend(sub)
    return groups


MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "idxs": {"type": "array", "items": {"type": "integer"}},
                "event_type": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["idxs", "title"],
        }}
    },
    "required": ["merges"],
}


def _consolidate_catalysts(config: dict, groups: list[dict], rollup_types: set[str]) -> list[dict]:
    """Merge separately-detected events that are downstream consequences of the SAME
    catalyst (e.g. an oil shock's crude / rupee / bond events) into one multi-chain event.

    Conservative by design (prompt + this being an event-level, not article-level, pass):
    it only fuses whole events the model is confident share a driver, so it cannot recreate
    the article-level over-merge. Degrades to no-op on any LLM failure. Rollup events are
    never merged. NEW (non-appended) events only are considered.
    """
    # only non-append, non-rollup events with a title are merge-eligible
    eligible = [i for i, g in enumerate(groups)
                if not g.get("assign_to_event_id") and g.get("event_type") not in rollup_types]
    if len(eligible) < 2:
        return groups

    payload = [{"idx": i, "event_type": groups[i]["event_type"],
                "title": groups[i].get("title", "")} for i in eligible]
    template = load_prompt("catalyst_merge_v1.txt")
    marker = "RUNTIME INPUTS\n======================================================================\n"
    system = template.split("======================================================================\nRUNTIME INPUTS")[0].strip()
    user = template.split(marker, 1)[-1].replace("{{EVENTS}}",
                                                 json.dumps(payload, ensure_ascii=False))
    try:
        llm = get_llm(config, tier="cheap")
        res = llm.complete_json(system=system.strip(), user=user.strip(),
                                json_schema=MERGE_SCHEMA, schema_name="catalyst_merge",
                                temperature=0.0, max_tokens=2000)
        merges = (res.data or {}).get("merges") or []
    except Exception as exc:  # noqa: BLE001 — merging is a refinement; never crash S3
        log.warning("S3 catalyst-merge failed (%s) — leaving events unmerged", exc)
        return groups

    eligible_set = set(eligible)
    used: set[int] = set()
    merged_groups: list[dict] = []
    for m in merges:
        idxs = [i for i in m.get("idxs", []) if i in eligible_set and i not in used]
        if len(idxs) < 2:
            continue
        used.update(idxs)
        members: list[dict] = []
        seen_ids: set[int] = set()
        for i in idxs:
            for a in groups[i]["members"]:
                if a["id"] not in seen_ids:
                    members.append(a); seen_ids.add(a["id"])
        etype = (m.get("event_type") or groups[idxs[0]]["event_type"] or "").strip().lower()
        merged_groups.append({"assign_to_event_id": None, "event_type": etype,
                              "title": (m.get("title") or groups[idxs[0]]["title"]).strip(),
                              "members": members, "catalyst_merged": True})
    if not merged_groups:
        return groups
    log.info("S3 catalyst-merge: fused %d event(s) into %d macro-event(s)",
             len(used), len(merged_groups))
    kept = [g for i, g in enumerate(groups) if i not in used]
    return kept + merged_groups


def _materialise_groups(config: dict, groups: list[dict], articles: list[dict],
                        taxonomy: list[dict], rollup_types: set[str]):
    """Turn LLM groups into DB events. Returns (created, appended, singletons, bundles)."""
    created = appended = 0
    bundles: list[dict] = []
    used_keys: set[str] = set()
    for g in groups:
        members = g["members"]
        if g.get("assign_to_event_id"):
            _append_articles(config, g["assign_to_event_id"], members)
            appended += 1
            continue
        etype = g["event_type"] if any(t["slug"] == g["event_type"] for t in taxonomy) else "uncategorised"
        base_key = build_event_key(etype, taxonomy)
        if etype in rollup_types:
            key = base_key                      # rollups share the canonical monthly/quarterly key
        else:
            key = f"{base_key}-{_slugify(g['title'])}"   # distinct events get a story-specific key
        # keep keys unique within this run
        k, n = key, 2
        while k in used_keys:
            k, n = f"{key}-{n}", n + 1
        used_keys.add(k)
        row = _new_event(config, k, g["title"], members)
        created += 1
        if row is not None:
            bundles.append(build_bundle(row, members))
    singletons = sum(1 for g in groups if len(g["members"]) == 1)
    return created, appended, singletons, bundles


# ---------------------------------------------------------------- stage
def run(config: dict, **_inputs) -> dict:
    ccfg = config.get("cluster", {})
    lookback_days = int(ccfg.get("open_event_lookback_days", 7))
    close_days = int(ccfg.get("auto_close_days", 7))

    articles = db.fetch_all(
        config,
        """
        SELECT id, url, source, title, rss_summary, raw_text, published_at, event_key,
               triage_score,
               COALESCE(published_at, ingested_at) AS cluster_time
        FROM news_articles
        WHERE status = 'kept' AND event_id IS NULL
        ORDER BY COALESCE(published_at, ingested_at) ASC, id ASC
        """,
    )
    if not articles:
        closed = _auto_close(config, close_days)
        return {"stage": "s3_cluster", "articles": 0, "events_created": 0,
                "events_appended": 0, "events_closed": closed}

    # ---- form events (LLM-based, deterministic fallback) --------------------
    # PRIMARY: an LLM groups the kept articles into distinct STORIES (see clustering
    # rationale in cluster_v1.txt / BUILD_NOTES D10 — a shared taxonomy event_key must
    # never merge a whole bucket). FALLBACK: if a bucket's LLM call fails, its articles
    # become singleton events
    # (never one-event-per-bucket), so a model outage can split stories but never
    # over-merge them or crash the run (Prime Directive #3).
    taxonomy = load_taxonomy()
    rollup_types = {t["slug"] for t in taxonomy if t.get("aggregation") == "rollup"}
    open_brief = _load_open_events_brief(config, lookback_days)

    groups = _form_event_groups(config, articles, open_brief, taxonomy, rollup_types)
    # Second pass: fuse separately-detected events that share one catalyst (e.g. an oil
    # shock's crude/rupee/bond events) into a single multi-chain macro-event.
    groups = _consolidate_catalysts(config, groups, rollup_types)
    created, appended, singletons, bundles = _materialise_groups(
        config, groups, articles, taxonomy, rollup_types
    )
    cluster_mode = "llm-per-bucket+catalyst-merge"

    closed = _auto_close(config, close_days)

    result = {
        "stage": "s3_cluster",
        "articles": len(articles),
        "events_created": created,
        "events_appended": appended,
        "events_closed": closed,
        "singletons": singletons,
        "cluster_mode": cluster_mode,
        "bundles": bundles,
    }
    return result


def _auto_close(config: dict, close_days: int) -> int:
    if config.get("dry_run"):
        return 0
    rows = db.fetch_all(
        config,
        """UPDATE news_events
           SET status='closed'
           WHERE status='open'
             AND last_updated < now() - (%s || ' days')::interval
           RETURNING id""",
        (str(close_days),),
    )
    return len(rows)


def merge_events(config: dict, primary_id: int, duplicate_ids: list[int],
                 reextract_facts: bool = False) -> dict:
    """Fold duplicate events into a primary, in ONE transaction (BUILD_NOTES F25).

    reextract_facts=True (only allowed while NO card exists on the primary or any
    duplicate): deletes the merged events' derived facts and clears their articles'
    facts_extracted_at, so the next S4+S5 run extracts the merged event as one bundle and
    S5 cross-checks all its sources together. Without it, facts are only re-pointed and
    are never consistency-checked across the former events. Raw articles are untouched,
    so this is fully reproducible (Prime Directive #6).

    Articles and facts are re-pointed to the primary; duplicates are closed with
    `merged_into = primary_id` so the merge stays auditable and reversible; the primary's
    article_count / first_seen / last_updated are recomputed and it is flagged for
    re-interpretation. Duplicates that already have interpretation cards are refused —
    merging those would orphan published cards and needs a human decision.
    """
    dups = sorted({int(d) for d in duplicate_ids} - {int(primary_id)})
    if not dups:
        return {"primary": primary_id, "merged": []}

    def _tx(cur):
        cur.execute("SELECT id, status FROM news_events WHERE id = ANY(%s)", (dups + [primary_id],))
        found = {r["id"]: r["status"] for r in cur.fetchall()}
        missing = [i for i in dups + [primary_id] if i not in found]
        if missing:
            raise ValueError(f"merge_events: unknown event id(s) {missing}")
        if found[primary_id] != "open":
            raise ValueError(f"merge_events: primary {primary_id} is not open")
        cur.execute("SELECT DISTINCT event_id FROM event_interpretations WHERE event_id = ANY(%s)",
                    (dups,))
        carded = [r["event_id"] for r in cur.fetchall()]
        if carded:
            raise ValueError(f"merge_events: duplicate(s) {carded} already have cards")
        if reextract_facts:
            cur.execute("SELECT 1 FROM event_interpretations WHERE event_id = %s LIMIT 1",
                        (primary_id,))
            if cur.fetchone():
                raise ValueError(f"merge_events: primary {primary_id} has cards; "
                                 f"reextract_facts would orphan their fact citations")
        cur.execute("UPDATE news_articles SET event_id=%s WHERE event_id = ANY(%s)",
                    (primary_id, dups))
        if reextract_facts:
            cur.execute("DELETE FROM extracted_facts WHERE event_id = ANY(%s)",
                        (dups + [primary_id],))
            cur.execute("UPDATE news_articles SET facts_extracted_at = NULL WHERE event_id = %s",
                        (primary_id,))
        else:
            cur.execute("UPDATE extracted_facts SET event_id=%s WHERE event_id = ANY(%s)",
                        (primary_id, dups))
        cur.execute("""UPDATE news_events SET status='closed', merged_into=%s,
                              needs_reinterpretation=false WHERE id = ANY(%s)""",
                    (primary_id, dups))
        cur.execute("""
            UPDATE news_events e SET
              article_count = (SELECT count(*) FROM news_articles a
                               WHERE a.event_id = e.id AND a.status = 'kept'),
              first_seen = (SELECT min(first_seen) FROM news_events WHERE id = ANY(%s)),
              last_updated = (SELECT max(last_updated) FROM news_events WHERE id = ANY(%s)),
              needs_reinterpretation = EXISTS (SELECT 1 FROM event_interpretations i
                                               WHERE i.event_id = e.id)
            WHERE e.id = %s""", (dups + [primary_id], dups + [primary_id], primary_id))
        return {"primary": primary_id, "merged": dups, "reextract_facts": reextract_facts}

    if config.get("dry_run"):
        return {"primary": primary_id, "merged": dups, "dry_run": True}
    return db.run_transaction(config, _tx)

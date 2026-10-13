"""Backlog S3 in agent mode: export -> (Claude groups) -> apply through s3's own materialisation.

    python s3_agent.py export          # kept+unclustered articles + open events -> s3_in.json
    python s3_agent.py apply [--dry]   # s3_groups.json -> s3_cluster._materialise_groups + auto-close

s3_groups.json: [{"assign_to_event_id": <open event id or null>, "event_type": "<slug>",
                  "title": "...", "member_ids": [article ids]}]
Rules enforced here: every exported article in exactly one group; rollup types
(earnings, market wrap) form ONE group per type; assign_to_event_id must be an open event.
"""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import load_config
from src import db, s3_cluster
from src.s2_filter import load_taxonomy

# working files (exports, agent outputs) live in data/backlog/ — gitignored
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "backlog"))
os.makedirs(HERE, exist_ok=True)
SQL = """SELECT id, url, source, title, rss_summary, raw_text, published_at, event_key, triage_score,
               COALESCE(published_at, ingested_at) AS cluster_time
        FROM news_articles WHERE status='kept' AND event_id IS NULL
        ORDER BY COALESCE(published_at, ingested_at), id"""


def export():
    cfg = load_config()
    arts = db.fetch_all(cfg, SQL)
    tax = load_taxonomy()
    rollup = sorted(t["slug"] for t in tax if t.get("aggregation") == "rollup")
    open_ev = s3_cluster._load_open_events_brief(cfg, int(cfg["cluster"].get("open_event_lookback_days", 7)))
    out = {"rollup_types": rollup, "open_events": open_ev,
           "articles": [{"id": a["id"], "event_type": s3_cluster._event_type_from_key(a["event_key"] or ""),
                         "published_at": a["published_at"], "source": a["source"], "title": a["title"],
                         "triage_score": a["triage_score"],
                         "summary": (a.get("rss_summary") or (a.get("raw_text") or ""))[:400]} for a in arts]}
    json.dump(out, open(os.path.join(HERE, "s3_in.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("articles:", len(arts), "| open events:", len(open_ev), "| by type:",
          dict(Counter(a["event_type"] for a in out["articles"]).most_common()))
    db.close_shared()


def apply(dry):
    cfg = load_config()
    if dry:
        cfg["dry_run"] = True
    tax = load_taxonomy()
    rollup = {t["slug"] for t in tax if t.get("aggregation") == "rollup"}
    slugs = {t["slug"] for t in tax}
    arts = {a["id"]: a for a in db.fetch_all(cfg, SQL)}
    open_ids = {e["event_id"] for e in s3_cluster._load_open_events_brief(
        cfg, int(cfg["cluster"].get("open_event_lookback_days", 7)))}
    groups = json.load(open(os.path.join(HERE, "s3_groups.json"), encoding="utf-8"))
    seen, problems, rollup_groups = Counter(), [], defaultdict(int)
    for g in groups:
        seen.update(g["member_ids"])
        if g["event_type"] not in slugs:
            problems.append(f"unknown event_type {g['event_type']}")
        if g.get("assign_to_event_id") and g["assign_to_event_id"] not in open_ids:
            problems.append(f"assign_to_event_id {g['assign_to_event_id']} is not an open event")
        if g["event_type"] in rollup and not g.get("assign_to_event_id"):
            rollup_groups[g["event_type"]] += 1
    problems += [f"article {i} in {n} groups" for i, n in seen.items() if n > 1]
    problems += [f"article {i} not in any group" for i in set(arts) - set(seen)]
    problems += [f"unknown article {i}" for i in set(seen) - set(arts)]
    problems += [f"rollup type {t} split into {n} new events" for t, n in rollup_groups.items() if n > 1]
    if problems:
        print("REFUSING:", problems[:20]); return 1
    mat = [{"assign_to_event_id": g.get("assign_to_event_id"), "event_type": g["event_type"],
            "title": g["title"], "members": [arts[i] for i in g["member_ids"]]} for g in groups]
    created, appended, singletons, _ = s3_cluster._materialise_groups(cfg, mat, list(arts.values()), tax, rollup)
    closed = s3_cluster._auto_close(cfg, int(cfg["cluster"].get("auto_close_days", 7)))
    print(json.dumps({"groups": len(groups), "events_created": created, "events_appended": appended,
                      "singletons": singletons, "events_closed": closed, "dry_run": bool(dry)}))
    db.close_shared()
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    else:
        sys.exit(apply("--dry" in sys.argv))

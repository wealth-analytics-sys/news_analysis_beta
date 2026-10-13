"""Backlog S2 in agent mode: export -> (agents score) -> apply through the REAL s2_filter.run().

    python s2_agent.py export          # Pass A survivors -> in/batch_NN.json (+ taxonomy.json)
    python s2_agent.py apply [--dry]   # scores/batch_NN.json -> s2_filter.run with _pass_b replaced
"""
import glob, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import load_config
from src import db, s2_filter

# working files (exports, agent outputs) live in data/backlog/ — gitignored
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "backlog"))
os.makedirs(HERE, exist_ok=True)
BATCH = 50


def _new_articles(cfg):
    return db.fetch_all(cfg, """SELECT id, url, source, title, rss_summary, raw_text
        FROM news_articles WHERE status='new' ORDER BY published_at DESC NULLS LAST, id DESC""")


def export():
    cfg = load_config()
    arts = _new_articles(cfg)
    survivors, dropped = s2_filter._pass_a(arts, cfg.get("filter", {}))
    os.makedirs(os.path.join(HERE, "in"), exist_ok=True)
    tax = [{"slug": t["slug"], "description": (t.get("description") or "").strip()}
           for t in s2_filter.load_taxonomy()]
    json.dump(tax, open(os.path.join(HERE, "taxonomy.json"), "w", encoding="utf-8"), indent=1)
    payload = [{"article_id": a["id"], "source": a.get("source"), "title": a.get("title"),
                "summary": (a.get("rss_summary") or (a.get("raw_text") or "")[:600])} for a in survivors]
    n = 0
    for i in range(0, len(payload), BATCH):
        n += 1
        json.dump(payload[i:i + BATCH], open(os.path.join(HERE, "in", f"batch_{n:02d}.json"), "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"new={len(arts)} pass_a_survivors={len(survivors)} pass_a_dropped={len(dropped)} batches={n}")
    db.close_shared()


def apply(dry):
    cfg = load_config()
    if dry:
        cfg["dry_run"] = True
    tax = s2_filter.load_taxonomy()
    slugs = {t["slug"] for t in tax}
    scores, bad = {}, []
    for p in sorted(glob.glob(os.path.join(HERE, "scores", "batch_*.json"))):
        for r in json.load(open(p, encoding="utf-8")):
            pr, kb = float(r["portfolio_relevance"]), float(r["kb_value"])
            et = (r.get("event_type") or "uncategorised").strip().lower()
            if et not in slugs:
                bad.append((r["article_id"], et)); et = "uncategorised"
            scores[int(r["article_id"])] = {
                "portfolio_relevance": pr, "kb_value": kb, "score": max(pr, kb), "event_type": et,
                "event_key": s2_filter.build_event_key(et, tax),
                "portfolio_reason": r.get("portfolio_reason", ""), "kb_reason": r.get("kb_reason", "")}
    survivors_ids = {a["id"] for a in s2_filter._pass_a(_new_articles(cfg), cfg.get("filter", {}))[0]}
    missing = sorted(survivors_ids - set(scores))
    if missing:
        print(f"REFUSING: {len(missing)} Pass-A survivors have no score, e.g. {missing[:10]}")
        return 1
    # the only substitution: the LLM call; thresholds, keys and DB writes are s2_filter's own
    s2_filter._pass_b = lambda config, articles: ({a["id"]: scores[a["id"]] for a in articles}, [])
    res = s2_filter.run(cfg)
    print(json.dumps(res), "| invalid event_types coerced:", bad)
    db.close_shared()
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    else:
        sys.exit(apply("--dry" in sys.argv))

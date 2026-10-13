"""Backlog S4 in agent mode: export bundles -> (Claude extracts) -> S4 post-processing -> real S5.

    python s4_agent.py export [--event ID ...]   # s4_extract._load_event_bundles -> s4_in/<id>.json
    python s4_agent.py apply  [--dry]            # s4_out/<id>.json -> s5_fact_ingest.run

s4_out/<id>.json: {"event_id": <id>, "facts": [...], "transient_context": [...]} exactly as
prompts/extract_v1.txt rule 11 specifies.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import load_config
from src import db, s4_extract, s5_fact_ingest

# working files (exports, agent outputs) live in data/backlog/ — gitignored
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "backlog"))
os.makedirs(HERE, exist_ok=True)


def export(event_ids):
    cfg = load_config()
    bundles = s4_extract._load_event_bundles(cfg, event_ids or None)
    metrics, entities = s4_extract._load_dictionaries(cfg)
    os.makedirs(os.path.join(HERE, "s4_in"), exist_ok=True)
    json.dump({"METRICS_DICTIONARY": metrics, "ENTITY_DICTIONARY": entities},
              open(os.path.join(HERE, "s4_dictionaries.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    for b in bundles:
        json.dump(b, open(os.path.join(HERE, "s4_in", f"{b['event_id']}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=str)
    print(json.dumps([{"event_id": b["event_id"], "articles": len(b["articles"]), "title": b["title"][:70]}
                      for b in bundles], ensure_ascii=False))
    db.close_shared()


def apply(dry):
    cfg = load_config()
    if dry:
        cfg["dry_run"] = True
    events, problems = [], []
    for p in sorted(glob.glob(os.path.join(HERE, "s4_out", "*.json"))):
        eid = int(os.path.basename(p)[:-5])
        bundle = json.load(open(os.path.join(HERE, "s4_in", f"{eid}.json"), encoding="utf-8"))
        out = json.load(open(p, encoding="utf-8"))
        art_ids = {a["article_id"] for a in bundle["articles"]}
        try:
            facts = s4_extract._coerce_items(out, "facts")
            transient = s4_extract._coerce_items(out, "transient_context")
        except ValueError as exc:
            problems.append((eid, str(exc))); continue
        stray = sorted({f.get("article_id") for f in facts + transient} - art_ids)
        if stray:
            problems.append((eid, f"article_ids not in bundle: {stray}")); continue
        for f in facts:
            f["verification_status"] = "single_source"          # S5 upgrades, never extraction
        s4_extract._flag_post_publication_dates(facts + transient, bundle["articles"])
        events.append({"event_id": eid, "article_ids": sorted(art_ids), "facts": facts,
                       "transient_context": transient, "model_version": "claude-code-agent"})
    if problems:
        print("REFUSING:", problems); return 1
    res = s5_fact_ingest.run(cfg, extraction={"events": events})
    print(json.dumps({k: v for k, v in res.items() if k != "per_event"}))
    db.close_shared()
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "export":
        ids = [int(x) for x in sys.argv[3:]] if "--event" in sys.argv else []
        export(ids)
    else:
        sys.exit(apply("--dry" in sys.argv))

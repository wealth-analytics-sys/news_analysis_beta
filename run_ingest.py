"""Part 1 of the pipeline: S1 ingest -> S2 filter -> S3 cluster -> S4 extract -> S5 fact-ingest.

Runs independently of S6 (interpretation), which has its own entrypoint, run_s6.py.

Usage:
    python run_ingest.py                      # S1-S5
    python run_ingest.py --stages s1,s2,s3    # a subset (S5 needs S4 in the same run)
    python run_ingest.py --limit-events 10    # cap S4 to the newest N open events
    python run_ingest.py --dry-run            # no DB writes

S4's facts reach S5 in memory, and S5 persists them together with each event's transient
context (news_events.transient_context) — which is how a separately-run S6 gets it.
Secrets come from the environment only. Never commit them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src import db, s1_ingest, s2_filter, s3_cluster, s4_extract, s5_fact_ingest
from src.config import load_config

STAGES = {"s1": s1_ingest, "s2": s2_filter, "s3": s3_cluster,
          "s4": s4_extract, "s5": s5_fact_ingest}


def _newest_open_event_ids(config: dict, limit: int) -> list[int]:
    rows = db.fetch_all(config, """
        SELECT id FROM news_events WHERE status='open'
        ORDER BY last_updated DESC LIMIT %s
    """, (limit,))
    return [r["id"] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s1,s2,s3,s4,s5",
                    help="comma-separated subset of s1,s2,s3,s4,s5")
    ap.add_argument("--limit-events", type=int, default=0,
                    help="cap how many events S4 extracts (newest-first). 0 = no cap.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger("ingest")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        log.error("unknown stage(s) %s — S6 runs separately via run_s6.py", unknown)
        return 2
    if "s5" in stages and "s4" not in stages:
        log.error("s5 persists S4's in-memory output, so it must run with s4 in the same run")
        return 2

    config = load_config()
    if args.dry_run:
        config["dry_run"] = True
    log.info("llm_provider=%s dry_run=%s", config["llm_provider"], config["dry_run"])

    results, failed = {}, False
    for key in stages:
        log.info("--- running %s ---", key)
        try:
            if key == "s4":
                # Resolved here, AFTER S3 has created this run's events (F22). An empty
                # cap means "nothing to extract", never "extract everything".
                event_ids = None
                if args.limit_events and not config["dry_run"]:
                    event_ids = _newest_open_event_ids(config, args.limit_events) or [-1]
                    log.info("limit-events=%d -> S4 capped to %s", args.limit_events, event_ids)
                res = s4_extract.run(config, event_ids=event_ids)
            elif key == "s5":
                res = s5_fact_ingest.run(
                    config, extraction={"events": results["s4"].get("events_out", [])})
            else:
                res = STAGES[key].run(config)
            results[key] = res
            log.info("%s done: %s", key, json.dumps(
                {k: v for k, v in res.items() if k not in ("bundles", "events_out", "per_event")},
                default=str))
        except Exception as exc:  # noqa: BLE001
            failed = True
            log.exception("stage %s failed: %s", key, exc)
            results[key] = {"error": f"{type(exc).__name__}: {exc}"}
            break

    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("bundles", "events_out", "per_event")}
               for k, v in results.items()}
    print(json.dumps(summary, indent=2, default=str))
    db.close_shared()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

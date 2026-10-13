"""Part 2 of the pipeline: S6 interpretation, runnable on its own.

Agent mode — Claude Code writes the cards (no API credits); see .claude/commands/s6-interpret.md:
    python run_s6.py pending                      # events that need a card
    python run_s6.py export --all                 # write input bundles to data/s6/in/<id>.json
    python run_s6.py export --event 1004 1090     # ...or for specific events
    python run_s6.py check data/s6/cards/1004.json   # run every gate, write nothing
    python run_s6.py write data/s6/cards/*.json      # gates + insert (version N+1)
    python run_s6.py skip 1201 1202 --reason "..."   # close without a card, reason recorded

API mode — the strong-tier model writes the cards (spends API credits):
    python run_s6.py api [--event 1004 ...] [--dry-run]

`write` is the only way an agent-authored card reaches the database: it runs the citation
validator, the JSON schema and the wording check, and REJECTS the card (non-zero exit,
reasons printed) if the schema or wording check fails, so the author fixes and retries.

Gateway mode (BUILD_NOTES D17) — set S6_BACKEND=gateway (the Claude cloud routine does): the
same commands (pending / export / check / write / skip) talk to Supabase over HTTPS through
the S6 gateway functions instead of a Postgres connection; the content gates are the same
code (s6_interpret.prepare_card). API mode is not available in gateway mode.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

from src import db, s6_interpret as s6
from src.config import load_config
from src.s6_gateway import Gateway, GatewayError

ROOT = Path(__file__).resolve().parent
S6_DIR = ROOT / "data" / "s6"
AGENT_MODEL_VERSION = "claude-code-agent"


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _gateway_mode(config) -> bool:
    return config.get("s6_backend") == "gateway"


# ---------------------------------------------------------------- gateway mode (D17)
def gw_pending(gw, args) -> int:
    rows = gw.pending()
    if args.event:
        rows = [r for r in rows if r["event_id"] in set(args.event)]
    _print({"pending": len(rows), "events": [
        {"event_id": r["event_id"], "title": r["title"], "update": bool(r["needs_reinterpretation"]),
         "articles": r["articles"], "facts": r["facts"]} for r in rows]})
    return 0


def gw_export(gw, args) -> int:
    if not args.all and not args.event:
        print("export: pass --all or --event ID [ID ...]", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = {r["event_id"] for r in gw.pending()}
    wanted = sorted(pending) if args.all else [e for e in args.event if e in pending]
    index, aliases = [], {}
    for eid in wanted:
        inputs, aliases = gw.bundle(eid)
        if not inputs["ARTICLE_SUMMARIES"]:
            continue
        path = out_dir / f"{eid}.json"
        path.write_text(json.dumps(inputs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        index.append({"event_id": eid, "bundle": str(path),
                      "articles": len(inputs["ARTICLE_SUMMARIES"]),
                      "facts": len(inputs["EVENT_FACTS"]), "kb_facts": len(inputs["KB_FACTS"]),
                      "transient": len(inputs["TRANSIENT_CONTEXT"]),
                      "is_update": inputs["PRIOR_INTERPRETATION"] is not None})
    (out_dir / "_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (out_dir / "_aliases.json").write_text(json.dumps(aliases, ensure_ascii=False), encoding="utf-8")
    _print({"exported": len(index), "dir": str(out_dir), "events": index})
    return 0


def gw_write_or_check(gw, args, dry: bool) -> int:
    alias_file = Path(args.bundles) / "_aliases.json"
    aliases = json.loads(alias_file.read_text(encoding="utf-8")) if alias_file.exists() else None
    ok, rejected = [], []
    for path in _card_files(args.cards):
        try:
            event_id = args.event_id or int(path.stem)
            payload = json.loads(path.read_text(encoding="utf-8"))
            bundle = Path(args.bundles) / f"{event_id}.json"
            if bundle.exists():
                inputs = json.loads(bundle.read_text(encoding="utf-8"))
            else:
                inputs, fetched = gw.bundle(event_id)
                aliases = aliases if aliases is not None else fetched
            if aliases is None:
                _, aliases = gw.bundle(event_id)
            ok.append(s6.write_card_via_gateway(gw, event_id, payload, args.model_version,
                                                inputs, aliases, dry_run=dry))
        except s6.CardRejected as exc:
            rejected.append({"card": str(path), "event_id": exc.event_id, "problems": exc.problems})
        except (ValueError, OSError, GatewayError) as exc:
            rejected.append({"card": str(path), "problems": [f"{type(exc).__name__}: {exc}"]})
    _print({"mode": ("check" if dry else "write") + " (gateway)",
            "written" if not dry else "passed": ok, "rejected": rejected})
    return 1 if rejected else 0


def gw_skip(gw, args) -> int:
    if not args.reason or len(args.reason.strip()) < 10:
        print("skip: --reason is required (a real sentence a reviewer can check)", file=sys.stderr)
        return 2
    done = gw.skip(args.event_ids, args.reason.strip())
    _print({"skipped": done, "reason": args.reason.strip()})
    return 0 if all(d["closed"] for d in done) else 1


def cmd_pending(config, args) -> int:
    events = s6.pending_events(config, args.event or None)
    rows = []
    for e in events:
        c = db.fetch_one(config, """
            SELECT (SELECT count(*) FROM news_articles WHERE event_id=%s AND status='kept') AS articles,
                   (SELECT count(*) FROM extracted_facts WHERE event_id=%s) AS facts""",
                         (e["id"], e["id"]))
        rows.append({"event_id": e["id"], "title": e["title"], "update": bool(e["needs_reinterpretation"]),
                     **c})
    _print({"pending": len(rows), "events": rows})
    return 0


def cmd_export(config, args) -> int:
    if not args.all and not args.event:
        print("export: pass --all or --event ID [ID ...]", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for e in s6.pending_events(config, None if args.all else args.event):
        inputs = s6.build_inputs(config, e)
        if not inputs["ARTICLE_SUMMARIES"]:
            continue
        path = out_dir / f"{e['id']}.json"
        path.write_text(json.dumps(inputs, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
        index.append({"event_id": e["id"], "bundle": str(path),
                      "articles": len(inputs["ARTICLE_SUMMARIES"]),
                      "facts": len(inputs["EVENT_FACTS"]), "kb_facts": len(inputs["KB_FACTS"]),
                      "transient": len(inputs["TRANSIENT_CONTEXT"]),
                      "is_update": inputs["PRIOR_INTERPRETATION"] is not None})
    (out_dir / "_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    _print({"exported": len(index), "dir": str(out_dir), "events": index})
    return 0


def _card_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in patterns:
        files += [Path(x) for x in sorted(glob.glob(p))] or [Path(p)]
    return files


def _write_or_check(config, args, dry: bool) -> int:
    if dry:
        config = {**config, "dry_run": True}
    ok, rejected = [], []
    for path in _card_files(args.cards):
        try:
            event_id = args.event_id or int(path.stem)
            payload = json.loads(path.read_text(encoding="utf-8"))
            bundle = Path(args.bundles) / f"{event_id}.json"
            inputs = json.loads(bundle.read_text(encoding="utf-8")) if bundle.exists() else None
            ok.append(s6.write_card(config, event_id, payload, args.model_version,
                                    inputs, strict=True))
        except s6.CardRejected as exc:
            rejected.append({"card": str(path), "event_id": exc.event_id, "problems": exc.problems})
        except (ValueError, OSError) as exc:
            rejected.append({"card": str(path), "problems": [f"{type(exc).__name__}: {exc}"]})
    _print({"mode": "check" if dry else "write", "written" if not dry else "passed": ok,
            "rejected": rejected})
    return 1 if rejected else 0


def cmd_skip(config, args) -> int:
    """Close a pending event WITHOUT a card and record why (unattended mode, D16)."""
    if not args.reason or len(args.reason.strip()) < 10:
        print("skip: --reason is required (a real sentence a reviewer can check)", file=sys.stderr)
        return 2
    done = []
    for eid in args.event_ids:
        row = db.fetch_one(config, """UPDATE news_events SET status='closed', needs_reinterpretation=false,
                                      skip_reason=%s
                                      WHERE id=%s AND status='open'
                                        AND NOT EXISTS (SELECT 1 FROM event_interpretations i WHERE i.event_id=%s)
                                      RETURNING id""", (args.reason.strip(), eid, eid))
        done.append({"event_id": eid, "closed": bool(row)})
    _print({"skipped": done, "reason": args.reason.strip()})
    return 0 if all(d["closed"] for d in done) else 1


def cmd_api(config, args) -> int:
    if args.dry_run:
        config = {**config, "dry_run": True}
    _print(s6.run(config, event_ids=args.event or None))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pending", help="list events that need a card")
    p.add_argument("--event", type=int, nargs="*")

    p = sub.add_parser("export", help="write input bundles for the agent")
    p.add_argument("--all", action="store_true")
    p.add_argument("--event", type=int, nargs="*")
    p.add_argument("--out", default=str(S6_DIR / "in"))

    for name in ("check", "write"):
        p = sub.add_parser(name, help=("run all gates without writing" if name == "check"
                                       else "run all gates and insert the card"))
        p.add_argument("cards", nargs="+", help="card JSON file(s) named <event_id>.json, globs ok")
        p.add_argument("--event-id", type=int, help="override when a file isn't named <id>.json")
        p.add_argument("--bundles", default=str(S6_DIR / "in"),
                       help="dir with the exported bundles (used for facts; else re-read from DB)")
        p.add_argument("--model-version", default=AGENT_MODEL_VERSION)

    p = sub.add_parser("skip", help="close events without a card, recording why (unattended mode)")
    p.add_argument("event_ids", type=int, nargs="+")
    p.add_argument("--reason", required=True)

    p = sub.add_parser("api", help="API mode: the strong-tier model writes the cards")
    p.add_argument("--event", type=int, nargs="*")
    p.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    config = load_config()
    if _gateway_mode(config):
        if args.cmd == "api":
            print("api mode is not available with S6_BACKEND=gateway", file=sys.stderr)
            return 2
        try:
            gw = Gateway.from_config(config)
            if args.cmd == "pending":
                return gw_pending(gw, args)
            if args.cmd == "export":
                return gw_export(gw, args)
            if args.cmd in ("check", "write"):
                return gw_write_or_check(gw, args, dry=args.cmd == "check")
            return gw_skip(gw, args)
        except GatewayError as exc:
            print(f"gateway error: {exc}", file=sys.stderr)
            return 1
    try:
        if args.cmd == "pending":
            return cmd_pending(config, args)
        if args.cmd == "export":
            return cmd_export(config, args)
        if args.cmd in ("check", "write"):
            return _write_or_check(config, args, dry=args.cmd == "check")
        if args.cmd == "skip":
            return cmd_skip(config, args)
        return cmd_api(config, args)
    finally:
        db.close_shared()


if __name__ == "__main__":
    sys.exit(main())

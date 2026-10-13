"""Apply (or re-apply) the S6 HTTPS gateway migration — BUILD_NOTES D17.

    python tools/apply_s6_gateway.py            # apply; generates a token on first run
    python tools/apply_s6_gateway.py --rotate   # new token, functions re-created with its hash

The gateway token is stored ONLY in your local .env as S6_GATEWAY_TOKEN (gitignored) and in
the Claude cloud environment; the database keeps just its SHA-256 hash. Nothing is printed.
Needs DATABASE_URL (direct DB access) — run it from your machine or a GitHub Action.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db                      # noqa: E402
from src.config import load_config      # noqa: E402

ENV = ROOT / ".env"
SQL = ROOT / "sql" / "migrations" / "2026-10-06_s6_gateway.sql"
KEY = "S6_GATEWAY_TOKEN"


def _token(rotate: bool) -> tuple[str, str]:
    text = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    m = re.search(rf"^{KEY}=(.+)$", text, re.M)
    if m and not rotate:
        return m.group(1).strip().strip('"\''), "existing"
    token = secrets.token_urlsafe(32)
    if m:
        text = text[:m.start()] + f"{KEY}={token}" + text[m.end():]
    else:
        text = text.rstrip("\n") + (
            "\n\n# ---- S6 HTTPS gateway (Claude cloud routine) — also set in the cloud environment\n"
            f"{KEY}={token}\n")
    ENV.write_text(text, encoding="utf-8", newline="\n")
    os.environ[KEY] = token
    return token, "rotated" if m else "generated"


def main() -> int:
    token, how = _token("--rotate" in sys.argv)
    sql = SQL.read_text(encoding="utf-8").replace(
        "__S6_TOKEN_SHA256__", hashlib.sha256(token.encode()).hexdigest())
    assert "__S6_TOKEN_SHA256__" not in sql
    cfg = load_config()
    db.run_transaction(cfg, lambda cur: cur.execute(sql))
    fns = db.fetch_all(cfg, """SELECT p.proname FROM pg_proc p JOIN pg_namespace n
                               ON n.oid = p.pronamespace
                               WHERE n.nspname='public' AND p.proname LIKE 's6%%' ORDER BY 1""")
    db.close_shared()
    print(f"S6 gateway applied. Token: {how} (stored in .env as {KEY}). "
          f"Functions: {', '.join(r['proname'] for r in fns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

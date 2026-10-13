"""Single database connection module. Supabase session-mode pooler (IPv4).

ONE shared connection is opened lazily and reused for the whole run. This is the pattern
the session pooler is designed for, and it fixes two hangs found in live runs:

  * connect() had NO timeout, so a momentarily-slow pooler blocked the run FOREVER;
  * S1 opened a NEW connection per article (~270 connect/close cycles on --write),
    churning the pooler until it stalled.

Every connection now has connect_timeout (fail fast instead of hang), a server-side
statement_timeout (a stuck query cannot hang), and TCP keepalives. A dead connection is
dropped so the next call transparently reconnects.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterable, Sequence

CONNECT_TIMEOUT_S = 15
STATEMENT_TIMEOUT_MS = 30000

_CONN = None  # module-level shared connection


def _pg():
    import psycopg2
    import psycopg2.extras  # noqa: F401

    return psycopg2


def _validate_dsn(dsn: str) -> None:
    """Catch the #1 misconfiguration BEFORE a cryptic 'Network is unreachable'.

    The Supabase DIRECT connection host (db.<ref>.supabase.co) is IPv6-only. It works from
    an IPv6-capable laptop but FAILS from GitHub Actions (IPv4). The Session pooler
    (aws-0-<region>.pooler.supabase.com, user postgres.<ref>) is IPv4 and works everywhere.
    """
    import re

    m = re.search(r"@([^:/?]+)", dsn or "")
    host = m.group(1) if m else ""
    if host.startswith("db.") and host.endswith(".supabase.co"):
        raise RuntimeError(
            "DATABASE_URL is the Supabase DIRECT connection host '" + host + "', which is "
            "IPv6-only and is UNREACHABLE from GitHub Actions (IPv4). Use the SESSION POOLER "
            "DSN instead: Supabase dashboard -> Connect -> Session pooler -> URI. Its host "
            "looks like 'aws-0-<region>.pooler.supabase.com' (port 5432) and its user is "
            "'postgres.<project-ref>'. Update your DATABASE_URL (locally in .env, in CI as a "
            "GitHub secret)."
        )


def _connect(config: dict):
    dsn = config.get("database_url")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set. Use the Supabase SESSION-mode pooler DSN.")
    _validate_dsn(dsn)
    conn = _pg().connect(
        dsn,
        connect_timeout=CONNECT_TIMEOUT_S,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )
    conn.autocommit = False
    return conn


def _shared(config: dict):
    global _CONN
    if _CONN is None or getattr(_CONN, "closed", 1):
        _CONN = _connect(config)
    return _CONN


def _drop():
    global _CONN
    try:
        if _CONN is not None:
            _CONN.close()
    except Exception:  # noqa: BLE001
        pass
    _CONN = None


def close_shared() -> None:
    """Close the shared connection. Call at the end of a run for a clean exit."""
    _drop()


@contextlib.contextmanager
def get_conn(config: dict):
    """Backward-compatible context manager, now backed by the shared connection."""
    conn = _shared(config)
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 - connection is dead; drop so next call reconnects
            _drop()
        raise


@contextlib.contextmanager
def get_cursor(config: dict):
    import psycopg2.extras

    conn = _shared(config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            _drop()
        raise
    finally:
        try:
            cur.close()
        except Exception:  # noqa: BLE001
            pass


def _is_transient(exc: Exception) -> bool:
    """Connection-level errors that are worth retrying (the Supabase session pooler
    intermittently refuses/drops connections). Query/logic errors are NOT retried."""
    import psycopg2
    return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))


def _with_retry(fn, attempts: int = 3, base: float = 1.0):
    """Run fn(); on a TRANSIENT connection error, drop the dead shared connection so the
    next attempt reconnects, back off, and retry. Safe because a transaction that dies on
    a connection error did not commit — and the pipeline's writes are idempotent
    (ON CONFLICT) while portal writes (usage_events/feedback) tolerate a rare duplicate."""
    import time
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc) or i == attempts - 1:
                raise
            _drop()
            last = exc
            time.sleep(min(base * (2 ** i), 6.0))
    raise last  # pragma: no cover


def fetch_all(config: dict, sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    def _do():
        with get_cursor(config) as cur:
            cur.execute(sql, params or ())
            return [dict(r) for r in cur.fetchall()]
    return _with_retry(_do)


def fetch_one(config: dict, sql: str, params: Sequence[Any] | None = None) -> dict | None:
    def _do():
        with get_cursor(config) as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return dict(row) if row else None
    return _with_retry(_do)


def execute(config: dict, sql: str, params: Sequence[Any] | None = None) -> None:
    def _do():
        with get_cursor(config) as cur:
            cur.execute(sql, params or ())
    _with_retry(_do)


def run_transaction(config: dict, fn) -> Any:
    """Run fn(cursor) as ONE transaction: all of it commits, or none of it does.

    Used where several writes must land together (S5: an event's facts plus the
    facts_extracted_at stamp on its articles). A connection drop mid-way rolls back, so a
    retry never leaves a half-written event behind (BUILD_NOTES F31).
    """
    def _do():
        with get_cursor(config) as cur:
            return fn(cur)
    return _with_retry(_do)


def execute_many(config: dict, sql: str, rows: Iterable[Sequence[Any]]) -> None:
    import psycopg2.extras

    rows = list(rows)
    if not rows:
        return

    def _do():
        with get_cursor(config) as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
    _with_retry(_do)


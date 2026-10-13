"""S1 — Ingest.

RSS-first. Full text via trafilatura, with MANDATORY graceful degradation:
if extraction fails we keep headline + RSS description and set raw_text = NULL.
A single bad source must never crash the run — every per-source failure is logged
to ingest_log and the loop continues.

Idempotent: news_articles.url is UNIQUE, so re-ingestion is a no-op via
ON CONFLICT DO NOTHING.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from . import db

# feedparser is preferred (it tolerates badly-formed feeds far better) but it depends on
# sgmllib3k, a source-only package whose build fails on Windows when antivirus locks the
# %TEMP% file. Fall back to the stdlib parser so a dependency install failure cannot
# block the pipeline. See src/rss.py.
try:
    import feedparser  # type: ignore

    FEED_BACKEND = "feedparser"
except ImportError:  # pragma: no cover - exercised only when the dep is missing
    from . import rss as feedparser  # type: ignore

    FEED_BACKEND = "stdlib"

log = logging.getLogger(__name__)

# Several Indian publishers (Moneycontrol notably) return HTTP 403 to bot-style
# user-agents but serve the same public RSS fine to a browser UA. Verified during
# Phase 1 build: the identical Moneycontrol URLs 403 with a bot UA and 200 with this one.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

INSERT_SQL = """
INSERT INTO news_articles (url, source, title, published_at, raw_text, rss_summary, status)
VALUES (%s, %s, %s, %s, %s, %s, 'new')
ON CONFLICT (url) DO NOTHING
RETURNING id
"""

LOG_SQL = """
INSERT INTO ingest_log (source, articles_fetched, failures, error_detail)
VALUES (%s, %s, %s, %s)
"""


def _parse_published(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        tm = getattr(entry, key, None) or entry.get(key) if hasattr(entry, "get") else None
        if tm:
            try:
                return datetime(*tm[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


_TRAFILATURA_WARNED = False


def _silence_trafilatura() -> None:
    """trafilatura logs an ERROR per failed download, which becomes a wall of red on a
    corporate network and makes the run look hung. We handle failures ourselves, so mute it."""
    import logging as _logging

    for name in ("trafilatura", "trafilatura.downloads", "trafilatura.core",
                 "trafilatura.utils"):
        _logging.getLogger(name).setLevel(_logging.CRITICAL)


def _fetch_html(url: str, timeout: int = 20) -> str | None:
    """Download raw HTML via PYTHON's SSL, not trafilatura's pycurl.

    Corporate HTTPS inspection re-signs traffic with a private root. pycurl (schannel)
    rejects it -> 'CERT_TRUST_IS_UNTRUSTED_ROOT'. Python's SSL, once truststore has
    injected the OS cert store (config.install_ssl_truststore, called in load_config),
    DOES trust that root. So fetching here and only PARSING with trafilatura works behind
    the proxy where trafilatura's own downloader fails.
    """
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - degradation is normal
        return None


def _fetch_html_step(url: str, config: dict | None = None) -> tuple[str | None, str | None]:
    """Fetch stage ONLY (network I/O — thread-safe, safe to run in parallel).

    Returns (html, error). Never raises. The lxml PARSE is deliberately NOT done here:
    trafilatura/lxml is a C library that segfaults when run concurrently across threads,
    so parsing is done sequentially on the main thread by _parse_fulltext().
    """
    if config is not None and not config.get("extract_fulltext", True):
        return None, "SKIP:extract_fulltext-disabled"
    html = _fetch_html(url)
    if not html:
        return None, "SKIP:fetch-failed"              # SKIP => counted as degraded, not failure
    return html, None


def _parse_fulltext(html: str | None) -> tuple[str | None, str | None]:
    """Parse stage ONLY (lxml/libxml2 — MUST run single-threaded on the main thread).

    Returns (text, error). Never raises — degradation is normal.
    """
    global _TRAFILATURA_WARNED
    if not html:
        return None, "SKIP:no-html"
    try:
        try:
            import trafilatura
        except ImportError:
            if not _TRAFILATURA_WARNED:
                log.warning("trafilatura not installed — keeping headline + RSS summary "
                            "only (raw_text NULL). Pipeline continues normally.")
                _TRAFILATURA_WARNED = True
            return None, "SKIP:trafilatura-not-installed"

        _silence_trafilatura()
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text or len(text.strip()) < 200:
            return None, "SKIP:text-too-short"
        return text.strip(), None
    except Exception as exc:  # noqa: BLE001 - degradation is mandatory here
        return None, f"SKIP:{type(exc).__name__}"


def _extract_fulltext(url: str, config: dict | None = None) -> tuple[str | None, str | None]:
    """Fetch + parse one article, sequentially. Returns (text, error). Never raises.

    Kept for callers/tests that want the whole thing in one call (e.g. diagnostics). The
    S1 run() no longer uses this — it fetches in parallel then parses on the main thread.
    """
    if config is not None and not config.get("extract_fulltext", True):
        return None, "SKIP:extract_fulltext-disabled"
    html, err = _fetch_html_step(url, config)
    if err:
        return None, err
    return _parse_fulltext(html)


def _fetch_feed(src: dict):
    """Parse one feed. Returns (src, feed, error). Never raises."""
    try:
        feed = feedparser.parse(src.get("url"), agent=USER_AGENT)
        if getattr(feed, "bozo", 0) and not feed.entries:
            return src, None, RuntimeError(
                f"feed unparseable: {getattr(feed, 'bozo_exception', 'unknown')}")
        return src, feed, None
    except Exception as exc:  # noqa: BLE001
        return src, None, exc


def run(config: dict, **_inputs) -> dict:
    """Fetch all enabled sources. Returns a run summary.

    Feed fetches AND per-article full-text extraction are network I/O and are run
    CONCURRENTLY (they used to be sequential — the slow part of S1). Database writes stay
    on the main thread so the single shared connection is never shared across threads.
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor

    timeout = int(config.get("network_timeout_s", 20))
    socket.setdefaulttimeout(timeout)
    workers = int(config.get("ingest_workers", 8))

    sources = config.get("sources", [])
    total_new = total_seen = total_failures = total_degraded = 0
    per_source: list[dict] = []
    if not sources:
        return {"stage": "s1_ingest", "backend": FEED_BACKEND, "sources": 0,
                "articles_seen": 0, "articles_new": 0, "failures": 0,
                "degraded_no_fulltext": 0, "per_source": []}

    # 1) fetch all feeds in parallel — wall-clock drops from SUM(feed times) to ~slowest feed.
    log.info("S1 fetching %d sources in parallel (%d workers) ...", len(sources), workers)
    with ThreadPoolExecutor(max_workers=min(workers, len(sources))) as ex:
        fetched_feeds = list(ex.map(_fetch_feed, sources))

    # 2) collect every article link, then extract full text for all of them in parallel.
    jobs = []  # (src, entry, link, title)
    for src, feed, err in fetched_feeds:
        if err is not None or feed is None:
            continue
        for entry in feed.entries:
            link = (entry.get("link") or "").strip()
            title = (entry.get("title") or "").strip()
            if link and title:
                jobs.append((src, entry, link, title))

    # Two-phase extraction: FETCH in parallel (network I/O, thread-safe), then PARSE
    # sequentially on the main thread. trafilatura/lxml is a C library that segfaults
    # under concurrent use, so only the fetch is threaded — the wall-clock cost is the
    # network wait anyway, so this keeps ~all the speedup without the crash.
    extracted: dict[str, tuple] = {}
    if jobs and config.get("extract_fulltext", True):
        links = [j[2] for j in jobs]
        log.info("S1 fetching HTML for %d articles in parallel (%d workers) ...",
                 len(links), workers)
        html_by_link: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for link, res in zip(links, ex.map(lambda ln: _fetch_html_step(ln, config), links)):
                html_by_link[link] = res
        log.info("S1 parsing %d articles (sequential — lxml is not thread-safe) ...", len(links))
        for link in links:
            html, ferr = html_by_link[link]
            if ferr:                       # fetch failed/disabled — carry the SKIP through
                extracted[link] = (None, ferr)
            else:
                extracted[link] = _parse_fulltext(html)

    # 3) per-source accounting + sequential DB writes (main thread only).
    for i, (src, feed, fetch_err) in enumerate(fetched_feeds, 1):
        name = src.get("name")
        fetched = failures = degraded = 0
        errors: list[str] = []
        new_ids: list[int] = []

        if fetch_err is not None or feed is None:
            failures += 1
            errors.append(f"source-level: {type(fetch_err).__name__}: {fetch_err}")
            log.warning("S1 source failed: %s (%s)", name, fetch_err)
        else:
            for entry in feed.entries:
                link = (entry.get("link") or "").strip()
                title = (entry.get("title") or "").strip()
                if not link or not title:
                    continue
                fetched += 1
                summary = (entry.get("summary") or entry.get("description") or "").strip() or None
                published = _parse_published(entry)

                raw_text, err = extracted.get(link, (None, "SKIP:extract_fulltext-disabled"))
                if err and err.startswith("SKIP:"):
                    degraded += 1
                elif err:
                    failures += 1
                    errors.append(f"{link}: {err}")

                if config.get("dry_run"):
                    continue
                try:
                    row = db.fetch_one(config, INSERT_SQL,
                                       (link, name, title, published, raw_text, summary))
                    if row:  # None => ON CONFLICT skipped an already-ingested url
                        new_ids.append(row["id"])
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    errors.append(f"db insert {link}: {type(exc).__name__}: {exc}")

        total_seen += fetched
        total_new += len(new_ids)
        total_failures += failures
        total_degraded += degraded
        log.info("S1 [%d/%d] %s: fetched=%d new=%d failures=%d degraded=%d",
                 i, len(fetched_feeds), name, fetched, len(new_ids), failures, degraded)
        per_source.append({"source": name, "fetched": fetched, "new": len(new_ids),
                           "failures": failures, "degraded": degraded})

        if not config.get("dry_run"):
            detail = "\n".join(errors)[:4000] if errors else None
            try:
                db.execute(config, LOG_SQL, (name, fetched, failures, detail))
            except Exception as exc:  # noqa: BLE001
                log.warning("S1 could not write ingest_log for %s: %s", name, exc)

    return {
        "stage": "s1_ingest",
        "backend": FEED_BACKEND,
        "sources": len(sources),
        "articles_seen": total_seen,
        "articles_new": total_new,
        "failures": total_failures,
        "degraded_no_fulltext": total_degraded,
        "per_source": per_source,
    }

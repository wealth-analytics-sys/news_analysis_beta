"""Minimal RSS/Atom parser using only the standard library.

Why this exists: `feedparser` depends on `sgmllib3k`, which ships as a source tarball with
no wheel. On Windows that build step frequently dies with a permission error when antivirus
locks the file in %TEMP% — which blocked this project twice. Rather than let a hard
dependency gate the whole pipeline, S1 uses feedparser when available and falls back here.

This handles the ~95% case: well-formed RSS 2.0 and Atom, which is what every source in
sources.yaml serves. feedparser remains preferred because it tolerates genuinely broken
feeds far better; this is a safety net, not a replacement.
"""
from __future__ import annotations

import email.utils
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ATOM = "{http://www.w3.org/2005/Atom}"


class _Entry(dict):
    """dict that also allows attribute access, so callers can use e.get() or e['x']."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Feed:
    def __init__(self, entries, status=None, bozo=0, bozo_exception=None):
        self.entries = entries
        self.status = status
        self.bozo = bozo
        self.bozo_exception = bozo_exception


def _text(node, *paths) -> str:
    for p in paths:
        found = node.find(p)
        if found is not None:
            if found.text and found.text.strip():
                return found.text.strip()
            href = found.get("href")
            if href:
                return href.strip()
    return ""


def _parse_date(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    try:                                    # RFC 822 — RSS pubDate
        dt = email.utils.parsedate_to_datetime(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw.replace("Z", "+0000"), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse(url: str, agent: str = "Mozilla/5.0", timeout: int = 25) -> _Feed:
    """Fetch and parse a feed. Never raises — mirrors feedparser's bozo behaviour."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001
        return _Feed([], status=None, bozo=1, bozo_exception=exc)

    try:
        root = ET.fromstring(raw)
    except Exception as exc:  # noqa: BLE001
        return _Feed([], status=status, bozo=1, bozo_exception=exc)

    entries: list[_Entry] = []

    for item in root.iter("item"):                       # RSS 2.0
        published = _text(item, "pubDate", "date")
        entries.append(_Entry(
            title=_text(item, "title"),
            link=_text(item, "link", "guid"),
            summary=_text(item, "description", "summary"),
            published=published,
            published_parsed=(_parse_date(published).timetuple()
                              if _parse_date(published) else None),
        ))

    for item in root.iter(f"{ATOM}entry"):               # Atom
        published = _text(item, f"{ATOM}published", f"{ATOM}updated")
        entries.append(_Entry(
            title=_text(item, f"{ATOM}title"),
            link=_text(item, f"{ATOM}link", f"{ATOM}id"),
            summary=_text(item, f"{ATOM}summary", f"{ATOM}content"),
            published=published,
            published_parsed=(_parse_date(published).timetuple()
                              if _parse_date(published) else None),
        ))

    return _Feed(entries, status=status, bozo=0 if entries else 1)

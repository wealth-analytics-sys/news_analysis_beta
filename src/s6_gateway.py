"""HTTPS client for the S6 gateway (BUILD_NOTES D17).

The Claude Code cloud routine cannot open a Postgres connection (Anthropic-hosted sessions
only allow HTTP(S) through their proxy), so in the cloud run_s6.py talks to Supabase through
the four gateway functions in sql/migrations/2026-10-06_s6_gateway.sql, called as
POST {SUPABASE_URL}/rest/v1/rpc/<fn>. Standard library only (urllib honours HTTPS_PROXY).

Config: SUPABASE_URL, SUPABASE_ANON_KEY (the project's anon/publishable API key) and
S6_GATEWAY_TOKEN. None of them grants direct table access.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class GatewayError(RuntimeError):
    """The gateway refused or failed a call; .message carries Postgres's reason."""


class Gateway:
    def __init__(self, url: str, anon_key: str, token: str, timeout: float = 60.0,
                 retries: int = 3, opener=None):
        missing = [n for n, v in (("SUPABASE_URL", url), ("SUPABASE_ANON_KEY", anon_key),
                                  ("S6_GATEWAY_TOKEN", token)) if not v]
        if missing:
            raise GatewayError(f"gateway mode needs {', '.join(missing)} in the environment")
        self.base = url.rstrip("/") + "/rest/v1/rpc/"
        self.anon_key, self.token = anon_key, token
        self.timeout, self.retries = timeout, retries
        self._open = opener or urllib.request.urlopen

    @classmethod
    def from_config(cls, config: dict) -> "Gateway":
        return cls(config.get("supabase_url") or "", config.get("supabase_anon_key") or "",
                   config.get("s6_gateway_token") or "")

    def _rpc(self, fn: str, **args):
        body = json.dumps({"p_token": self.token, **args}, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.base + fn, data=body, method="POST",
            headers={"apikey": self.anon_key, "Authorization": f"Bearer {self.anon_key}",
                     "Content-Type": "application/json", "Accept": "application/json"})
        last = None
        for attempt in range(self.retries):
            try:
                with self._open(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                try:
                    detail = json.loads(detail).get("message") or detail
                except (ValueError, AttributeError):
                    pass
                if exc.code >= 500 and attempt < self.retries - 1:
                    last = GatewayError(f"{fn}: HTTP {exc.code}: {detail}")
                    time.sleep(2 ** attempt)
                    continue
                raise GatewayError(f"{fn}: HTTP {exc.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                last = GatewayError(f"{fn}: network error: {exc}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        raise last  # pragma: no cover

    # ---------------------------------------------------------------- operations
    def pending(self) -> list[dict]:
        return self._rpc("s6_pending") or []

    def bundle(self, event_id: int) -> tuple[dict, dict]:
        """(inputs, aliases): inputs has the same keys as s6_interpret.build_inputs."""
        from .s2_filter import primary_category_for_event_key

        b = self._rpc("s6_bundle", p_event_id=int(event_id))
        aliases = b.pop("ALIASES", {}) or {}
        b["EVENT_META"]["suggested_category"] = primary_category_for_event_key(
            b["EVENT_META"].get("event_key") or "")
        return b, aliases

    def write_card(self, event_id: int, payload: dict, model_version: str,
                   citation_warnings: list | None) -> dict:
        return self._rpc("s6_write_card", p_event_id=int(event_id), p_payload=payload,
                         p_model_version=model_version,
                         p_citation_warnings=citation_warnings or None)

    def skip(self, event_ids: list[int], reason: str) -> list[dict]:
        return self._rpc("s6_skip", p_event_ids=[int(e) for e in event_ids], p_reason=reason)

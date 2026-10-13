"""S6 HTTPS gateway (BUILD_NOTES D17): client wiring and the gateway write path.
Fakes only — no network, no DB."""
import io
import json
import urllib.error

import pytest

from src import s6_interpret as s6
from src.s6_gateway import Gateway, GatewayError


class _Resp:
    def __init__(self, body):
        self._b = json.dumps(body).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _gw(responses, seen):
    def opener(req, timeout):
        seen.append(req)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)
    return Gateway("https://x.supabase.co", "anon-key", "tok", retries=2, opener=opener)


def test_gateway_requires_all_three_settings():
    with pytest.raises(GatewayError, match="SUPABASE_ANON_KEY, S6_GATEWAY_TOKEN"):
        Gateway("https://x.supabase.co", "", "")


def test_rpc_posts_token_and_api_key_to_the_rpc_endpoint():
    seen = []
    gw = _gw([[{"event_id": 1, "title": "t", "needs_reinterpretation": False,
                "articles": 2, "facts": 3}]], seen)
    assert gw.pending()[0]["event_id"] == 1
    req = seen[0]
    assert req.full_url == "https://x.supabase.co/rest/v1/rpc/s6_pending"
    assert req.get_header("Apikey") == "anon-key"
    assert req.get_header("Authorization") == "Bearer anon-key"
    assert json.loads(req.data) == {"p_token": "tok"}


def test_postgres_rejection_surfaces_as_gateway_error_with_reason():
    err = urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(
        json.dumps({"message": "s6 gateway: card rejected: contains a URL"}).encode()))
    gw = _gw([err], [])
    with pytest.raises(GatewayError, match="contains a URL"):
        gw.write_card(7, {}, "m", [])


def test_server_errors_are_retried():
    err = urllib.error.HTTPError("u", 503, "Unavailable", {}, io.BytesIO(b"{}"))
    gw = _gw([err, []], [])
    assert gw.pending() == []


def test_bundle_splits_aliases_and_adds_suggested_category():
    seen = []
    gw = _gw([{"EVENT_META": {"event_id": 7, "event_key": "rbi-policy-oct2026", "title": "t"},
               "ARTICLE_SUMMARIES": [], "EVENT_FACTS": [], "KB_FACTS": [],
               "PRIOR_INTERPRETATION": None, "TRANSIENT_CONTEXT": [],
               "ALIASES": {"inr": ["rupee"]}}], seen)
    inputs, aliases = gw.bundle(7)
    assert aliases == {"inr": ["rupee"]} and "ALIASES" not in inputs
    assert inputs["EVENT_META"]["suggested_category"] == "indian_economy"
    assert json.loads(seen[0].data) == {"p_token": "tok", "p_event_id": 7}


# ---------------------------------------------------------------- write path
CARD = {"event_title": "RBI holds", "event_type": "policy_regulation", "summary": "s",
        "primary_category": "indian_economy",
        "impact_chains": [{"chain": [{"from": "repo_rate", "to": "bank_nim",
                                      "mechanism": "repo rate decision sets bank funding cost",
                                      "fact_ids": [1, 99]}],
                           "terminal_impact": {"asset": "banks", "direction": "mixed",
                                               "magnitude": "small", "conditionality": "if held"}}],
        "rm_one_liner": "The RBI kept rates unchanged in early October 2026."}
INPUTS = {"EVENT_META": {"event_key": "rbi-policy-oct2026"},
          "EVENT_FACTS": [{"fact_id": 1, "metric": "repo_rate", "entities": ["rbi"]}],
          "KB_FACTS": [], "PRIOR_INTERPRETATION": {"summary": "old"}}


class _FakeGw:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def write_card(self, event_id, payload, model_version, warnings):
        if self.fail:
            raise GatewayError(self.fail)
        self.calls.append((event_id, payload, model_version, warnings))
        return {"event_id": event_id, "version": 2}


def test_gateway_write_runs_the_same_gates_then_inserts():
    gw = _FakeGw()
    out = s6.write_card_via_gateway(gw, 7, CARD, "claude-code-agent", INPUTS, {})
    assert out["version"] == 2 and out["citations_stripped"] == 1     # 99 not in the facts
    _, payload, mv, warnings = gw.calls[0]
    assert payload["impact_chains"][0]["chain"][0]["fact_ids"] == [1]
    assert payload["is_update"] is True and mv == "claude-code-agent"
    assert warnings[0]["fact_id"] == 99


def test_gateway_check_mode_writes_nothing():
    gw = _FakeGw()
    out = s6.write_card_via_gateway(gw, 7, CARD, "m", INPUTS, {}, dry_run=True)
    assert gw.calls == [] and out["dry_run"] is True and out["version"] is None


def test_gateway_write_rejects_wording_before_any_network_call():
    gw = _FakeGw()
    bad = {**CARD, "rm_one_liner": "You should consider reducing bank exposure."}
    with pytest.raises(s6.CardRejected):
        s6.write_card_via_gateway(gw, 7, bad, "m", INPUTS, {})
    assert gw.calls == []


def test_server_side_rejection_becomes_card_rejected():
    gw = _FakeGw(fail="s6_write_card: HTTP 400: s6 gateway: card rejected: event is closed, not open")
    with pytest.raises(s6.CardRejected) as exc:
        s6.write_card_via_gateway(gw, 7, CARD, "m", INPUTS, {})
    assert exc.value.problems[0]["kind"] == "gateway" and "closed" in exc.value.problems[0]["error"]

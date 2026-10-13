"""Regression tests for the Sep-2026 backlog-recovery flags (BUILD_NOTES F22-F33).

Every case below is taken from real data seen in that run. Pure logic — no DB, no API.
"""
import types

import pytest

from src import s2_filter, s4_extract, s5_fact_ingest
from src.providers import AnthropicLLM, TruncatedOutputError
from src.s5_fact_ingest import consolidate, normalise
from src.s6_interpret import validate_citations


def _f(article_id, metric, value, period="2026-08", unit=None, **kw):
    d = {"article_id": article_id, "metric": metric, "value": value, "as_of_period": period,
         "unit": unit, "claim_context": "ctx", "category": "macro_economics",
         "entities": ["india"], "fact_type": "point_in_time",
         "persistence_tier": "event_context", "verification_status": "single_source",
         "confidence": "high"}
    d.update(kw)
    return d


# ---------------------------------------------------------------- F29: S5 subject/units
def test_normalise_indian_and_si_scales():
    assert normalise("24.51", "billion")[0] == pytest.approx(2.451e10)
    assert normalise("2451", "crore")[0] == pytest.approx(2.451e10)
    assert normalise("12.1", "lakh crore")[0] == pytest.approx(1.21e13)
    assert normalise("$42.8 billion", "USD billion")[1] == "usd"
    assert normalise("strong", None) == (None, None)


def test_unit_mismatch_same_count_is_consistent_not_a_discrepancy():
    """893: UPI Aug-2026 volume as 24.51 billion and as 2,451 crore — the same number."""
    out = consolidate([_f(8737, "upi_transactions_processed_monthly", "24.51", unit="billion"),
                       _f(8998, "upi_transactions_processed_monthly", "2451", unit="crore")])
    assert len(out) == 1
    assert out[0]["verification_status"] == "cross_source_consistent"
    assert out[0]["conflicting_values"] is None


def test_one_article_with_several_values_is_several_subjects():
    """1011: per-state FDI all keyed fdi_inflow_by_state, entities ['india'] on every row.
    Article 10370 states two values, so the group mixes subjects — never a discrepancy."""
    facts = [_f(10370, "fdi_inflow_by_state", "5.9", "Q1 FY27", unit="USD billion"),
             _f(10370, "fdi_inflow_by_state", "4.2", "Q1 FY27", unit="USD billion"),
             _f(10395, "fdi_inflow_by_state", "5.95", "Q1 FY27", unit="USD billion"),
             _f(10395, "fdi_inflow_by_state", "4.22", "Q1 FY27", unit="USD billion"),
             _f(10395, "fdi_inflow_by_state", "2.67", "Q1 FY27", unit="USD billion")]
    out = consolidate(facts)
    assert not any(f["verification_status"] == "discrepancy_flagged" for f in out)
    by_val = {f["value"]: f for f in out}
    assert by_val["5.9"]["verification_status"] == "cross_source_consistent"   # TN, 2 sources
    assert by_val["5.9"]["supporting_article_ids"] == [10370, 10395]
    assert by_val["4.2"]["verification_status"] == "cross_source_consistent"   # MH, 2 sources
    assert by_val["2.67"]["verification_status"] == "single_source"
    assert len(out) == 3


def test_paraphrased_text_values_are_not_a_discrepancy():
    """975: 'slowest in 33 months' vs 'weakest in around three years'."""
    out = consolidate([_f(9530, "export_orders_growth_pace", "slowest in 33 months"),
                       _f(9801, "export_orders_growth_pace", "weakest in around three years")])
    assert [f["verification_status"] for f in out] == ["single_source", "single_source"]


def test_different_currencies_are_not_compared():
    """1006: NZ$7 billion vs Rs 35,000 crore — the same target restated, not a conflict."""
    out = consolidate([_f(9013, "india_nz_bilateral_trade_target_2030", "NZ$7 billion"),
                       _f(9308, "india_nz_bilateral_trade_target_2030", "₹35,000 crore")])
    assert not any(f["verification_status"] == "discrepancy_flagged" for f in out)


def test_genuine_same_quantity_conflict_is_still_flagged_with_units():
    """1039: yen move on 18-Sep reported as -0.5% and -0.6% — a real disagreement."""
    out = consolidate([_f(8712, "jpy_usd_change_pct", "-0.5", "18-Sep-2026", unit="%"),
                       _f(8665, "jpy_usd_change_pct", "-0.6", "18-Sep-2026", unit="%")])
    assert len(out) == 1
    assert out[0]["verification_status"] == "discrepancy_flagged"
    vals = {(v["article_id"], v["value"], v["unit"]) for v in out[0]["conflicting_values"]}
    assert vals == {(8712, "-0.5", "%"), (8665, "-0.6", "%")}


def test_s5_skips_failed_extractions_and_reports_them():
    """A partial/failed S4 event must not be ingested (nor its articles stamped)."""
    res = s5_fact_ingest.run(
        {"dry_run": True},
        extraction={"events": [
            {"event_id": 1, "article_ids": [10], "facts": [], "transient_context": [],
             "error": "TruncatedOutputError: stopped"},
            {"event_id": 2, "article_ids": [20], "transient_context": [],
             "facts": [_f(20, "repo_rate", "5.5")]},
        ]})
    assert res["skipped_failed_extraction"] == [1]
    assert res["facts_consolidated"] == 1


# ---------------------------------------------------------------- F23 / F27: S4
def test_coerce_items_decodes_string_encoded_array_with_trailing_text():
    data = {"facts": '[{"metric": "a"}, {"metric": "b"}]\n, "transient_context": []'}
    assert [f["metric"] for f in s4_extract._coerce_items(data, "facts")] == ["a", "b"]


def test_coerce_items_drops_non_objects_and_rejects_garbage():
    assert s4_extract._coerce_items({"facts": [{"m": 1}, "stray"]}, "facts") == [{"m": 1}]
    with pytest.raises(ValueError):
        s4_extract._coerce_items({"facts": "not json at all"}, "facts")


def test_chunk_bundle_splits_large_events():
    b = {"event_id": 1, "articles": [{"article_id": i} for i in range(12)]}
    chunks = s4_extract._chunk_bundle(b, 5)
    assert [len(c["articles"]) for c in chunks] == [5, 5, 2]
    assert s4_extract._chunk_bundle(b, 0) == [b]


class _FakeLLM:
    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def complete_json(self, **kw):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise TruncatedOutputError("stopped at max_tokens")
        assert kw["max_tokens"] == s4_extract.EXTRACT_MAX_TOKENS
        return types.SimpleNamespace(
            data={"facts": [{"metric": f"m{self.calls}", "value": "1"}]}, model="fake")


def _bundle(n):
    return {"event_id": 7, "event_key": "k", "title": "t",
            "articles": [{"article_id": i, "source": "s", "url": f"u{i}", "published_at": None,
                          "title": "x", "body_or_summary": "y"} for i in range(n)]}


def test_s4_merges_chunk_results_into_one_event(monkeypatch):
    llm = _FakeLLM()
    monkeypatch.setattr(s4_extract, "get_llm", lambda cfg, tier="cheap": llm)
    monkeypatch.setattr(s4_extract, "load_prompt",
                        lambda n: "SYS\n======================================================================\nRUNTIME INPUTS\n======================================================================\n{{ARTICLES}}")
    res = s4_extract.run({}, bundles=[_bundle(11)], dictionaries=([], []))
    ev = res["events_out"][0]
    assert llm.calls == 3 and len(ev["facts"]) == 3
    assert ev["article_ids"] == list(range(11)) and "error" not in ev


def test_s4_is_all_or_nothing_when_a_chunk_truncates(monkeypatch):
    monkeypatch.setattr(s4_extract, "get_llm", lambda cfg, tier="cheap": _FakeLLM(fail_on_call=2))
    monkeypatch.setattr(s4_extract, "load_prompt",
                        lambda n: "SYS\n======================================================================\nRUNTIME INPUTS\n======================================================================\n{{ARTICLES}}")
    res = s4_extract.run({}, bundles=[_bundle(11)], dictionaries=([], []))
    ev = res["events_out"][0]
    assert res["failures"] == 1 and ev["facts"] == [] and "TruncatedOutputError" in ev["error"]


def test_anthropic_truncation_raises_instead_of_returning_empty():
    """F27: stop_reason=max_tokens came back as a tool_use with only event_id."""
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._default_model, llm._omit_temperature, llm._auto_tool_choice = "m", True, False
    resp = types.SimpleNamespace(stop_reason="max_tokens", content=[
        types.SimpleNamespace(type="tool_use", input={"event_id": 889})])
    llm._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **k: resp))
    with pytest.raises(TruncatedOutputError):
        llm.complete_json(system="s", user="u", json_schema={}, max_tokens=8000)


# ---------------------------------------------------------------- F30: validator
def _hop_card(fid, frm, to, mech):
    return {"impact_chains": [{"chain": [
        {"from": frm, "to": to, "mechanism": mech, "fact_ids": [fid]}]}]}


def test_validator_no_longer_passes_on_india_alone():
    facts = {1: {"fact_id": 1, "metric": "india_gdp_growth_yoy", "entities": ["india"]}}
    card, warns = validate_citations(
        _hop_card(1, "crude_price", "india_import_bill", "India pays more for imported oil"),
        facts)
    assert card["impact_chains"][0]["chain"][0]["fact_ids"] == [] and len(warns) == 1


def test_validator_uses_entity_aliases():
    """1090: an INR fact on a hop that says 'rupee' was stripped for lack of the slug."""
    facts = {5: {"fact_id": 5, "metric": "usd_inr_level", "entities": ["inr"]}}
    card = _hop_card(5, "dollar_strength", "rupee_weakness", "A firmer dollar pushes the rupee down")
    stripped, _ = validate_citations(card, facts)
    assert stripped["impact_chains"][0]["chain"][0]["fact_ids"] == []      # no alias -> no match
    card = _hop_card(5, "dollar_strength", "rupee_weakness", "A firmer dollar pushes the rupee down")
    kept, warns = validate_citations(card, facts, aliases={"inr": ["rupee", "indian rupee"]})
    assert kept["impact_chains"][0]["chain"][0]["fact_ids"] == [5] and warns == []


# ---------------------------------------------------------------- F33: S2 fails loudly
def test_s2_raises_when_every_triage_batch_fails(monkeypatch):
    arts = [{"id": i, "url": f"u{i}", "source": "s", "title": f"Distinct headline {i}",
             "rss_summary": "x", "raw_text": None} for i in range(3)]
    monkeypatch.setattr(s2_filter.db, "fetch_all", lambda cfg, sql, params=(): arts)
    monkeypatch.setattr(s2_filter, "_pass_a", lambda a, cfg: (a, []))
    monkeypatch.setattr(s2_filter, "_pass_b", lambda cfg, a: ({}, [x["id"] for x in a]))
    with pytest.raises(RuntimeError, match="ALL 3"):
        s2_filter.run({"dry_run": True})


def test_validator_uses_metric_aliases():
    """1010: import_duty_rate cited on a 'tariff' hop — valid, matched via metric alias."""
    facts = {9: {"fact_id": 9, "metric": "import_duty_rate", "entities": ["usa", "india"]}}
    card = _hop_card(9, "us_sanctions_law", "tariff_risk_on_indian_goods",
                     "The law authorises tariffs of up to 100% on Indian goods")
    kept, warns = validate_citations(card, facts, aliases={"import_duty_rate": ["tariff", "customs duty"]})
    assert kept["impact_chains"][0]["chain"][0]["fact_ids"] == [9] and warns == []


# ---------------------------------------------------------------- F35: news-time recency
def test_event_recency_follows_news_time_not_processing_time():
    from datetime import datetime, timezone
    from src.s3_cluster import _news_time
    old = datetime(2026, 9, 17, 8, 0, tzinfo=timezone.utc)
    newer = datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)
    assert _news_time([{"cluster_time": old}, {"published_at": newer}]) == newer
    assert _news_time([{"title": "no times"}]) is None        # caller falls back to now()


# ---------------------------------------------------------------- F36: date sanity
def test_fact_dated_after_publication_is_downgraded_not_dropped():
    """1000: a 'Friday' close resolved to 26-Sep-2026 (a Saturday) in a 25-Sep article."""
    arts = [{"article_id": 10434, "published_at": "2026-09-25T13:00:00+00:00"}]
    facts = [{"article_id": 10434, "metric": "nifty50_close", "value": "23000",
              "as_of_period": "2026-09-26", "confidence": "high"},
             {"article_id": 10434, "metric": "nifty50_close", "value": "23000",
              "as_of_period": "2026-09-25", "confidence": "high"}]
    assert s4_extract._flag_post_publication_dates(facts, arts) == 1
    assert facts[0]["confidence"] == "low" and "after publication" in facts[0]["date_check"]
    assert facts[1]["confidence"] == "high" and "date_check" not in facts[1]


# ---------------------------------------------------------------- F25: merge helper
def test_merge_events_dry_run_is_a_no_op_that_reports_the_plan():
    from src.s3_cluster import merge_events
    res = merge_events({"dry_run": True}, 1004, [934, 1004, 1048, 934])
    assert res == {"primary": 1004, "merged": [934, 1048], "dry_run": True}
    assert merge_events({"dry_run": True}, 1004, [1004]) == {"primary": 1004, "merged": []}


def test_parse_day_understands_canonical_periods():
    from datetime import date
    assert s4_extract._parse_day("2026-09-28 close") == date(2026, 9, 28)
    assert s4_extract._parse_day("asof:2026-09-17") == date(2026, 9, 17)
    assert s4_extract._parse_day("2026-09-25T13:00:00+00:00") == date(2026, 9, 25)
    for not_a_day in ("2026-08", "FY27-Q1", "CY2026", "2026-04..2026-08", None, "Friday"):
        assert s4_extract._parse_day(not_a_day) is None


def test_different_measures_are_not_compared():
    """974 (revised prompt A/B): repo_rate_forecast as '25 bps' (hike size) vs '5.5 %'
    (level) is two quantities, not a discrepancy. Same money measure still compares."""
    out = consolidate([_f(9536, "repo_rate_forecast", "25", "FY27", unit="bps"),
                       _f(9576, "repo_rate_forecast", "5.5", "FY27", unit="%")])
    assert not any(f["verification_status"] == "discrepancy_flagged" for f in out)
    out = consolidate([_f(9050, "corporate_advance_tax_collections", "4.16", "2026-09-17", unit="INR trillion"),
                       _f(9019, "corporate_advance_tax_collections", "4.16", "2026-09-17", unit="INR lakh crore")])
    assert len(out) == 1 and out[0]["verification_status"] == "cross_source_consistent"


# ---------------------------------------------------------------- D15: S6 write gate
from src import s6_interpret as s6  # noqa: E402


def test_lint_flags_advice_and_relative_time_but_not_description():
    """Real 30-Sep one-liners: advisory phrasing and reader-relative time are flagged;
    descriptive 'sell'/'buy' (companies that sell to the US) is not."""
    card = {"rm_one_liner": "It may be worth reviewing how much of your portfolio depends on oil.",
            "summary": "So far the selloff has hit exporters; companies that sell to the US fell.",
            "footnotes": {"*": "today"}}
    kinds = sorted({i["kind"] for i in s6.lint_wording(card)})
    assert kinds == ["advice", "relative_time"]
    assert all(i["path"] != "footnotes.*" for i in s6.lint_wording(card))
    assert s6.lint_wording({"summary": "Indian exporters that sell to America may gain."}) == []


def _gate_fakes(monkeypatch, status="open", prior=None):
    writes = []

    def fetch_one(cfg, sql, params=()):
        if "FROM news_events" in sql:
            return {"id": 7, "event_key": "rbi-policy-sep2026", "title": "t", "status": status}
        return {"v": prior}

    monkeypatch.setattr(s6.db, "fetch_one", fetch_one)
    monkeypatch.setattr(s6, "_load_entity_aliases", lambda cfg: {})
    monkeypatch.setattr(s6.db, "run_transaction",
                        lambda cfg, fn: fn(type("C", (), {"execute": lambda self, q, p=None: writes.append((q, p))})()))
    return writes


def _card(**kw):
    c = {"event_title": "RBI holds", "event_type": "policy_regulation", "summary": "s",
         "primary_category": "indian_economy",
         "impact_chains": [{"chain": [{"from": "repo_rate", "to": "bank_nim",
                                       "mechanism": "repo rate decision sets bank funding cost",
                                       "fact_ids": [1]}],
                            "terminal_impact": {"asset": "banks", "direction": "mixed",
                                                "magnitude": "small", "conditionality": "if held"}}],
         "rm_one_liner": "The RBI kept rates unchanged in late September 2026."}
    c.update(kw)
    return c


INPUTS = {"EVENT_FACTS": [{"fact_id": 1, "metric": "repo_rate", "entities": ["rbi"]}], "KB_FACTS": []}


def test_write_card_inserts_next_version_and_clears_flag(monkeypatch):
    writes = _gate_fakes(monkeypatch, prior=2)
    out = s6.write_card({}, 7, _card(), "claude-code-agent", INPUTS)
    assert out["version"] == 3 and out["citations_stripped"] == 0
    assert "INSERT INTO event_interpretations" in writes[0][0]
    payload = __import__("json").loads(writes[0][1][3])
    assert payload["is_update"] is True                       # prior version existed
    assert "needs_reinterpretation=false" in writes[1][0]


def test_write_card_strict_rejects_wording_api_mode_records_it(monkeypatch):
    _gate_fakes(monkeypatch)
    bad = _card(rm_one_liner="You should consider reducing bank exposure this week.")
    with pytest.raises(s6.CardRejected) as exc:
        s6.write_card({}, 7, bad, "claude-code-agent", INPUTS, strict=True)
    assert {p["kind"] for p in exc.value.problems} == {"advice", "relative_time"}
    out = s6.write_card({}, 7, bad, "claude-sonnet-5", INPUTS, strict=False)
    assert out["wording_flags"] == 3      # "you should", "consider reducing", "this week"


def test_write_card_rejects_schema_errors_and_closed_events(monkeypatch):
    _gate_fakes(monkeypatch)
    with pytest.raises(s6.CardRejected, match="schema"):
        s6.write_card({}, 7, {"event_title": "x"}, "m", INPUTS, strict=False)
    _gate_fakes(monkeypatch, status="closed")
    with pytest.raises(s6.CardRejected, match="closed"):
        s6.write_card({}, 7, _card(), "m", INPUTS)


def test_write_card_strips_unrelated_citation(monkeypatch):
    _gate_fakes(monkeypatch)
    inputs = {"EVENT_FACTS": [{"fact_id": 1, "metric": "brent_crude_price", "entities": ["brent"]}],
              "KB_FACTS": []}
    out = s6.write_card({}, 7, _card(), "m", inputs)
    assert out["citations_stripped"] == 1


# ---------------------------------------------------------------- D15: transient handoff
def test_s5_stores_transient_context_on_the_event(monkeypatch):
    calls = []
    monkeypatch.setattr(s5_fact_ingest.db, "fetch_all", lambda *a, **k: [])
    monkeypatch.setattr(s5_fact_ingest.db, "run_transaction",
                        lambda cfg, fn: fn(type("C", (), {"execute": lambda self, q, p=None: calls.append((q, p))})()))
    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_batch", lambda cur, sql, rows, page_size=100: calls.append((sql, rows)))
    s5_fact_ingest.run({}, extraction={"events": [{
        "event_id": 5, "article_ids": [50],
        "facts": [_f(50, "repo_rate", "5.5")],
        "transient_context": [{"metric": "nifty50_close", "value": "23000", "article_id": 50}]}]})
    sqls = " ".join(q for q, _ in calls)
    assert "INSERT INTO extracted_facts" in sqls
    assert "transient_context = COALESCE(transient_context, '[]'::jsonb) ||" in sqls
    assert "facts_extracted_at = now()" in sqls

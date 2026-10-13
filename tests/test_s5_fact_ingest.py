"""S5 tests — deterministic consolidation. Pure logic, no DB, no API.

Includes the two regression cases named in CLAUDE.md §6 Phase 2 DoD:
  - the same-period cross-source upgrade rule
  - the Brent non-merge case from the dry-run
"""
import pytest

from src.s5_fact_ingest import (
    consolidate,
    dedup_key,
    parse_numeric,
    split_transient,
    values_agree,
)


def _f(article_id, metric, value, period, tier="event_context", **kw):
    d = {"article_id": article_id, "metric": metric, "value": value, "as_of_period": period,
         "unit": "%", "claim_context": "ctx", "category": "macro_economics",
         "entities": ["india"], "fact_type": "point_in_time",
         "persistence_tier": tier, "verification_status": "single_source",
         "confidence": "high"}
    d.update(kw)
    return d


# ---------------------------------------------------------------- numeric parsing
def test_parse_numeric_handles_real_world_formats():
    assert parse_numeric("86.85") == 86.85
    assert parse_numeric("~72") == 72.0
    assert parse_numeric("-3,062.27") == -3062.27
    assert parse_numeric("9.87%") == 9.87
    assert parse_numeric("not a number") is None


def test_values_agree_exact_and_within_tolerance():
    assert values_agree("9.87", "9.87")
    assert values_agree("100", "101")           # 1% apart, inside ±2%
    assert not values_agree("100", "105")       # 5% apart
    assert values_agree("Brent rose", "brent rose")   # case-insensitive string match


# ---------------------------------------------------------------- dedup identity
def test_dedup_key_treats_period_as_part_of_identity():
    a = _f(1, "brent_crude_price", "86.85", "15-Jul-2026")
    b = _f(3, "brent_crude_price", "85.6", "past 4 sessions to 16-Jul-2026")
    assert dedup_key(a) != dedup_key(b)


# ---------------------------------------------------------------- REGRESSION CASES
def test_regression_brent_prints_from_different_periods_are_not_merged():
    """The dry-run failure class. $86.85 and $85.6 are within 2% — but the PERIODS differ,
    so they must remain two separate single_source facts. Period is checked BEFORE value."""
    facts = [
        _f(1, "brent_crude_price", "86.85", "15-Jul-2026"),
        _f(3, "brent_crude_price", "85.6", "past 4 sessions to 16-Jul-2026"),
    ]
    assert values_agree("86.85", "85.6")        # they WOULD pass the value test
    out = consolidate(facts)
    assert len(out) == 2                         # ...but they are still not merged
    assert all(f["verification_status"] == "single_source" for f in out)


def test_regression_same_period_same_value_two_sources_upgrades():
    """WPI 9.87% for June 2026 stated by articles 1 and 2 -> cross_source_consistent."""
    facts = [
        _f(1, "wpi_inflation_yoy", "9.87", "June 2026 YoY"),
        _f(2, "wpi_inflation_yoy", "9.87", "June 2026 YoY"),
    ]
    out = consolidate(facts)
    assert len(out) == 1
    assert out[0]["verification_status"] == "cross_source_consistent"
    assert out[0]["supporting_article_ids"] == [1, 2]


def test_same_period_conflicting_values_flags_discrepancy_and_keeps_all():
    facts = [
        _f(1, "cpi_inflation_yoy", "5.20", "June 2026 YoY"),
        _f(2, "cpi_inflation_yoy", "6.10", "June 2026 YoY"),
    ]
    out = consolidate(facts)
    assert len(out) == 1
    assert out[0]["verification_status"] == "discrepancy_flagged"
    assert out[0]["supporting_article_ids"] == [1, 2]
    values = {v["value"] for v in out[0]["conflicting_values"]}
    assert values == {"5.20", "6.10"}            # never silently picks one


def test_single_article_stays_single_source():
    out = consolidate([_f(1, "repo_rate", "6.5", "Aug 2026")])
    assert out[0]["verification_status"] == "single_source"
    assert out[0]["supporting_article_ids"] == [1]


def test_same_article_twice_does_not_count_as_cross_source():
    """Two mentions in ONE article is not corroboration."""
    facts = [
        _f(1, "repo_rate", "6.5", "Aug 2026"),
        _f(1, "repo_rate", "6.5", "Aug 2026"),
    ]
    out = consolidate(facts)
    assert out[0]["verification_status"] == "single_source"
    assert out[0]["supporting_article_ids"] == [1]


# ---------------------------------------------------------------- tiers
def test_kb_durable_facts_are_flagged_as_kb_candidates():
    out = consolidate([_f(3, "strait_of_hormuz_global_oil_supply_share", "~20", None,
                          tier="kb_durable", fact_type="structural")])
    assert out[0]["kb_candidate"] is True


def test_event_context_facts_are_not_kb_candidates():
    out = consolidate([_f(1, "inr_usd", "96", "15-Jul-2026")])
    assert out[0]["kb_candidate"] is False


def test_transient_metrics_are_split_out_and_never_stored():
    """Decision D1: index closes and daily % moves are used, not persisted."""
    facts = [
        _f(1, "sensex_close", "77054.94", "15-Jul-2026 close"),
        _f(1, "nifty50_change_pct", "-0.66", "15-Jul-2026 close"),
        _f(1, "inr_usd", "96", "15-Jul-2026"),
    ]
    keep, transient = split_transient(facts, {"sensex_close", "nifty50_change_pct"})
    assert [f["metric"] for f in keep] == ["inr_usd"]
    assert len(transient) == 2


def test_full_event_consolidation_matches_the_phase_0_sample_shape():
    """End-to-end shape check against the dry-run event."""
    facts = [
        _f(1, "wpi_inflation_yoy", "9.87", "June 2026 YoY"),
        _f(2, "wpi_inflation_yoy", "9.87", "June 2026 YoY"),
        _f(1, "brent_crude_price", "86.85", "15-Jul-2026"),
        _f(3, "brent_crude_price", "85.6", "past 4 sessions to 16-Jul-2026"),
        _f(3, "strait_of_hormuz_global_oil_supply_share", "~20", None, tier="kb_durable"),
    ]
    out = consolidate(facts)
    assert len(out) == 4                                     # WPI merged, Brents not
    statuses = sorted(f["verification_status"] for f in out)
    assert statuses == ["cross_source_consistent", "single_source",
                        "single_source", "single_source"]
    assert sum(1 for f in out if f["kb_candidate"]) == 1

"""S3 clustering tests — bucketing, the no-embedding fallback, and the bundle contract.
No DB, no API."""
from datetime import datetime, timedelta, timezone

import pytest

from src import s3_cluster
from src.s3_cluster import _form_event_groups, build_bundle

T0 = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _art(i, key="crude-price-move-jul2026", hours=0, **kw):
    d = {"id": i, "published_at": T0 + timedelta(hours=hours), "event_key": key,
         "title": f"title {i}", "source": "src", "url": f"https://example.com/{i}",
         "rss_summary": "s", "triage_score": 7}
    d.update(kw)
    return d


# ---------------------------------------------------------------- event formation
def test_rollup_bucket_becomes_one_event_without_an_llm_call(monkeypatch):
    """Earnings / market wraps are rolled up deterministically — never sent to the LLM."""
    monkeypatch.setattr(s3_cluster, "llm_cluster",
                        lambda *a, **k: pytest.fail("rollup buckets must not call the LLM"))
    arts = [_art(i, key="india-earnings-season-q1fy27", triage_score=i) for i in (1, 2, 3)]
    groups = _form_event_groups({}, arts, [], [], {"india-earnings-season"})
    assert len(groups) == 1 and len(groups[0]["members"]) == 3
    assert groups[0]["title"] == "title 3"                      # highest-triage member


def test_llm_failure_falls_back_to_singletons_never_one_bucket_event(monkeypatch):
    """D15: with no embeddings, a failed per-bucket LLM call splits the bucket into one
    event per article (precision-first) — it must never merge the whole bucket (D10)."""
    def boom(*a, **k):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(s3_cluster, "llm_cluster", boom)
    arts = [_art(1), _art(2), _art(3)]
    groups = _form_event_groups({}, arts, [], [], set())
    assert [len(g["members"]) for g in groups] == [1, 1, 1]
    assert all(g["assign_to_event_id"] is None for g in groups)


def test_llm_grouping_is_used_when_it_succeeds(monkeypatch):
    arts = [_art(1), _art(2), _art(3)]
    monkeypatch.setattr(s3_cluster, "llm_cluster", lambda cfg, bucket, open_ev, roll: [
        {"assign_to_event_id": None, "event_type": "crude-price-move", "title": "Oil",
         "members": bucket[:2]},
        {"assign_to_event_id": None, "event_type": "crude-price-move", "title": "Other",
         "members": bucket[2:]}])
    groups = _form_event_groups({}, arts, [], [], set())
    assert [len(g["members"]) for g in groups] == [2, 1]


def test_single_article_bucket_needs_no_llm(monkeypatch):
    monkeypatch.setattr(s3_cluster, "llm_cluster",
                        lambda *a, **k: pytest.fail("a one-article bucket is already an event"))
    groups = _form_event_groups({}, [_art(1, key="rbi-policy-jul2026")], [], [], set())
    assert len(groups) == 1 and groups[0]["members"][0]["id"] == 1


def test_reporting_quarter_uses_completed_fiscal_quarter():
    """F20 regression: earnings announced in a month key to the quarter that just ENDED,
    in Indian FY convention (Apr-Mar). July 2026 results are Q1FY27, not calendar Q3."""
    from src.s2_filter import period_suffix, reporting_quarter

    d = datetime(2026, 7, 21, tzinfo=timezone.utc)   # the run that produced the bug
    assert reporting_quarter(d) == (1, 2027)
    assert period_suffix("quarterly", d) == "q1fy27"

    assert reporting_quarter(datetime(2026, 11, 5, tzinfo=timezone.utc)) == (2, 2027)   # q2fy27
    assert reporting_quarter(datetime(2027, 2, 10, tzinfo=timezone.utc)) == (3, 2027)   # q3fy27
    assert reporting_quarter(datetime(2027, 5, 9, tzinfo=timezone.utc)) == (4, 2027)    # q4fy27
    # whole Q1 announcement window (mid-Jul to mid-Aug) collapses to ONE key
    assert (period_suffix("quarterly", datetime(2026, 7, 20, tzinfo=timezone.utc))
            == period_suffix("quarterly", datetime(2026, 8, 12, tzinfo=timezone.utc)))


# ---------------------------------------------------------------- bundle contract
def test_bundle_contract_includes_url():
    ev = {"id": 7, "event_key": "k", "title": "t"}
    bundle = build_bundle(ev, [_art(1, raw_text="body text")])
    art = bundle["articles"][0]
    assert set(art) == {"article_id", "source", "url", "published_at", "title", "body_or_summary"}
    assert art["url"] == "https://example.com/1"
    assert art["body_or_summary"] == "body text"


def test_bundle_contract_violation_raises_when_url_missing():
    ev = {"id": 7, "event_key": "k", "title": "t"}
    bad = _art(1)
    bad["url"] = None
    with pytest.raises(ValueError, match="Bundle contract violation"):
        build_bundle(ev, [bad])


def test_bundle_falls_back_to_rss_summary_when_fulltext_failed():
    """S1 graceful degradation leaves raw_text NULL; the bundle must still carry content."""
    ev = {"id": 7, "event_key": "k", "title": "t"}
    art = _art(1, raw_text=None, rss_summary="fallback summary")
    assert build_bundle(ev, [art])["articles"][0]["body_or_summary"] == "fallback summary"

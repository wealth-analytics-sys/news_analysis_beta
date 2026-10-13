"""S2 Pass A tests — deterministic filtering. No DB, no API."""
import pytest

from src.s2_filter import (
    PRIMARY_CATEGORIES,
    PRIMARY_CATEGORY_LABELS,
    _pass_a,
    is_near_duplicate,
    matches_section_blacklist,
    normalize_title,
    primary_category_for_event_key,
    title_similarity,
)


# ---------------------------------------------------------------- card categories (round 7)
def test_the_six_card_categories_plus_other():
    assert PRIMARY_CATEGORIES == [
        "global_economy", "global_equity", "indian_economy",
        "indian_equity_corporate", "commodity", "regulatory", "other",
    ]
    assert PRIMARY_CATEGORY_LABELS["global_economy"] == "Global Economy"
    assert PRIMARY_CATEGORY_LABELS["global_equity"] == "Global Equity"
    assert PRIMARY_CATEGORY_LABELS["indian_equity_corporate"] == "Indian Equity and Corporates"


@pytest.mark.parametrize("event_key,expected", [
    ("crude-price-move-jul2026", "commodity"),
    ("gold-price-move-jul2026", "commodity"),
    ("india-earnings-season-q1fy27", "indian_equity_corporate"),   # multi-hyphen type
    ("market-daily-wrap-21jul2026", "indian_equity_corporate"),
    ("trade-and-external-jul2026", "indian_economy"),              # multi-hyphen type
    ("rbi-policy-jul2026", "indian_economy"),
    ("global-monetary-policy-jul2026", "global_economy"),
    ("geopolitical-risk-jul2026", "global_economy"),
    ("global-markets-move-jul2026", "global_equity"),
    ("regulatory-change-jul2026", "regulatory"),
    ("something-unknown-jul2026", "other"),
])
def test_event_key_maps_to_correct_card_category(event_key, expected):
    assert primary_category_for_event_key(event_key) == expected


def test_normalize_title_strips_punctuation_and_case():
    assert normalize_title("Sensex DROPS 561 pts, Nifty below 24,100!") == \
        "sensex drops 561 pts nifty below 24 100"


def test_exact_duplicate_after_normalization_is_caught():
    a = "Sensex drops 561 pts as oil surges"
    b = "SENSEX DROPS 561 PTS AS OIL SURGES!!"
    assert is_near_duplicate(b, [a])


def test_near_duplicate_above_threshold_is_caught():
    a = "Sensex drops 561 points as crude oil prices surge"
    b = "Sensex drops 561 points as crude oil price surges"
    assert title_similarity(a, b) >= 0.90
    assert is_near_duplicate(b, [a], 0.90)


def test_distinct_titles_are_not_duplicates():
    a = "Sensex drops 561 points as crude oil prices surge"
    b = "RBI holds repo rate at 6.5% in August policy review"
    assert not is_near_duplicate(b, [a], 0.90)


def test_section_blacklist_matches_url():
    pats = ["/sports/", "/entertainment/"]
    assert matches_section_blacklist("https://x.com/sports/ipl-final", "IPL final", pats) == "/sports/"


def test_section_blacklist_also_matches_title_text():
    pats = ["/entertainment/"]
    assert matches_section_blacklist("https://x.com/news/a", "Entertainment roundup", pats) == "/entertainment/"


def test_section_blacklist_leaves_market_news_alone():
    pats = ["/sports/", "/entertainment/"]
    assert matches_section_blacklist("https://x.com/markets/a", "Nifty falls", pats) is None


def test_pass_a_drops_blacklisted_duplicate_and_keeps_the_rest():
    fcfg = {"source_blacklist": ["Tabloid Daily"],
            "section_blacklist": ["/sports/"],
            "duplicate_title_ratio": 0.90}
    articles = [
        {"id": 1, "source": "Mint", "url": "https://m.com/markets/a", "title": "Nifty falls as crude surges"},
        {"id": 2, "source": "Mint", "url": "https://m.com/markets/b", "title": "Nifty falls as crude surge"},
        {"id": 3, "source": "Mint", "url": "https://m.com/sports/c", "title": "IPL final tonight"},
        {"id": 4, "source": "Tabloid Daily", "url": "https://t.com/x", "title": "Some market take"},
        {"id": 5, "source": "BS", "url": "https://bs.com/economy/d", "title": "WPI inflation rises to 9.87%"},
    ]
    survivors, dropped = _pass_a(articles, fcfg)
    kept_ids = {a["id"] for a in survivors}
    reasons = dict(dropped)

    assert kept_ids == {1, 5}
    assert "near_duplicate" in reasons[2]
    assert "section_blacklist" in reasons[3]
    assert "source_blacklist" in reasons[4]


def test_pass_a_is_order_stable_first_occurrence_wins():
    """Dedup keeps the first article seen and drops later near-identical ones."""
    fcfg = {"source_blacklist": [], "section_blacklist": [], "duplicate_title_ratio": 0.90}
    articles = [
        {"id": 10, "source": "Mint", "url": "u1", "title": "RBI holds repo rate at 6.5 percent"},
        {"id": 11, "source": "ET", "url": "u2", "title": "RBI holds repo rate at 6.5 per cent"},
    ]
    survivors, dropped = _pass_a(articles, fcfg)
    assert [a["id"] for a in survivors] == [10]
    assert dropped[0][0] == 11

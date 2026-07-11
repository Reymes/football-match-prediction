"""Offline tests for the crest resolver — no network.

Network access is injected as ``fetch_json``, so the accuracy-critical logic
(country verification, alias resolution, fuzzy matching, league-id pinning) is
exercised deterministically with canned responses.
"""
from __future__ import annotations

from match_predict.data.schema import LEAGUE_BY_DIV
from match_predict.viz import slugify
from match_predict.viz.logo_fetch import (
    LEAGUE_TSDB,
    accepts_country,
    is_senior_mens,
    league_dest,
    resolve_league_badge,
    resolve_team,
    similarity,
    team_dest,
)
from match_predict.viz.team_aliases import TEAM_ALIASES


def _hit(name, country="England", badge="https://x/badge.png", gender="Male"):
    return {
        "strTeam": name,
        "strCountry": country,
        "strBadge": badge,
        "strGender": gender,
        "strSport": "Soccer",
    }


def _fetcher(mapping):
    """Return a fetch_json that yields canned results keyed by substring."""

    def fetch(url):
        for needle, payload in mapping.items():
            if needle in url:
                return payload
        return {"teams": None}

    return fetch


# --- coverage / config integrity ------------------------------------------
def test_every_division_league_has_pinned_id():
    for label in LEAGUE_BY_DIV.values():
        assert label in LEAGUE_TSDB, f"{label} missing a TheSportsDB id"
        assert LEAGUE_TSDB[label].tsdb_id > 0
        assert LEAGUE_TSDB[label].country


def test_dest_paths_match_flask_lookup():
    assert team_dest("Paris SG") == "static/logos/team/paris-sg.png"
    assert league_dest("England-PL") == "static/logos/league/england-pl.png"
    assert slugify("Nott'm Forest") in team_dest("Nott'm Forest")


# --- similarity & country verification -------------------------------------
def test_similarity_ignores_filler_and_accents():
    assert similarity("Bournemouth", "AFC Bournemouth") > 0.9
    assert similarity("Lazio", "SS Lazio") > 0.9
    assert similarity("Munchen", "München") > 0.9
    assert similarity("Arsenal", "Chelsea") < 0.5


def test_country_alias_accepts_variants_and_rejects_foreign():
    assert accepts_country("The Netherlands", "Netherlands")
    assert accepts_country("England", "England")
    assert not accepts_country("Argentina", "England")
    assert not accepts_country(None, "England")


# --- team resolution --------------------------------------------------------
def test_resolves_exact_name_with_country_check():
    fetch = _fetcher({"searchteams": {"teams": [_hit("Arsenal")]}})
    m = resolve_team("Arsenal", "England", fetch)
    assert m is not None and m.matched_name == "Arsenal"
    assert m.badge_url.endswith("badge.png")


def test_rejects_same_name_wrong_country():
    fetch = _fetcher({"searchteams": {"teams": [_hit("Arsenal", "Argentina")]}})
    assert resolve_team("Arsenal", "England", fetch) is None


def test_alias_is_tried_first():
    # Raw "Ath Madrid" would not match; the alias -> "Atletico Madrid" does.
    assert TEAM_ALIASES["Ath Madrid"] == "Atletico Madrid"
    fetch = _fetcher({
        "Atletico": {"teams": [_hit("Atletico Madrid", "Spain")]},
    })
    m = resolve_team("Ath Madrid", "Spain", fetch)
    assert m is not None and m.via_alias
    assert m.matched_name == "Atletico Madrid"


def test_alias_query_scores_against_official_name():
    # "Wolves" is nothing like "Wolverhampton Wanderers"; the alias query is.
    fetch = _fetcher({
        "Wolverhampton": {"teams": [_hit("Wolverhampton Wanderers")]},
    })
    m = resolve_team("Wolves", "England", fetch)
    assert m is not None and m.matched_name == "Wolverhampton Wanderers"


def test_womens_team_is_rejected():
    fetch = _fetcher({
        "searchteams": {"teams": [_hit("West Ham Women", gender="Female")]},
    })
    assert resolve_team("West Ham", "England", fetch) is None


def test_youth_squad_is_rejected():
    assert not is_senior_mens(_hit("Chelsea U21"), "Chelsea")
    assert not is_senior_mens(_hit("Ajax Reserves"), "Ajax")
    assert is_senior_mens(_hit("Arsenal"), "Arsenal")
    # A source name that legitimately says 'Women' must still be allowed.
    assert is_senior_mens(_hit("Arsenal Women"), "Arsenal Women")


def test_non_soccer_result_is_rejected():
    hit = _hit("Arsenal")
    hit["strSport"] = "Basketball"
    assert not is_senior_mens(hit, "Arsenal")


def test_low_similarity_result_is_not_accepted():
    fetch = _fetcher({"searchteams": {"teams": [_hit("Barcelona", "Spain")]}})
    assert resolve_team("Real Madrid", "Spain", fetch) is None


def test_missing_badge_is_skipped():
    fetch = _fetcher({"searchteams": {"teams": [_hit("Arsenal", badge=None)]}})
    assert resolve_team("Arsenal", "England", fetch) is None


def test_network_error_is_a_miss_not_a_crash():
    def boom(url):
        raise TimeoutError("slow")

    assert resolve_team("Arsenal", "England", boom) is None


# --- league resolution ------------------------------------------------------
def test_resolve_league_badge_uses_pinned_id():
    seen = {}

    def fetch(url):
        seen["url"] = url
        return {"leagues": [{"strBadge": "https://x/epl.png"}]}

    url = resolve_league_badge("England-PL", fetch)
    assert url == "https://x/epl.png"
    assert str(LEAGUE_TSDB["England-PL"].tsdb_id) in seen["url"]


def test_resolve_unknown_league_returns_none():
    assert resolve_league_badge("Narnia-PL", lambda u: {}) is None

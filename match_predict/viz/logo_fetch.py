"""Resolve real club/league crests from TheSportsDB, accurately and offline-safe.

The Flask UI always has a badge to show: a generated SVG (``logos.py``) is the
baseline, and a real crest in ``static/logos/`` is preferred when present. This
module is the *enrichment* layer that finds those real crests.

Accuracy is the whole point, so this module does three things the naive
"search the raw name" approach does not:

1. **Pinned league IDs.** Each Football-Data division maps to a *verified*
   TheSportsDB league id (``LEAGUE_TSDB``). League crests are fetched by id, so
   they are never the wrong competition, and the id doubles as the source of a
   team's expected country.
2. **Country verification.** A team search result is only accepted if its
   country matches the division's country. This blocks same-named clubs abroad
   (e.g. Arsenal Sarandi in Argentina, Everton de Vina in Chile).
3. **Alias + fuzzy matching.** Terse Football-Data spellings are expanded via
   ``TEAM_ALIASES`` and, failing that, matched against the search results by a
   diacritic- and filler-insensitive similarity score above a threshold.

Anything that cannot be resolved with confidence is reported as a miss (it
falls back to the generated SVG) rather than guessed — see ``scripts/fetch_logos.py``.

Network access is injected as a ``fetch_json`` / ``fetch_bytes`` callable so the
resolution logic is fully unit-testable without hitting the network.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from .logos import slugify

# Free test key "3"; endpoints used are on the free tier.
_BASE = "https://www.thesportsdb.com/api/v1/json/3"
SEARCH_TEAMS = _BASE + "/searchteams.php?t={q}"
LOOKUP_LEAGUE = _BASE + "/lookupleague.php?id={id}"


@dataclass(frozen=True)
class LeagueRef:
    """A verified TheSportsDB league: its id and the country of its clubs."""

    tsdb_id: int
    country: str


# Division -> verified TheSportsDB league id + club country. Each id was
# confirmed via lookupleague.php (see the project logo docs); do not "fix" one
# without re-verifying, as neighbouring ids are unrelated competitions.
LEAGUE_TSDB: dict[str, LeagueRef] = {
    "England-PL": LeagueRef(4328, "England"),
    "England-Champ": LeagueRef(4329, "England"),
    "England-L1": LeagueRef(4396, "England"),
    "England-L2": LeagueRef(4397, "England"),
    "England-NL": LeagueRef(4590, "England"),
    "Scotland-PR": LeagueRef(4330, "Scotland"),
    "Scotland-Champ": LeagueRef(4395, "Scotland"),
    "Scotland-L1": LeagueRef(4669, "Scotland"),
    "Scotland-L2": LeagueRef(4670, "Scotland"),
    "Germany-BL": LeagueRef(4331, "Germany"),
    "Germany-BL2": LeagueRef(4399, "Germany"),
    "Italy-SA": LeagueRef(4332, "Italy"),
    "Italy-SB": LeagueRef(4394, "Italy"),
    "Spain-LL": LeagueRef(4335, "Spain"),
    "Spain-LL2": LeagueRef(4400, "Spain"),
    "France-L1": LeagueRef(4334, "France"),
    "France-L2": LeagueRef(4401, "France"),
    "Netherlands-ED": LeagueRef(4337, "Netherlands"),
    "Belgium-PL": LeagueRef(4338, "Belgium"),
    "Portugal-PL": LeagueRef(4344, "Portugal"),
    "Turkey-SL": LeagueRef(4339, "Turkey"),
    "Greece-SL": LeagueRef(4336, "Greece"),
}

# TheSportsDB spells a few countries differently; accept the variants.
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "Netherlands": {"netherlands", "the netherlands", "holland"},
    "England": {"england", "united kingdom", "great britain"},
    "Scotland": {"scotland", "united kingdom", "great britain"},
}

_WORD_RE = re.compile(r"[a-z0-9]+")
# Filler tokens dropped before fuzzy comparison so "SS Lazio" ~ "Lazio" and
# "AFC Bournemouth" ~ "Bournemouth" score as near-identical.
_FILLER = {
    "fc", "cf", "sc", "ac", "afc", "cd", "ud", "sv", "us", "as", "rc", "ss",
    "ssc", "ogc", "vfl", "vfb", "tsg", "krc", "kaa", "sl", "sd", "cp", "if",
    "the", "de", "of", "club", "calcio", "1", "04", "05", "96",
}

# The free key returns one hit per search, so the wrong squad (women's/youth/
# reserve) must be rejected outright — accepting it would show a wrong crest.
_NON_SENIOR = {
    "women", "womens", "ladies", "wfc", "frauen", "femenino", "feminin",
    "feminine", "feminines", "femminile", "damen", "u18", "u19", "u20",
    "u21", "u23", "youth", "academy", "reserve", "reserves", "futures",
}

DEFAULT_THRESHOLD = 0.72


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _tokens(name: str) -> list[str]:
    return _WORD_RE.findall(_strip_accents(str(name)).lower())


def _norm(name: str) -> str:
    """Diacritic- and filler-insensitive key used for fuzzy comparison."""
    toks = [t for t in _tokens(name) if t not in _FILLER]
    return " ".join(toks or _tokens(name))


def similarity(a: str, b: str) -> float:
    """Blend of sequence ratio and token-set overlap on normalised names."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    sa, sb = set(na.split()), set(nb.split())
    overlap = len(sa & sb) / len(sa | sb)
    return max(seq, overlap)


def accepts_country(result_country: str | None, expected: str) -> bool:
    """True if a search hit's country is compatible with the expected one."""
    if not result_country:
        return False
    rc = result_country.strip().lower()
    allowed = _COUNTRY_ALIASES.get(expected, {expected.lower()})
    return rc in allowed


def is_senior_mens(team: dict, name: str) -> bool:
    """Reject women's / youth / reserve squads unless the source name is one.

    Football-Data only lists senior men's sides, so any non-senior marker on a
    result that is absent from the queried name means it is the wrong squad.
    """
    if (team.get("strSport") or "Soccer") != "Soccer":
        return False
    if (team.get("strGender") or "").strip().lower() == "female":
        return False
    src = set(_tokens(name))
    result = set(_tokens(team.get("strTeam") or ""))
    return not (_NON_SENIOR & result - src)


def _candidate_queries(name: str) -> list[str]:
    """Search terms to try in order: alias first, then the raw name."""
    from .team_aliases import TEAM_ALIASES

    seen: list[str] = []
    for q in (TEAM_ALIASES.get(name), name):
        if q and q not in seen:
            seen.append(q)
    return seen


@dataclass(frozen=True)
class TeamMatch:
    """A resolved crest: where it came from and how confident we are."""

    query: str
    matched_name: str
    country: str
    badge_url: str
    score: float
    via_alias: bool


def _best_hit(
    name: str, query: str, teams: list[dict], country: str, threshold: float
) -> tuple[dict, float] | None:
    """Highest-scoring valid hit, scored against name *or* query (max).

    Scoring against the query too means an alias like ``Wolves`` ->
    ``Wolverhampton Wanderers`` matches at 1.0 even though the raw name is
    nothing like the official one.
    """
    best: tuple[dict, float] | None = None
    for t in teams:
        if not accepts_country(t.get("strCountry"), country):
            continue
        if not is_senior_mens(t, name):
            continue
        badge = t.get("strBadge") or t.get("strTeamBadge")
        if not badge:
            continue
        result_name = t.get("strTeam") or ""
        score = max(similarity(name, result_name), similarity(query, result_name))
        if score >= threshold and (best is None or score > best[1]):
            best = (t, score)
    return best


def resolve_team(
    name: str,
    expected_country: str,
    fetch_json: Callable[[str], dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> TeamMatch | None:
    """Find a country-verified, senior-men's crest for ``name`` or None.

    ``fetch_json`` maps a URL to the decoded JSON dict; injecting it keeps this
    pure and testable. Candidate queries (alias, then raw name) are tried until
    one yields a hit that passes country, squad, and similarity checks.
    """
    from urllib.parse import quote

    from .team_aliases import TEAM_ALIASES

    aliased = name in TEAM_ALIASES
    for i, query in enumerate(_candidate_queries(name)):
        try:
            data = fetch_json(SEARCH_TEAMS.format(q=quote(query)))
        except Exception:  # noqa: BLE001 - network flakiness is a miss, not a crash
            continue
        teams = (data or {}).get("teams") or []
        hit = _best_hit(name, query, teams, expected_country, threshold)
        if hit:
            team, score = hit
            return TeamMatch(
                query=query,
                matched_name=team.get("strTeam") or query,
                country=team.get("strCountry") or expected_country,
                badge_url=team.get("strBadge") or team.get("strTeamBadge"),
                score=round(score, 3),
                via_alias=aliased and i == 0,
            )
    return None


def resolve_league_badge(
    league: str, fetch_json: Callable[[str], dict]
) -> str | None:
    """Return the crest URL for a league label via its pinned id, or None."""
    ref = LEAGUE_TSDB.get(league)
    if ref is None:
        return None
    try:
        data = fetch_json(LOOKUP_LEAGUE.format(id=ref.tsdb_id))
    except Exception:  # noqa: BLE001
        return None
    lg = ((data or {}).get("leagues") or [{}])[0]
    return lg.get("strBadge") or lg.get("strLogo")


def team_dest(name: str, root: str = "static/logos") -> str:
    """Filesystem path the Flask layer will look for this team's crest at."""
    return f"{root}/team/{slugify(name)}.png"


def league_dest(league: str, root: str = "static/logos") -> str:
    """Filesystem path the Flask layer will look for this league's crest at."""
    return f"{root}/league/{slugify(league)}.png"

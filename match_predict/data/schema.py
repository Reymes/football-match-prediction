"""Canonical match schema and the mapping from raw Football-Data columns.

Football-Data.co.uk publishes one CSV per league-season. The column set has
grown over time:

    1993+  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR                (results)
    2000+  HS/AS/HST/AST/HC/AC/HY/AY/HR/AR, Referee                 (match stats)
    2002+  B365H/D/A, ...                                           (1X2 odds)
    2012+  PSCH/PSCD/PSCA                                           (Pinnacle closing)
    2019+  Time, AvgH/D/A, AHh, AH odds                             (avg + Asian handicap)

We normalize every file into a single stable schema so downstream code never
has to know which era a row came from. Missing columns become NaN.

IMPORTANT: The odds columns are model INPUTS/BENCHMARKS, never targets that
leak. Pre-match odds (B365H/D/A, AvgH/D/A) are known before kickoff and are
legitimate features. CLOSING odds (PSCH/PSCA, ...C...) are also pre-kickoff but
are treated carefully in the leakage layer (see features/build.py) because in a
live setting the *closing* line is only available seconds before kick and using
it inflates offline metrics.
"""
from __future__ import annotations

# --- The unified schema every downstream module relies on -------------------
CANONICAL_COLUMNS = [
    # identity / context
    "match_id", "league", "season", "date", "time",
    "home_team", "away_team",
    # full-time & half-time outcome (targets)
    "fthg", "ftag", "ftr", "hthg", "htag", "htr",
    # match statistics (post-match; used ONLY to build lagged features)
    "referee",
    "hs", "as_", "hst", "ast",          # shots / shots on target
    "hf", "af",                         # fouls
    "hc", "ac",                         # corners
    "hy", "ay", "hr", "ar",             # cards
    # pre-match bookmaker 1X2 odds (legitimate pre-kickoff features)
    "odds_h", "odds_d", "odds_a",       # best available (B365, else Avg)
    "odds_over25", "odds_under25",      # Over/Under 2.5 goals
    "ah_line", "odds_ah_home", "odds_ah_away",  # Asian handicap
    # closing odds (pre-kickoff but leakage-sensitive; see build.py)
    "close_h", "close_d", "close_a",
]

# Raw -> canonical. Several raw candidates map to one canonical field; the
# first present (in order) wins. This lets us gracefully prefer Bet365, then
# market average, then Pinnacle, etc.
RAW_TO_CANONICAL = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
    "HTHG": "hthg", "HTAG": "htag", "HTR": "htr",
    "Referee": "referee",
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HF": "hf", "AF": "af",
    "HC": "hc", "AC": "ac",
    "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "AHh": "ah_line",
}

# Ordered preference lists for fields that have multiple raw sources.
ODDS_PREFERENCE = {
    "odds_h": ["B365H", "AvgH", "BWH", "WHH", "PSH"],
    "odds_d": ["B365D", "AvgD", "BWD", "WHD", "PSD"],
    "odds_a": ["B365A", "AvgA", "BWA", "WHA", "PSA"],
    "odds_over25": ["B365>2.5", "Avg>2.5", "P>2.5"],
    "odds_under25": ["B365<2.5", "Avg<2.5", "P<2.5"],
    "odds_ah_home": ["B365AHH", "AvgAHH", "PAHH"],
    "odds_ah_away": ["B365AHA", "AvgAHA", "PAHA"],
    "close_h": ["PSCH", "B365CH", "AvgCH"],
    "close_d": ["PSCD", "B365CD", "AvgCD"],
    "close_a": ["PSCA", "B365CA", "AvgCA"],
}

# Map a raw Football-Data Div code to a stable, unique human league label.
# Covers every division published in the "main leagues" archive. Any code not
# listed here falls back to the raw code itself (see ingest.load_file), so a
# brand-new division added upstream still loads with a unique — if terse —
# label instead of crashing or silently merging into another league.
LEAGUE_BY_DIV = {
    # England
    "E0": "England-PL", "E1": "England-Champ", "E2": "England-L1",
    "E3": "England-L2", "EC": "England-NL",
    # Scotland
    "SC0": "Scotland-PR", "SC1": "Scotland-Champ",
    "SC2": "Scotland-L1", "SC3": "Scotland-L2",
    # Germany
    "D1": "Germany-BL", "D2": "Germany-BL2",
    # Italy
    "I1": "Italy-SA", "I2": "Italy-SB",
    # Spain
    "SP1": "Spain-LL", "SP2": "Spain-LL2",
    # France
    "F1": "France-L1", "F2": "France-L2",
    # Rest of Europe (single top flight in the archive)
    "N1": "Netherlands-ED", "B1": "Belgium-PL", "P1": "Portugal-PL",
    "T1": "Turkey-SL", "G1": "Greece-SL",
}

# Map a raw Div code to the on-disk "<country>/<League>" folder + filename stem
# used by the sync layer. Kept alongside LEAGUE_BY_DIV so the two never drift:
# every division we know how to *store* we also know how to *label*.
DIV_TO_FOLDER = {
    "E0": "england/PremierLeague", "E1": "england/Championship",
    "E2": "england/LeagueOne", "E3": "england/LeagueTwo",
    "EC": "england/NationalLeague",
    "SC0": "scotland/Premiership", "SC1": "scotland/Championship",
    "SC2": "scotland/LeagueOne", "SC3": "scotland/LeagueTwo",
    "D1": "germany/Bundesliga", "D2": "germany/Bundesliga2",
    "I1": "italy/SerieA", "I2": "italy/SerieB",
    "SP1": "spain/LaLiga", "SP2": "spain/LaLiga2",
    "F1": "france/Ligue1", "F2": "france/Ligue2",
    "N1": "netherlands/Eredivisie", "B1": "belgium/ProLeague",
    "P1": "portugal/PrimeiraLiga", "T1": "turkey/SuperLig",
    "G1": "greece/SuperLeague",
}

# All season codes Football-Data publishes in the main-leagues archive,
# newest first. Two-digit start+end year, e.g. "2526" == 2025/26.
SEASON_CODES = [
    "2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718",
    "1617", "1516", "1415", "1314", "1213", "1112", "1011", "0910", "0809",
    "0708", "0607", "0506", "0405", "0304", "0203", "0102", "0001", "9900",
    "9899", "9798", "9697", "9596", "9495", "9394",
]


def season_code_to_name(code: str) -> str:
    """'2526' -> '2025-2026'. Two-digit years pivot at 93 (1993..2092)."""
    a, b = int(code[:2]), int(code[2:])
    y1 = 1900 + a if a >= 93 else 2000 + a
    y2 = 1900 + b if b >= 93 else 2000 + b
    return f"{y1}-{y2}"

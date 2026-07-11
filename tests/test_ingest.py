"""Ingestion robustness for the full football-data archive + future updates.

These tests guard the two ways new data can break the loader:
  1. A new *division* (league) added to the archive.
  2. A new *season* file dropped into an existing league folder.

They use tiny synthetic CSVs written to a temp dir, so they are fast and do
not depend on the licensed real data being present.
"""
import os

import pandas as pd
import pytest

from match_predict.data import load_directory, load_file
from match_predict.data.schema import LEAGUE_BY_DIV


# All division codes the current archive actually ships (see football-data/).
ARCHIVE_DIVS = [
    "E0", "E1", "E2", "E3", "EC",
    "SC0", "SC1", "SC2", "SC3",
    "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
    "N1", "B1", "P1", "T1", "G1",
]


def _write_csv(path, div, home="Alpha", away="Beta", date="10/08/2024"):
    """Minimal but valid football-data row (modern DD/MM/YYYY layout)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{
        "Div": div, "Date": date, "Time": "15:00",
        "HomeTeam": home, "AwayTeam": away,
        "FTHG": 2, "FTAG": 1, "FTR": "H",
        "B365H": 1.9, "B365D": 3.4, "B365A": 4.2,
    }]).to_csv(path, index=False)


def test_every_archive_division_has_a_label():
    """No shipped division falls through to a raw code / silent merge."""
    missing = [d for d in ARCHIVE_DIVS if d not in LEAGUE_BY_DIV]
    assert not missing, f"divisions missing a league label: {missing}"


def test_league_labels_are_unique():
    """Two divisions must never collapse into the same league label."""
    labels = list(LEAGUE_BY_DIV.values())
    dupes = {x for x in labels if labels.count(x) > 1}
    assert not dupes, f"non-unique league labels: {dupes}"


def test_div_code_drives_league_label(tmp_path):
    """League is taken from the Div column, not the folder name."""
    # File lives under england/ but carries a Championship Div code.
    p = tmp_path / "england" / "Championship_2024-2025.csv"
    _write_csv(str(p), "E1")
    df = load_file(str(p))
    assert df["league"].iloc[0] == "England-Champ"
    assert df["season"].iloc[0] == "2024-2025"


def test_all_divisions_load_with_distinct_leagues(tmp_path):
    """A full archive-shaped tree loads into exactly one league per division."""
    for div in ARCHIVE_DIVS:
        _write_csv(str(tmp_path / div.lower() / f"{div}_2024-2025.csv"), div)
    df = load_directory(str(tmp_path))
    assert df["league"].nunique() == len(ARCHIVE_DIVS)
    assert set(df["league"]) == {LEAGUE_BY_DIV[d] for d in ARCHIVE_DIVS}


def test_new_season_file_is_auto_discovered(tmp_path):
    """Dropping a later-season CSV in place is picked up with no code change."""
    league_dir = tmp_path / "england"
    _write_csv(str(league_dir / "PremierLeague_2024-2025.csv"), "E0",
               date="10/08/2024")
    first = load_directory(str(tmp_path))
    assert set(first["season"]) == {"2024-2025"}

    # Simulate a future update: a new season file appears.
    _write_csv(str(league_dir / "PremierLeague_2025-2026.csv"), "E0",
               home="Gamma", away="Delta", date="12/08/2025")
    second = load_directory(str(tmp_path))
    assert set(second["season"]) == {"2024-2025", "2025-2026"}
    assert len(second) == 2


def test_unknown_future_division_gets_unique_fallback(tmp_path):
    """A brand-new division not yet in the map loads under its raw code."""
    _write_csv(str(tmp_path / "x" / "X9_2025-2026.csv"), "X9")
    df = load_file(str(tmp_path / "x" / "X9_2025-2026.csv"))
    # Not crash, not empty, and label is the (unique) raw div code.
    assert df["league"].iloc[0] == "X9"

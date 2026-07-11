"""Sync layer: season-code math, division routing, fixtures parsing.

Network calls are never made here — we test the pure logic and the local
parsing against a synthetic fixtures file.
"""
import os
from datetime import date

import pandas as pd

from match_predict.data import parse_fixtures, current_season_code
from match_predict.data.schema import (
    DIV_TO_FOLDER, LEAGUE_BY_DIV, SEASON_CODES, season_code_to_name,
)
from match_predict.data.sync import season_codes_for_dates, _extract_zip


def test_season_code_to_name():
    assert season_code_to_name("2526") == "2025-2026"
    assert season_code_to_name("9394") == "1993-1994"
    assert season_code_to_name("0001") == "2000-2001"


def test_current_season_code_rolls_over_in_july():
    assert current_season_code(date(2025, 7, 1)) == "2526"     # July -> new
    assert current_season_code(date(2025, 6, 30)) == "2425"    # June -> old
    assert current_season_code(date(2026, 1, 15)) == "2526"


def test_season_codes_for_dates():
    s = pd.Series(pd.to_datetime(["2026-05-31", "2025-09-01", "2024-08-10"]))
    assert season_codes_for_dates(s) == ["2425", "2526"]       # sorted, deduped


def test_every_folder_div_has_a_label():
    """DIV_TO_FOLDER and LEAGUE_BY_DIV must cover the same divisions."""
    assert set(DIV_TO_FOLDER) == set(LEAGUE_BY_DIV)


def test_folder_targets_are_unique():
    paths = list(DIV_TO_FOLDER.values())
    assert len(paths) == len(set(paths))


def test_season_codes_cover_1993_to_present():
    names = [season_code_to_name(c) for c in SEASON_CODES]
    assert "1993-1994" in names and "2025-2026" in names
    assert len(SEASON_CODES) == len(set(SEASON_CODES))


def test_extract_zip_routes_divisions(tmp_path):
    """A synthetic zip's CSVs land in the right country/league files."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("E0.csv", "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,10/08/2024,A,B,1,0\n")
        zf.writestr("SP2.csv", "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP2,10/08/2024,C,D,2,2\n")
        zf.writestr("ZZ.csv", "junk\n")           # unknown div -> ignored
    written = _extract_zip(buf.getvalue(), "2024-2025", str(tmp_path))
    assert os.path.exists(tmp_path / "england" / "PremierLeague_2024-2025.csv")
    assert os.path.exists(tmp_path / "spain" / "LaLiga2_2024-2025.csv")
    assert len(written) == 2                       # ZZ skipped


def _write_fixtures(path):
    header = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    rows = [
        "E0,15/08/2025,20:00,Liverpool,Arsenal,2.10,3.40,3.50,2.05,3.45,3.55",
        "ZZ,15/08/2025,20:00,Foo,Bar,,,,,,",           # unknown div -> dropped
        "SP1,16/08/2025,21:00,Barcelona,Sevilla,1.55,4.20,6.00,,,",
        ",,,,,,,,,,",                                  # blank -> dropped
    ]
    with open(path, "w") as fh:
        fh.write(header + "\n".join(rows) + "\n")


def test_parse_fixtures(tmp_path):
    p = tmp_path / "fixtures.csv"
    _write_fixtures(str(p))
    fx = parse_fixtures(str(p))
    assert list(fx.columns) == ["league", "date", "time", "home_team",
                                "away_team", "odds_h", "odds_d", "odds_a"]
    assert set(fx["league"]) == {"England-PL", "Spain-LL"}   # ZZ + blank dropped
    liv = fx[fx["home_team"] == "Liverpool"].iloc[0]
    assert liv["league"] == "England-PL"
    assert abs(liv["odds_h"] - 2.10) < 1e-9                  # B365 preferred
    assert str(liv["date"].date()) == "2025-08-15"


def test_parse_fixtures_missing_file(tmp_path):
    fx = parse_fixtures(str(tmp_path / "nope.csv"))
    assert fx.empty and "league" in fx.columns

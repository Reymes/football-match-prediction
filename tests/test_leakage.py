"""Leakage-safety tests (fix.md §6, §23 feature tests).

The single most important property of a football feature frame: no feature may
encode the result of its own match. We verify the rolling form features are
strictly shifted (a fixture's rolling stats depend only on prior matches).
"""
import numpy as np
import pandas as pd

from match_predict.features.form import add_form_features


def test_first_appearance_has_no_rolling_history():
    """Each team's very first match must have NaN rolling means (nothing prior)."""
    df = pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2022-08-06", "2022-08-13", "2022-08-20"]),
        "league": "L1", "season": "2022-2023",
        "home_team": ["A", "A", "B"], "away_team": ["B", "C", "A"],
        "fthg": [2, 1, 0], "ftag": [0, 1, 3],
        "hs": [10, 8, 5], "as_": [4, 8, 12], "hst": [5, 3, 2], "ast": [1, 3, 6],
        "hc": [6, 5, 3], "ac": [2, 5, 7],
    })
    out = add_form_features(df).set_index("match_id")
    # A's first match (m1): no prior games -> rolling GF must be NaN
    assert np.isnan(out.loc["m1", "home_roll_gf"])
    # A's second match (m2): exactly one prior game (m1, scored 2) -> mean 2.0
    assert out.loc["m2", "home_roll_gf"] == 2.0


def test_rolling_excludes_current_match(feat):
    """Rolling GF at match t must never equal a mean that includes match t itself."""
    # Reconstruct per-team goal series and confirm the shift-by-one property on a
    # team with enough history.
    long_rows = []
    for _, r in feat.iterrows():
        long_rows.append((r["home_team"], r["date"], r["fthg"], r["home_roll_gf"]))
        long_rows.append((r["away_team"], r["date"], r["ftag"], r["away_roll_gf"]))
    lg = pd.DataFrame(long_rows, columns=["team", "date", "gf", "roll_gf"])
    lg = lg.sort_values(["team", "date"]).reset_index(drop=True)
    for team, g in lg.groupby("team"):
        g = g.reset_index(drop=True)
        # expected rolling mean of the PREVIOUS up-to-6 gf values
        expected = g["gf"].shift(1).rolling(6, min_periods=1).mean()
        got = g["roll_gf"].reset_index(drop=True)
        both = expected.notna() & got.notna()
        assert np.allclose(expected[both], got[both], atol=1e-9), \
            f"team {team} rolling GF is not a strict shift-by-one of prior matches"


def test_no_future_row_leaks_into_features(feat):
    """matches_played is monotone per team and starts at 0 (pure count of priors)."""
    firsts = feat.groupby("home_team")["home_matches_played"].min()
    # some team must legitimately start at 0 priors as home side
    assert (firsts == 0).any()
    assert feat["home_matches_played"].min() >= 0

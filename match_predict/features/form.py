"""Rolling / exponentially-weighted form features.

The one rule that matters here: **every rolling statistic is shifted by one
match** within each team's own timeline, so a fixture only ever sees results
that happened strictly before it. This is the single most common source of
leakage in football models (using the current match in its own rolling mean).

We build a long "team-match" table (two rows per fixture — one per side),
compute the rolling features there where each team's history is contiguous,
then pivot back to home_*/away_* columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (canonical stat -> home column, away column) for stats we roll.
_STAT_SOURCES = {
    "gf": ("fthg", "ftag"),      # goals for
    "ga": ("ftag", "fthg"),      # goals against
    "sf": ("hs", "as_"),         # shots for
    "sa": ("as_", "hs"),         # shots against
    "stf": ("hst", "ast"),       # shots on target for
    "sta": ("ast", "hst"),       # shots on target against
    "cf": ("hc", "ac"),          # corners for
}

_ROLL_WINDOW = 6
_EWMA_HALFLIFE = 5


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows."""
    rows = []
    base_cols = ["match_id", "date", "league", "season"]
    for side, opp in (("home", "away"), ("away", "home")):
        sub = df[base_cols].copy()
        sub["team"] = df[f"{side}_team"].values
        sub["venue"] = side
        # points from this match (target-derived, will be SHIFTED before use)
        if side == "home":
            pts = np.select([df.fthg > df.ftag, df.fthg == df.ftag],
                            [3, 1], default=0)
        else:
            pts = np.select([df.ftag > df.fthg, df.fthg == df.ftag],
                            [3, 1], default=0)
        sub["pts"] = pts
        for stat, (hcol, acol) in _STAT_SOURCES.items():
            sub[stat] = (df[hcol] if side == "home" else df[acol]).values
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    return long.sort_values(["league", "team", "date"]).reset_index(drop=True)


def _rolling_shifted(g: pd.DataFrame) -> pd.DataFrame:
    """Per-team rolling features using only prior matches (shift(1))."""
    out = pd.DataFrame(index=g.index)
    shifted = g.shift(1)  # exclude the current match entirely
    # form: rolling mean points, goals, shots over the last N matches
    out["form_pts"] = shifted["pts"].rolling(_ROLL_WINDOW, min_periods=1).mean()
    for stat in _STAT_SOURCES:
        roll = shifted[stat].rolling(_ROLL_WINDOW, min_periods=1).mean()
        out[f"roll_{stat}"] = roll
        out[f"ewma_{stat}"] = shifted[stat].ewm(
            halflife=_EWMA_HALFLIFE, min_periods=1).mean()
    # momentum: last 3 points vs previous 3
    last3 = shifted["pts"].rolling(3, min_periods=1).mean()
    prev3 = shifted["pts"].shift(3).rolling(3, min_periods=1).mean()
    out["momentum"] = last3 - prev3
    # matches played so far (maturity of the estimate)
    out["matches_played"] = np.arange(len(g))
    return out


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach home_*/away_* rolling form features (all pre-kickoff)."""
    long = _to_long(df)
    feats = (long.groupby(["league", "team"], group_keys=False,
                          sort=False)[list(_STAT_SOURCES) + ["pts"]]
             .apply(_rolling_shifted))
    long = pd.concat([long[["match_id", "venue"]], feats], axis=1)

    feat_cols = [c for c in long.columns if c not in ("match_id", "venue")]
    home = (long[long.venue == "home"]
            .set_index("match_id")[feat_cols]
            .add_prefix("home_"))
    away = (long[long.venue == "away"]
            .set_index("match_id")[feat_cols]
            .add_prefix("away_"))
    out = df.merge(home, left_on="match_id", right_index=True, how="left")
    out = out.merge(away, left_on="match_id", right_index=True, how="left")

    # A few informative differentials the models like.
    out["form_pts_diff"] = out["home_form_pts"] - out["away_form_pts"]
    out["roll_gf_diff"] = out["home_roll_gf"] - out["away_roll_gf"]
    out["roll_ga_diff"] = out["home_roll_ga"] - out["away_roll_ga"]
    out["ewma_stf_diff"] = out["home_ewma_stf"] - out["away_ewma_stf"]
    return out

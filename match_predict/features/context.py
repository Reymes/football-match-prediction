"""Match-context features: rest, fixture congestion, season timing, derbies.

All derived from the fixture list itself (dates, teams) — available before
kickoff, so leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _team_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous match, and matches in the last 14 days."""
    rows = []
    for side in ("home", "away"):
        sub = df[["match_id", "date", "league"]].copy()
        sub["team"] = df[f"{side}_team"].values
        sub["venue"] = side
        rows.append(sub)
    long = (pd.concat(rows, ignore_index=True)
            .sort_values(["league", "team", "date"]))
    prev_date = long.groupby(["league", "team"])["date"].shift(1)
    long["rest_days"] = (long["date"] - prev_date).dt.days
    # congestion: number of matches by this team in the trailing 14 days
    def _congestion(g):
        dates = g["date"].values.astype("datetime64[D]")
        out = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            window = (dates[i] - dates[:i]).astype("timedelta64[D]").astype(int)
            out[i] = np.sum((window > 0) & (window <= 14))
        return pd.Series(out, index=g.index)
    long["congestion_14d"] = (
        long.groupby(["league", "team"], group_keys=False).apply(_congestion))
    return long


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    long = _team_rest_days(df)
    home = (long[long.venue == "home"].set_index("match_id")
            [["rest_days", "congestion_14d"]].add_prefix("home_"))
    away = (long[long.venue == "away"].set_index("match_id")
            [["rest_days", "congestion_14d"]].add_prefix("away_"))
    out = df.merge(home, left_on="match_id", right_index=True, how="left")
    out = out.merge(away, left_on="match_id", right_index=True, how="left")

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    # Cap absurd rest gaps (season breaks) so they don't dominate splits.
    for c in ("home_rest_days", "away_rest_days"):
        out[c] = out[c].clip(upper=30)
    out["rest_diff"] = out["rest_diff"].clip(-30, 30)

    # Season phase: fraction of the season elapsed (0 early .. 1 late), where
    # motivation / relegation-title pressure concentrates.
    out["season_progress"] = (
        out.groupby(["league", "season"])["date"]
        .transform(lambda s: (s - s.min()) / (s.max() - s.min() + pd.Timedelta(days=1))))

    # Derby proxy: both teams have appeared against each other historically a lot
    # is expensive; use a cheap same-first-word city heuristic is unreliable, so
    # we expose a placeholder the enrichment layer (Transfermarkt/city geodata)
    # would fill in production. Kept explicit to document the extension point.
    out["is_derby"] = 0
    return out

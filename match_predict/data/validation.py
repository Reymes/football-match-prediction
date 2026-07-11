"""Lightweight data-quality validation (a hand-rolled subset of what
Great Expectations / pandera would enforce in production).

Fails loud on structural problems; warns on quality issues that are tolerable
but worth surfacing (e.g. missing odds in pre-2002 rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"ValidationReport: {self.n_rows} rows  "
                 f"({'OK' if self.ok else 'FAILED'})"]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  warning {w}")
        for k, v in self.stats.items():
            lines.append(f"  stat    {k}: {v}")
        return "\n".join(lines)


def validate_matches(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport(n_rows=len(df))

    # --- Hard invariants (errors) -----------------------------------------
    required = ["date", "home_team", "away_team", "fthg", "ftag"]
    for c in required:
        if c not in df.columns:
            rep.errors.append(f"missing required column: {c}")
    if rep.errors:
        return rep

    if df["date"].isna().any():
        rep.errors.append(f"{int(df['date'].isna().sum())} rows with null date")
    if (df["fthg"] < 0).any() or (df["ftag"] < 0).any():
        rep.errors.append("negative goal counts present")
    self_play = df["home_team"] == df["away_team"]
    if self_play.any():
        rep.errors.append(f"{int(self_play.sum())} rows where team plays itself")

    # Half-time goals cannot exceed full-time (when both present)
    if {"hthg", "fthg"}.issubset(df.columns):
        bad = (df["hthg"].notna() & (df["hthg"] > df["fthg"]))
        if bad.any():
            rep.errors.append(f"{int(bad.sum())} rows with HT goals > FT goals (home)")

    # --- Soft quality checks (warnings) -----------------------------------
    if {"odds_h", "odds_d", "odds_a"}.issubset(df.columns):
        no_odds = df[["odds_h", "odds_d", "odds_a"]].isna().any(axis=1)
        rep.stats["pct_rows_missing_1x2_odds"] = round(100 * no_odds.mean(), 1)
        if no_odds.mean() > 0.5:
            rep.warnings.append("majority of rows lack 1X2 odds (expected pre-2002)")
        # Overround sanity: sum of implied probs should be > 1 (the vig)
        with np.errstate(divide="ignore"):
            impl = (1 / df["odds_h"] + 1 / df["odds_d"] + 1 / df["odds_a"])
        weird = impl.notna() & ((impl < 1.0) | (impl > 1.5))
        if weird.any():
            rep.warnings.append(f"{int(weird.sum())} rows with implausible overround")

    # Duplicate fixtures
    dupes = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
    if dupes:
        rep.warnings.append(f"{int(dupes)} duplicate (date, home, away) rows")

    # Coverage stats
    rep.stats["date_range"] = f"{df['date'].min().date()} .. {df['date'].max().date()}"
    rep.stats["n_leagues"] = df["league"].nunique() if "league" in df else "n/a"
    rep.stats["n_teams"] = pd.unique(
        df[["home_team", "away_team"]].values.ravel()).size
    rep.stats["home_win_rate"] = round((df["ftr"] == "H").mean(), 3)
    rep.stats["draw_rate"] = round((df["ftr"] == "D").mean(), 3)
    rep.stats["away_win_rate"] = round((df["ftr"] == "A").mean(), 3)
    rep.stats["mean_goals_per_game"] = round((df["fthg"] + df["ftag"]).mean(), 3)
    return rep

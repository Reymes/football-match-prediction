"""Dynamic Elo rating engine.

A goal-difference-aware club Elo with home-field advantage and per-season
regression to the mean. Ratings are updated *after* each match; the value
attached to a fixture is always the rating that existed **before** kickoff,
so the feature is leakage-free by construction.

Elo is one of the strongest single predictors of match outcome and gives the
gradient-boosted models a compact, well-behaved strength signal that the
Dixon-Coles attack/defence parameters complement rather than duplicate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EloEngine:
    base: float = 1500.0
    k: float = 20.0                 # base learning rate
    home_advantage: float = 65.0    # rating points added to the home side
    season_regression: float = 0.25  # fraction pulled back to `base` each new season
    promoted_penalty: float = 40.0   # new (unseen) teams start slightly below base

    ratings: dict = field(default_factory=dict)   # (league, team) -> rating
    _last_season: dict = field(default_factory=dict)  # league -> season string

    def _key(self, league: str, team: str) -> tuple:
        return (league, team)

    def get(self, league: str, team: str) -> float:
        return self.ratings.get(self._key(league, team),
                                self.base - self.promoted_penalty)

    def _maybe_regress(self, league: str, season: str) -> None:
        """At a season boundary, pull every team in the league toward the mean.

        Carry-over form is real but attenuated across summers; regression also
        stops ratings drifting apart without bound."""
        if self._last_season.get(league) == season:
            return
        self._last_season[league] = season
        for (lg, team), r in list(self.ratings.items()):
            if lg == league:
                self.ratings[(lg, team)] = (
                    r + self.season_regression * (self.base - r))

    @staticmethod
    def _margin_multiplier(goal_diff: int, elo_diff: float) -> float:
        """Bigger wins move ratings more, with the standard autocorrelation
        correction (favourites winning big move less). From clubelo/538."""
        gd = abs(goal_diff)
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = (11 + gd) / 8.0
        return g * (2.2 / (0.001 * abs(elo_diff) + 2.2))

    def update(self, league, season, home, away, home_goals, away_goals):
        """Process one match; return (pre_home, pre_away, exp_home).

        If the result is unknown (goals are None/NaN — i.e. an upcoming fixture)
        we read the current ratings for features but perform NO update, so
        predicting a future match never mutates state.
        """
        self._maybe_regress(league, season)
        rh = self.get(league, home)
        ra = self.get(league, away)
        exp_home = 1.0 / (1.0 + 10 ** (-((rh + self.home_advantage) - ra) / 400))

        if home_goals is None or away_goals is None or \
                (isinstance(home_goals, float) and np.isnan(home_goals)) or \
                (isinstance(away_goals, float) and np.isnan(away_goals)):
            return rh, ra, exp_home

        if home_goals > away_goals:
            score = 1.0
        elif home_goals < away_goals:
            score = 0.0
        else:
            score = 0.5

        elo_diff = (rh + self.home_advantage) - ra
        mult = self._margin_multiplier(home_goals - away_goals, elo_diff)
        delta = self.k * mult * (score - exp_home)

        self.ratings[self._key(league, home)] = rh + delta
        self.ratings[self._key(league, away)] = ra - delta
        return rh, ra, exp_home


def add_elo_features(df: pd.DataFrame, engine: EloEngine | None = None
                     ) -> pd.DataFrame:
    """Attach pre-match Elo features. Assumes ``df`` is sorted chronologically.

    Adds:
      home_elo, away_elo     — ratings BEFORE the match
      elo_diff               — (home_elo + HFA) - away_elo
      elo_exp_home           — Elo win expectancy for the home side (0..1)
    """
    engine = engine or EloEngine()
    df = df.sort_values(["date", "league"]).reset_index(drop=True)
    hh = np.empty(len(df)); aa = np.empty(len(df)); ex = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        hg = row.fthg if pd.isna(row.fthg) else int(row.fthg)
        ag = row.ftag if pd.isna(row.ftag) else int(row.ftag)
        rh, ra, exp = engine.update(
            row.league, row.season, row.home_team, row.away_team, hg, ag)
        hh[i], aa[i], ex[i] = rh, ra, exp
    df["home_elo"] = hh
    df["away_elo"] = aa
    df["elo_diff"] = (hh + engine.home_advantage) - aa
    df["elo_exp_home"] = ex
    return df

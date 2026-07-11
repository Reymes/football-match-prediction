"""Baselines the system must beat (or match).

The bookmaker's de-vigged 1X2 line is a famously strong baseline: markets
aggregate enormous information and are hard to out-predict on log-loss. If our
model can't match the *pre-match* line it isn't ready; the real target is to
add value on top of it (via the ensemble) and to be at least as well-calibrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import market_implied_probs


class MarketBaseline:
    """Predicts the de-vigged bookmaker 1X2 probabilities."""

    name = "market"

    def predict_proba(self, df: pd.DataFrame, prefix: str = "odds_") -> np.ndarray:
        p = market_implied_probs(df, prefix=prefix)
        return p[["p_h", "p_d", "p_a"]].to_numpy()


class UniformBaseline:
    """Predicts the marginal base rate (home/draw/away) — the weakest baseline."""

    name = "base_rate"

    def __init__(self, rates=(0.46, 0.26, 0.28)):
        self.rates = np.array(rates) / np.sum(rates)

    def fit(self, df: pd.DataFrame):
        counts = df["ftr"].value_counts(normalize=True)
        self.rates = np.array([counts.get("H", 0.46),
                               counts.get("D", 0.26),
                               counts.get("A", 0.28)])
        self.rates = self.rates / self.rates.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.rates, (len(df), 1))

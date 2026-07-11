"""Clearly-isolated SYNTHETIC fixture generator for the decision layer.

Per CLAUDE.md and bet.md: when licensed real data is unavailable we use
synthetic data, but it must be ISOLATED and never presented as real model
performance. Every caller of this module labels its output as synthetic.

This produces a deterministic multi-season canonical match frame with plausible
goals and pre-match 1X2 odds, suitable for exercising the walk-forward decision
backtest offline (no network, no real results).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import CANONICAL_COLUMNS


def make_synthetic_matches(n_teams: int = 16, seasons: int = 6,
                           seed: int = 1) -> pd.DataFrame:
    """Deterministic synthetic league history. NOT real data."""
    rng = np.random.default_rng(seed)
    teams = [f"SYN{i:02d}" for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.35) for t in teams}
    rows = []
    date = pd.Timestamp("2019-08-03")
    mid = 0
    for s in range(seasons):
        for _ in range(34):
            order = rng.permutation(teams)
            for k in range(0, n_teams - 1, 2):
                h, a = order[k], order[k + 1]
                lam = np.exp(0.25 + strength[h] - 0.5 * strength[a])
                mu = np.exp(0.0 + strength[a] - 0.5 * strength[h])
                hg = int(rng.poisson(max(lam, 0.1)))
                ag = int(rng.poisson(max(mu, 0.1)))
                ftr = "H" if hg > ag else ("A" if ag > hg else "D")
                base = np.array([
                    0.45 + 0.3 * (strength[h] - strength[a]), 0.27,
                    0.28 - 0.3 * (strength[h] - strength[a])])
                base = np.clip(base + rng.normal(0, 0.02, 3), 0.05, None)
                base = base / base.sum()
                odds = np.round(1.06 / base, 2)
                rows.append({
                    "match_id": f"SYN|{2019+s}|{date:%Y%m%d}|{h}|{a}|{mid}",
                    "league": "SYN-LEAGUE", "season": f"{2019+s}-{2020+s}",
                    "date": date, "time": "15:00", "home_team": h, "away_team": a,
                    "fthg": hg, "ftag": ag, "ftr": ftr,
                    "hs": hg + 6, "as_": ag + 5, "hst": hg + 2, "ast": ag + 2,
                    "hc": 5, "ac": 4,
                    "odds_h": odds[0], "odds_d": odds[1], "odds_a": odds[2]})
                mid += 1
            date += pd.Timedelta(days=7)
        date += pd.Timedelta(days=21)
    df = pd.DataFrame(rows)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CANONICAL_COLUMNS]

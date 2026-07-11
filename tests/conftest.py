"""Shared fixtures: a small, deterministic, leakage-testable match dataset."""
import numpy as np
import pandas as pd
import pytest

from match_predict.data.schema import CANONICAL_COLUMNS


def _make_matches(n_teams=6, n_rounds=12, seed=0):
    """Deterministic double round-robin-ish fixture list with plausible stats."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strengths = {t: rng.normal(0, 0.3) for t in teams}
    rows = []
    date = pd.Timestamp("2022-08-06")
    mid = 0
    for _ in range(n_rounds):
        order = rng.permutation(teams)
        for k in range(0, n_teams - 1, 2):
            h, a = order[k], order[k + 1]
            lam = np.exp(0.2 + strengths[h] - strengths[a] * 0.5)
            mu = np.exp(0.0 + strengths[a] - strengths[h] * 0.5)
            hg = int(rng.poisson(max(lam, 0.1)))
            ag = int(rng.poisson(max(mu, 0.1)))
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            # crude but valid pre-match odds (over-round ~1.05)
            base = np.array([0.45, 0.27, 0.28]) + rng.normal(0, 0.03, 3)
            base = np.clip(base, 0.05, None)
            base = base / base.sum()
            odds = np.round(1.05 / base, 2)
            rows.append({
                "match_id": f"L1|2022-2023|{date:%Y%m%d}|{h}|{a}|{mid}",
                "league": "L1", "season": "2022-2023", "date": date,
                "time": "15:00", "home_team": h, "away_team": a,
                "fthg": hg, "ftag": ag, "ftr": ftr,
                "hs": hg + 6, "as_": ag + 5, "hst": hg + 2, "ast": ag + 2,
                "hc": 5, "ac": 4,
                "odds_h": odds[0], "odds_d": odds[1], "odds_a": odds[2],
            })
            mid += 1
        date += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CANONICAL_COLUMNS]


@pytest.fixture(scope="session")
def matches():
    return _make_matches()


@pytest.fixture(scope="session")
def feat(matches):
    from match_predict.features.build import build_feature_frame
    return build_feature_frame(matches)

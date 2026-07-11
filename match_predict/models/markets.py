"""Turn a goal-expectation pair (λ_home, λ_away) into a joint score matrix,
then derive every betting market from that single matrix.

This is the design principle in task.md: we NEVER predict correct scores as
classification labels. We predict two goal rates, build one coherent joint
distribution, and read all markets (1X2, O/U, BTTS, correct score, Asian
handicap, team totals) off it. Every market is therefore internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson


def _dc_correction(matrix, lam, mu, rho):
    """Apply the Dixon-Coles low-score adjustment to the 2x2 corner."""
    matrix = matrix.copy()
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho
    return matrix


def score_matrix(lam: float, mu: float, rho: float = 0.0,
                 max_goals: int = 10) -> np.ndarray:
    """P(home=i, away=j) for i,j in 0..max_goals, DC-corrected & normalised."""
    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, lam)
    pa = poisson.pmf(i, mu)
    m = np.outer(ph, pa)
    if rho:
        m = _dc_correction(m, lam, mu, rho)
    m = np.clip(m, 0, None)
    return m / m.sum()


@dataclass
class MarketBook:
    """All derived markets for one fixture."""
    lam_home: float
    lam_away: float
    p_home: float
    p_draw: float
    p_away: float
    over_under: dict = field(default_factory=dict)      # line -> {"over","under"}
    btts: dict = field(default_factory=dict)            # {"yes","no"}
    correct_score: list = field(default_factory=list)   # [((i,j), prob), ...]
    asian_handicap: dict = field(default_factory=dict)  # line -> {home,push,away}
    team_totals: dict = field(default_factory=dict)     # {"home":{...},"away":{...}}

    def top_scores(self, k: int = 5):
        return self.correct_score[:k]


def _totals_probs(matrix, lines=(0.5, 1.5, 2.5, 3.5, 4.5)):
    total = matrix.shape[0] - 1
    goals = np.add.outer(np.arange(total + 1), np.arange(total + 1))
    out = {}
    for L in lines:
        over = matrix[goals > L].sum()
        out[L] = {"over": float(over), "under": float(1 - over)}
    return out


def _btts(matrix):
    yes = matrix[1:, 1:].sum()
    return {"yes": float(yes), "no": float(1 - yes)}


def _team_totals(matrix, lines=(0.5, 1.5, 2.5)):
    ph = matrix.sum(axis=1)   # marginal home goals
    pa = matrix.sum(axis=0)   # marginal away goals
    goals = np.arange(matrix.shape[0])
    def side(pmf):
        return {L: {"over": float(pmf[goals > L].sum()),
                    "under": float(pmf[goals <= L].sum())} for L in lines}
    return {"home": side(ph), "away": side(pa)}


def _margin_distribution(matrix):
    n = matrix.shape[0] - 1
    margins = np.add.outer(np.arange(n + 1), -np.arange(n + 1))  # i - j
    dist = {}
    for m in range(-n, n + 1):
        dist[m] = float(matrix[margins == m].sum())
    return dist


def _ah_single(margin_dist, line):
    """Home-handicap ``line`` (added to home score). Returns home/push/away."""
    home = push = away = 0.0
    for m, p in margin_dist.items():
        adj = m + line
        if adj > 1e-9:
            home += p
        elif adj < -1e-9:
            away += p
        else:
            push += p
    return {"home": home, "push": push, "away": away}


def _asian_handicap(matrix, lines=(-1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
                                   0.25, 0.5, 0.75, 1.0, 1.5)):
    md = _margin_distribution(matrix)
    out = {}
    for L in lines:
        # quarter lines split between the two neighbouring half/whole lines
        frac = round(L * 4) % 2
        if frac == 1:  # quarter line
            a = _ah_single(md, L - 0.25)
            b = _ah_single(md, L + 0.25)
            out[L] = {k: (a[k] + b[k]) / 2 for k in ("home", "push", "away")}
        else:
            out[L] = _ah_single(md, L)
    return out


def derive_markets(matrix: np.ndarray, lam_home: float, lam_away: float,
                   top_n: int = 12) -> MarketBook:
    """Read all markets off a joint score matrix."""
    n = matrix.shape[0]
    tri = np.tril_indices(n, -1)          # i>j : home win
    p_home = float(matrix[tri].sum())
    p_draw = float(np.trace(matrix))
    p_away = float(1 - p_home - p_draw)

    # correct score ranking
    flat = [((i, j), float(matrix[i, j]))
            for i in range(n) for j in range(n)]
    flat.sort(key=lambda kv: kv[1], reverse=True)

    return MarketBook(
        lam_home=lam_home, lam_away=lam_away,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        over_under=_totals_probs(matrix),
        btts=_btts(matrix),
        correct_score=flat[:top_n],
        asian_handicap=_asian_handicap(matrix),
        team_totals=_team_totals(matrix),
    )

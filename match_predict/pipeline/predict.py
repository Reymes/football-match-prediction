"""Assemble the final per-match prediction in the format task.md specifies.

Two probability sources are reconciled into ONE coherent object:

  * The calibrated stacked ensemble gives the headline 1X2 probabilities
    (its meta-learner blends market + GBM + Dixon-Coles and is the best
    single estimate of the outcome).
  * The Dixon-Coles score matrix gives the full joint goal distribution, from
    which correct-score / O-U / BTTS / Asian-handicap / team-totals are read.

`reconcile_matrix_to_1x2` rescales the three outcome regions of the score
matrix so its H/D/A marginals exactly match the ensemble headline, while
preserving the within-outcome score shape. Every reported market is then
mutually consistent.

Explanations combine per-match LightGBM SHAP values with plain-language reasons
derived from the strongest signals (Elo gap, form, market lean, rest).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.markets import derive_markets


def matrix_expected_goals(matrix: np.ndarray) -> tuple:
    """Expected (home, away) goals implied by a joint score matrix.

    These are the values that are actually consistent with every market read
    off ``matrix`` (correct score, O/U, totals). After reconciliation the raw
    model rates (lambda/mu) no longer match the matrix, so this is what must be
    displayed as the headline expected goals (task/fix.md §13).
    """
    n = matrix.shape[0]
    idx = np.arange(n)
    eg_home = float((matrix.sum(axis=1) * idx).sum())
    eg_away = float((matrix.sum(axis=0) * idx).sum())
    return eg_home, eg_away


def reconcile_matrix_to_1x2(matrix: np.ndarray, target_hda) -> np.ndarray:
    """Rescale score-matrix outcome regions to match target (H, D, A)."""
    n = matrix.shape[0]
    i = np.arange(n)
    home_mask = i[:, None] > i[None, :]
    away_mask = i[:, None] < i[None, :]
    draw_mask = i[:, None] == i[None, :]
    cur = np.array([matrix[home_mask].sum(),
                    matrix[draw_mask].sum(),
                    matrix[away_mask].sum()])
    cur = np.clip(cur, 1e-9, None)
    tgt = np.array(target_hda, float)
    scale = tgt / cur
    out = matrix.copy()
    out[home_mask] *= scale[0]
    out[draw_mask] *= scale[1]
    out[away_mask] *= scale[2]
    return out / out.sum()


@dataclass
class MatchPrediction:
    league: str
    date: str
    home_team: str
    away_team: str
    prob_home: float
    prob_draw: float
    prob_away: float
    exp_goals_home: float                    # matrix-derived (matches score matrix)
    exp_goals_away: float                     # matrix-derived (matches score matrix)
    model_rate_home: float = None            # raw Dixon-Coles lambda (pre-reconcile)
    model_rate_away: float = None            # raw Dixon-Coles mu (pre-reconcile)
    top_scores: list = field(default_factory=list)
    btts: dict = field(default_factory=dict)
    over_under: dict = field(default_factory=dict)
    asian_handicap: dict = field(default_factory=dict)
    team_totals: dict = field(default_factory=dict)
    confidence: float = 0.0
    uncertainty: float = 0.0
    reasons: list = field(default_factory=list)
    feature_importance: dict = field(default_factory=dict)
    actual: str = None                       # filled in backtests

    def to_dict(self):
        return self.__dict__

    def pretty(self) -> str:
        L = []
        L.append(f"── {self.home_team} vs {self.away_team}  "
                 f"({self.league}, {self.date}) ──")
        L.append(f"  1X2      : Home {self.prob_home:5.1%} | "
                 f"Draw {self.prob_draw:5.1%} | Away {self.prob_away:5.1%}")
        L.append(f"  Exp goals: {self.exp_goals_home:.2f} - {self.exp_goals_away:.2f}"
                 f"  (from score matrix)")
        ts = "  ".join(f"{i}-{j} {p:4.1%}" for (i, j), p in self.top_scores[:5])
        L.append(f"  Top scores: {ts}")
        L.append(f"  BTTS yes : {self.btts.get('yes', 0):.1%}   "
                 f"O/U 2.5 over: {self.over_under.get(2.5, {}).get('over', 0):.1%}")
        ah05 = self.asian_handicap.get(-0.5, {})
        L.append(f"  AH home -0.5: {ah05.get('home', 0):.1%}   "
                 f"confidence: {self.confidence:.2f}  (uncertainty {self.uncertainty:.2f})")
        if self.actual is not None:
            L.append(f"  ACTUAL   : {self.actual}")
        L.append("  Why:")
        for r in self.reasons:
            L.append(f"     • {r}")
        return "\n".join(L)


def _entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))   # 0..1 (1=max unsure)


def _reasons_from_row(row, hda, shap_top=None):
    """Plain-language drivers from the strongest pre-match signals."""
    reasons = []
    labels = ["home win", "draw", "away win"]
    pick = int(np.argmax(hda))
    reasons.append(f"Model leans toward a {labels[pick]} ({hda[pick]:.0%}).")

    elo_d = row.get("elo_diff", np.nan)
    if np.isfinite(elo_d):
        if abs(elo_d) > 50:
            side = row["home_team"] if elo_d > 0 else row["away_team"]
            reasons.append(f"{side} is materially stronger on Elo "
                           f"(gap {elo_d:+.0f} incl. home advantage).")
    fpd = row.get("form_pts_diff", np.nan)
    if np.isfinite(fpd) and abs(fpd) > 0.6:
        side = row["home_team"] if fpd > 0 else row["away_team"]
        reasons.append(f"{side} in better recent form "
                       f"({fpd:+.2f} pts/game over last matches).")
    mh, ma = row.get("mkt_prob_h", np.nan), row.get("mkt_prob_a", np.nan)
    if np.isfinite(mh):
        reasons.append(f"Market implies H/D/A "
                       f"{mh:.0%}/{row.get('mkt_prob_d', np.nan):.0%}/{ma:.0%}.")
    rd = row.get("rest_diff", np.nan)
    if np.isfinite(rd) and abs(rd) >= 3:
        side = row["home_team"] if rd > 0 else row["away_team"]
        reasons.append(f"{side} better rested ({rd:+.0f} days).")
    if shap_top:
        top = ", ".join(f"{k} ({v:+.2f})" for k, v in shap_top)
        reasons.append(f"Top model drivers (SHAP, predicted class): {top}.")
    return reasons


def format_prediction(row, ensemble_hda, lam, mu, rho, max_goals=10,
                      gbm=None, gbm_row_df=None, importance=None) -> MatchPrediction:
    """Build a full MatchPrediction from a feature row + model outputs."""
    from ..models.markets import score_matrix
    matrix = score_matrix(lam, mu, rho=rho, max_goals=max_goals)
    matrix = reconcile_matrix_to_1x2(matrix, ensemble_hda)
    book = derive_markets(matrix, lam, mu)
    # Displayed expected goals must be read off the FINAL (reconciled) matrix,
    # not the raw model rates — otherwise they contradict the shown markets.
    eg_home, eg_away = matrix_expected_goals(matrix)

    conf = float(np.max(ensemble_hda))
    unc = _entropy(ensemble_hda)

    shap_top = None
    if gbm is not None and gbm_row_df is not None:
        cls = int(np.argmax(ensemble_hda))
        contrib = gbm.shap_contributions(gbm_row_df, cls).iloc[0]
        shap_top = list(contrib.reindex(contrib.abs().sort_values(
            ascending=False).index).head(4).round(3).items())

    reasons = _reasons_from_row(row, ensemble_hda, shap_top)

    return MatchPrediction(
        league=row["league"], date=str(row["date"])[:10],
        home_team=row["home_team"], away_team=row["away_team"],
        prob_home=float(ensemble_hda[0]), prob_draw=float(ensemble_hda[1]),
        prob_away=float(ensemble_hda[2]),
        exp_goals_home=eg_home, exp_goals_away=eg_away,
        model_rate_home=float(lam), model_rate_away=float(mu),
        top_scores=book.top_scores(12), btts=book.btts,
        over_under=book.over_under, asian_handicap=book.asian_handicap,
        team_totals=book.team_totals,
        confidence=round(conf, 3), uncertainty=round(unc, 3),
        reasons=reasons,
        feature_importance=(dict(importance.head(8).round(1))
                            if importance is not None else {}),
    )

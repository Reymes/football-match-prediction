"""Dixon-Coles (1997) time-weighted bivariate-Poisson-style goal model.

This is the statistical heart of the system. It estimates, per team, an
attack and defence strength plus a global home advantage, and a low-score
dependence parameter ``rho`` that corrects the independent-Poisson model's
well-known mis-fit on 0-0 / 1-0 / 0-1 / 1-1 scores.

Goal rates for a fixture:
    log λ_home = μ + home_adv + attack_home - defence_away
    log λ_away = μ           + attack_away - defence_home

Fitting maximises a *time-weighted* log-likelihood: match ``t`` gets weight
    w(t) = exp(-ξ · Δdays)
so recent form dominates. Weakly-observed teams are shrunk toward the league
mean by a small L2 penalty on attack/defence (this also removes the additive
identifiability ridge, keeping the optimiser well-conditioned).

Only matches strictly BEFORE the prediction reference date are ever passed to
``fit`` — the walk-forward backtester enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correlation correction (vectorised)."""
    out = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColes:
    xi: float = 0.0018          # time-decay per day (~0.66/yr half-life ≈ 1yr)
    reg: float = 0.01           # L2 shrinkage on attack/defence
    max_goals: int = 10         # score-matrix truncation

    teams_: list = field(default_factory=list)
    idx_: dict = field(default_factory=dict)
    attack_: np.ndarray = None
    defence_: np.ndarray = None
    home_adv_: float = 0.25
    intercept_: float = 0.0
    rho_: float = -0.05
    fitted_: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, ref_date=None) -> "DixonColes":
        """Fit on `matches` (needs home_team, away_team, fthg, ftag, date).

        `ref_date` anchors the time-decay weights (defaults to latest date).
        """
        m = matches.dropna(subset=["fthg", "ftag"]).copy()
        m = m[m["home_team"].notna() & m["away_team"].notna()]
        if ref_date is None:
            ref_date = m["date"].max()
        ref_date = pd.Timestamp(ref_date)

        teams = sorted(set(m["home_team"]) | set(m["away_team"]))
        self.teams_ = teams
        self.idx_ = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = m["home_team"].map(self.idx_).to_numpy()
        ai = m["away_team"].map(self.idx_).to_numpy()
        hg = m["fthg"].to_numpy(dtype=float)
        ag = m["ftag"].to_numpy(dtype=float)
        age = (ref_date - m["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * np.clip(age, 0, None))

        lg_hg = gammaln(hg + 1.0)
        lg_ag = gammaln(ag + 1.0)

        def unpack(p):
            att = p[:n]
            dfc = p[n:2 * n]
            home_adv = p[2 * n]
            intercept = p[2 * n + 1]
            rho = p[2 * n + 2]
            return att, dfc, home_adv, intercept, rho

        def nll(p):
            att, dfc, home_adv, intercept, rho = unpack(p)
            log_lh = intercept + home_adv + att[hi] - dfc[ai]
            log_la = intercept + att[ai] - dfc[hi]
            lam = np.exp(log_lh)
            mu = np.exp(log_la)
            # clip rho to the region where tau stays positive-ish
            rho = np.clip(rho, -0.2, 0.2)
            tau = _dc_tau(hg, ag, lam, mu, rho)
            tau = np.clip(tau, 1e-6, None)
            ll = (hg * log_lh - lam - lg_hg) + (ag * log_la - mu - lg_ag) + np.log(tau)
            wnll = -np.sum(w * ll)
            wnll += 0.5 * self.reg * (np.sum(att ** 2) + np.sum(dfc ** 2))
            return wnll

        x0 = np.concatenate([
            np.zeros(n), np.zeros(n),
            [0.25, np.log(max(hg.mean(), 0.2)), -0.05]])
        bounds = ([(-3, 3)] * n + [(-3, 3)] * n +
                  [(-1, 1), (-2, 2), (-0.2, 0.2)])
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-9})

        att, dfc, home_adv, intercept, rho = unpack(res.x)
        # centre attack/defence for interpretability (prediction-invariant)
        att = att - att.mean()
        dfc = dfc - dfc.mean()
        self.attack_, self.defence_ = att, dfc
        self.home_adv_, self.intercept_, self.rho_ = home_adv, intercept, float(rho)
        self.fitted_ = True
        return self

    # -------------------------------------------------------------- predict
    def _strength(self, team, is_home):
        """Attack/defence for a team; unseen teams -> league-average (0)."""
        i = self.idx_.get(team)
        if i is None:
            return 0.0, 0.0
        return self.attack_[i], self.defence_[i]

    def expected_goals(self, home_team, away_team):
        """Return (λ_home, λ_away) expected goals for the fixture."""
        a_h, d_h = self._strength(home_team, True)
        a_a, d_a = self._strength(away_team, False)
        lam = np.exp(self.intercept_ + self.home_adv_ + a_h - d_a)
        mu = np.exp(self.intercept_ + a_a - d_h)
        return float(lam), float(mu)

    def score_matrix(self, home_team, away_team):
        """Joint P(home=i, away=j) matrix with the Dixon-Coles correction."""
        from .markets import score_matrix as _sm
        lam, mu = self.expected_goals(home_team, away_team)
        return _sm(lam, mu, rho=self.rho_, max_goals=self.max_goals)

"""Walk-forward (out-of-time) backtest — the ONLY honest way to evaluate a
football model. There are no random splits anywhere in this file.

Timeline for a chosen ``test_start`` (and a validation window before it):

    [ ......... train ......... | ... validation ... | ...... test ...... ]
                              val_start            test_start

  * Dixon-Coles is refit on a rolling trailing window as time advances
    (`dc_refit_days`), always on matches strictly before the fixture.
  * The GBM is trained on everything before the period boundary (a fresh fit
    for validation, another for test) — no future rows ever enter training.
  * The stacked ensemble and the calibrator are fit on VALIDATION predictions
    (out-of-time), then frozen and applied to test.
  * Every model is scored on the same set of test fixtures (those with odds,
    so the market baseline is defined for a fair comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models import DixonColes, MarketBaseline, GBMOutcomeModel
from ..models.markets import score_matrix, derive_markets
from ..ensemble import StackedEnsemble
from ..calibration import TemperatureScaler
from ..features.build import FEATURE_COLUMNS
from .metrics import evaluate_proba, poisson_deviance

_LABEL = {"H": 0, "D": 1, "A": 2}


@dataclass
class BacktestResult:
    scorecards: dict = field(default_factory=dict)
    test_frame: pd.DataFrame = None
    probas: dict = field(default_factory=dict)         # model -> (n,3)
    ensemble: StackedEnsemble = None
    calibrator: TemperatureScaler = None
    gbm: GBMOutcomeModel = None
    dc_final: DixonColes = None
    features: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return (pd.DataFrame(self.scorecards).T
                [["n", "log_loss", "brier", "rps", "ece", "accuracy"]])


class WalkForwardBacktest:
    def __init__(self, dc_window_days: int = 900, dc_refit_days: int = 7,
                 xi: float = 0.0018, dc_reg: float = 0.01,
                 features: list | None = None, gbm_rounds: int = 300):
        self.dc_window_days = dc_window_days
        self.dc_refit_days = dc_refit_days
        self.xi = xi
        self.dc_reg = dc_reg
        self.features = features or FEATURE_COLUMNS
        self.gbm_rounds = gbm_rounds

    # -- rolling Dixon-Coles predictions for an arbitrary set of rows --------
    def _dc_rolling(self, history: pd.DataFrame, pred_rows: pd.DataFrame):
        """Return dict match_id -> (p_h,p_d,p_a,lam,mu), refit per league/week."""
        out = {}
        for league, rows in pred_rows.groupby("league"):
            pool = history[history.league == league]
            rows = rows.sort_values("date")
            model = None
            last_fit = None
            for r in rows.itertuples(index=False):
                d = r.date
                if (model is None or last_fit is None
                        or (d - last_fit).days >= self.dc_refit_days):
                    train = pool[(pool.date < d) &
                                 (pool.date >= d - pd.Timedelta(days=self.dc_window_days))]
                    if len(train) < 60:
                        train = pool[pool.date < d].tail(400)
                    if len(train) >= 40:
                        model = DixonColes(xi=self.xi, reg=self.dc_reg).fit(
                            train, ref_date=d)
                        last_fit = d
                if model is None:
                    continue
                lam, mu = model.expected_goals(r.home_team, r.away_team)
                M = score_matrix(lam, mu, rho=model.rho_, max_goals=10)
                bk = derive_markets(M, lam, mu)
                out[r.match_id] = (bk.p_home, bk.p_draw, bk.p_away, lam, mu)
        return out

    def _dc_probs_for(self, frame, dc_map):
        rows = [dc_map.get(mid) for mid in frame.match_id]
        p = np.array([[r[0], r[1], r[2]] if r else [np.nan] * 3 for r in rows])
        lam = np.array([r[3] if r else np.nan for r in rows])
        mu = np.array([r[4] if r else np.nan for r in rows])
        return p, lam, mu

    # -- main entry ---------------------------------------------------------
    def run(self, feat: pd.DataFrame, test_start, val_start,
            leagues=None, verbose=True) -> BacktestResult:
        feat = feat.sort_values(["date", "league"]).reset_index(drop=True)
        if leagues:
            feat = feat[feat.league.isin(leagues)]
        test_start = pd.Timestamp(test_start)
        val_start = pd.Timestamp(val_start)

        has_odds = feat[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
        val = feat[(feat.date >= val_start) & (feat.date < test_start) & has_odds].copy()
        test = feat[(feat.date >= test_start) & has_odds].copy()
        if verbose:
            print(f"walk-forward: train<{val_start.date()}  "
                  f"val=[{val_start.date()},{test_start.date()})={len(val)}  "
                  f"test>={test_start.date()}={len(test)}")

        market = MarketBaseline()

        # --- base predictions on VALIDATION (to train meta + calibrator) ---
        gbm_val = GBMOutcomeModel(self.features, num_rounds=self.gbm_rounds)
        gbm_val.fit(feat[feat.date < val_start])
        dc_val_map = self._dc_rolling(feat[feat.date < test_start], val)
        dc_val_p, _, _ = self._dc_probs_for(val, dc_val_map)
        base_val = {
            "market": market.predict_proba(val),
            "gbm": gbm_val.predict_proba(val),
            "dixon_coles": dc_val_p,
        }
        yv = val["ftr"].map(_LABEL).to_numpy()
        # drop val rows with any missing base pred
        ok = np.isfinite(np.hstack([base_val[k] for k in base_val])).all(axis=1)
        base_val = {k: v[ok] for k, v in base_val.items()}
        yv = yv[ok]

        ensemble = StackedEnsemble(["market", "gbm", "dixon_coles"]).fit(base_val, yv)
        ens_val = ensemble.predict_proba(base_val)
        calibrator = TemperatureScaler().fit(ens_val, yv)

        # --- final models for TEST -----------------------------------------
        gbm_final = GBMOutcomeModel(self.features, num_rounds=self.gbm_rounds)
        gbm_final.fit(feat[feat.date < test_start])
        dc_test_map = self._dc_rolling(feat[feat.date < test_start], test)
        dc_test_p, dc_lam, dc_mu = self._dc_probs_for(test, dc_test_map)

        base_test = {
            "market": market.predict_proba(test),
            "gbm": gbm_final.predict_proba(test),
            "dixon_coles": dc_test_p,
        }
        ok_t = np.isfinite(np.hstack([base_test[k] for k in base_test])).all(axis=1)
        test = test[ok_t].copy()
        base_test = {k: v[ok_t] for k, v in base_test.items()}
        dc_lam, dc_mu = dc_lam[ok_t], dc_mu[ok_t]
        yt = test["ftr"].map(_LABEL).to_numpy()

        ens_test = ensemble.predict_proba(base_test)
        ens_cal = calibrator.transform(ens_test)

        probas = {
            "market": base_test["market"],
            "dixon_coles": base_test["dixon_coles"],
            "gbm": base_test["gbm"],
            "ensemble": ens_test,
            "ensemble_cal": ens_cal,
        }
        scorecards = {name: evaluate_proba(p, yt) for name, p in probas.items()}
        # xG goodness-of-fit for the Dixon-Coles rates
        pd_home = poisson_deviance(test["fthg"].to_numpy(), dc_lam)
        pd_away = poisson_deviance(test["ftag"].to_numpy(), dc_mu)

        # final DC fit on ALL pre-test data, for single-match prediction API
        dc_final = DixonColes(xi=self.xi, reg=self.dc_reg)
        # fit per-league is done at predict time; here fit a global-ish recent pool
        # (the predict CLI refits per league anyway). Keep the latest per league.

        test = test.reset_index(drop=True)
        test["y"] = yt
        return BacktestResult(
            scorecards=scorecards, test_frame=test, probas=probas,
            ensemble=ensemble, calibrator=calibrator, gbm=gbm_final,
            dc_final=dc_final, features=self.features,
            extra={"ensemble_influence": ensemble.influence,
                   "temperature": calibrator.T,
                   "poisson_deviance_home": round(pd_home, 4),
                   "poisson_deviance_away": round(pd_away, 4),
                   "dc_lambda": dc_lam, "dc_mu": dc_mu})

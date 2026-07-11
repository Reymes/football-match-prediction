"""Production Predictor: load trained artifacts and predict real fixtures.

A `Predictor` bundles everything needed to score a brand-new fixture that is
NOT in the training data:

  * the canonical match history (to compute the fixture's pre-kickoff features)
  * per-league Dixon-Coles models fit as-of the latest known date
  * the final LightGBM 1X2 model (trained on all history)
  * the stacked ensemble + calibrator (fit on out-of-time validation)
  * a market-free fallback ensemble + calibrator (used when odds are missing)

`predict_fixtures` appends the requested fixtures (results unknown) to history,
rebuilds features — Elo/form/context for the new rows come only from prior
matches — and produces a full explained MatchPrediction for each.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib

from ..data.schema import CANONICAL_COLUMNS
from ..features.build import build_feature_frame, FEATURE_COLUMNS, market_implied_probs
from ..models.markets import score_matrix, derive_markets
from .predict import format_prediction


def infer_season(date: pd.Timestamp) -> str:
    """European season string, e.g. 2025-08-16 -> '2025-2026'."""
    y = date.year
    return f"{y}-{y+1}" if date.month >= 7 else f"{y-1}-{y}"


@dataclass
class Predictor:
    history: pd.DataFrame                      # canonical matches (results known)
    gbm: object
    ensemble: object                           # market + gbm + dixon_coles
    calibrator: object
    ensemble_nomkt: object                     # gbm + dixon_coles fallback
    calibrator_nomkt: object
    dc_models: dict                            # league -> DixonColes
    features: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    default_rho: float = -0.045
    trained_through: str = ""

    # ------------------------------------------------------------ persistence
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.history.to_parquet(os.path.join(path, "history.parquet")) \
            if _has_parquet() else \
            self.history.to_pickle(os.path.join(path, "history.pkl"))
        joblib.dump({
            "gbm": self.gbm, "ensemble": self.ensemble,
            "calibrator": self.calibrator,
            "ensemble_nomkt": self.ensemble_nomkt,
            "calibrator_nomkt": self.calibrator_nomkt,
            "dc_models": self.dc_models, "features": self.features,
            "default_rho": self.default_rho,
            "trained_through": self.trained_through,
        }, os.path.join(path, "models.joblib"))

    @classmethod
    def load(cls, path: str) -> "Predictor":
        pq = os.path.join(path, "history.parquet")
        pk = os.path.join(path, "history.pkl")
        history = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_pickle(pk)
        blob = joblib.load(os.path.join(path, "models.joblib"))
        return cls(history=history, **blob)

    # ------------------------------------------------------------ prediction
    def _fixtures_to_canonical(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        fx = fixtures.copy()
        fx["date"] = pd.to_datetime(fx["date"])
        if "season" not in fx:
            fx["season"] = fx["date"].map(infer_season)
        for c in CANONICAL_COLUMNS:
            if c not in fx:
                fx[c] = np.nan
        fx["match_id"] = (
            fx["league"].astype(str) + "|" + fx["season"].astype(str) + "|"
            + fx["date"].dt.strftime("%Y%m%d") + "|"
            + fx["home_team"].astype(str) + "|" + fx["away_team"].astype(str)
            + "|UPCOMING")
        return fx[CANONICAL_COLUMNS]

    def predict_fixtures(self, fixtures: pd.DataFrame) -> list:
        """`fixtures` needs league, date, home_team, away_team; odds optional."""
        fx = self._fixtures_to_canonical(fixtures)
        combined = pd.concat([self.history, fx], ignore_index=True)
        feat = build_feature_frame(combined)
        rows = feat[feat["match_id"].isin(fx["match_id"])].copy()
        rows = rows.set_index("match_id").loc[fx["match_id"]].reset_index()

        importance = self.gbm.feature_importance()
        preds = []
        for i in range(len(rows)):
            row = rows.iloc[i]
            league = row["league"]
            dc = self.dc_models.get(league)
            if dc is None:                       # unknown league -> skip
                continue
            lam, mu = dc.expected_goals(row["home_team"], row["away_team"])
            M = score_matrix(lam, mu, rho=dc.rho_, max_goals=10)
            dcp = derive_markets(M, lam, mu)
            p_dc = np.array([[dcp.p_home, dcp.p_draw, dcp.p_away]])
            p_gbm = self.gbm.predict_proba(rows.iloc[[i]])

            has_odds = np.isfinite([row.get("mkt_prob_h"), row.get("mkt_prob_d"),
                                    row.get("mkt_prob_a")]).all()
            if has_odds:
                p_mkt = market_implied_probs(rows.iloc[[i]])[["p_h", "p_d", "p_a"]].to_numpy()
                base = {"market": p_mkt, "gbm": p_gbm, "dixon_coles": p_dc}
                ens = self.ensemble.predict_proba(base)
                hda = self.calibrator.transform(ens)[0]
            else:
                base = {"gbm": p_gbm, "dixon_coles": p_dc}
                ens = self.ensemble_nomkt.predict_proba(base)
                hda = self.calibrator_nomkt.transform(ens)[0]

            pred = format_prediction(
                row, hda, lam, mu, rho=dc.rho_, gbm=self.gbm,
                gbm_row_df=rows.iloc[[i]], importance=importance)
            preds.append(pred)
        return preds


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

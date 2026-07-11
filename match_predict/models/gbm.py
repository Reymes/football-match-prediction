"""LightGBM 1X2 outcome model.

A gradient-boosted multiclass classifier over the engineered pre-kickoff
features (Elo, form, context, market signal). It captures non-linear
interactions the Poisson family can't (e.g. congestion × form, market-vs-Elo
disagreement) and is a complementary member of the ensemble.

Native NaN handling means the pre-2000 rows with missing shot stats need no
imputation. `predict_proba` returns columns in [H, D, A] order to match the
rest of the system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                       # pragma: no cover
    _HAS_LGB = False

# label encoding used internally
_CLASSES = ["H", "D", "A"]
_LABEL = {"H": 0, "D": 1, "A": 2}


class GBMOutcomeModel:
    name = "gbm"

    def __init__(self, features, params=None, num_rounds: int = 300):
        if not _HAS_LGB:
            raise ImportError("lightgbm is required for GBMOutcomeModel")
        self.features = list(features)
        self.num_rounds = num_rounds
        self.params = params or {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": 7,
        }
        self.model = None

    def fit(self, df: pd.DataFrame, sample_weight=None):
        X = df[self.features]
        y = df["ftr"].map(_LABEL).to_numpy()
        dtrain = lgb.Dataset(X, label=y, weight=sample_weight,
                             free_raw_data=False)
        self.model = lgb.train(self.params, dtrain,
                               num_boost_round=self.num_rounds)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Safety net: guarantee numeric dtype (a single-fixture frame with all
        # blank odds can arrive as object dtype).
        X = df[self.features].apply(pd.to_numeric, errors="coerce")
        proba = self.model.predict(X)
        return np.asarray(proba).reshape(-1, 3)   # already [H, D, A]

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def shap_contributions(self, df: pd.DataFrame, class_idx: int) -> pd.DataFrame:
        """Per-row SHAP contributions for one class (H=0,D=1,A=2).

        LightGBM's built-in ``pred_contrib`` returns exact tree SHAP values.
        For multiclass the layout is
        ``[class0 feats..., class0 bias, class1 feats..., class1 bias, ...]``.
        """
        contrib = self.model.predict(df[self.features], pred_contrib=True)
        contrib = np.asarray(contrib)
        width = len(self.features) + 1
        start = class_idx * width
        vals = contrib[:, start:start + len(self.features)]
        return pd.DataFrame(vals, columns=self.features, index=df.index)

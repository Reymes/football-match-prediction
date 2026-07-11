"""Stacked ensemble (a.k.a. stacking / stacked generalization).

Base models each emit a 3-class [H, D, A] probability vector:
    * Dixon-Coles      (statistical goal model)
    * LightGBM         (non-linear feature model)
    * Market           (de-vigged bookmaker line)

The meta-learner is a multinomial logistic regression trained on the LOG
probabilities of the base models. Log-space is important: it makes the meta
model a (regularised) geometric-mean/product-of-experts combiner rather than a
naive arithmetic blend, which is better behaved for probabilities.

Weights are LEARNED on out-of-time validation predictions — never assigned by
hand (a requirement in task.md). L2 regularisation keeps the blend from
overfitting a single base model on small validation sets.

Design notes:
  * Blending (a single hold-out set) vs stacking (out-of-fold): we use a rolling
    out-of-time validation window, the time-series-correct analogue of OOF.
  * Weighted averaging is the special case where the meta model is constrained
    to non-negative weights summing to 1; we allow the logistic meta-learner to
    also recalibrate, which subsumes it.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit_features(prob_list):
    """Stack base-model probability vectors into log-prob meta-features."""
    feats = [np.log(np.clip(p, _EPS, 1.0)) for p in prob_list]
    return np.hstack(feats)


class StackedEnsemble:
    def __init__(self, base_names, C: float = 1.0):
        self.base_names = list(base_names)
        self.C = C
        # sklearn>=1.7 always uses multinomial for multiclass; no multi_class arg.
        self.meta = LogisticRegression(C=C, max_iter=2000)
        self.classes_ = np.array([0, 1, 2])   # H, D, A
        self.fitted_ = False

    def fit(self, base_probs: dict, y: np.ndarray):
        """`base_probs`: {name -> (n,3) array}; `y`: int labels 0/1/2 (H/D/A)."""
        X = _logit_features([base_probs[n] for n in self.base_names])
        self.meta.fit(X, y)
        self.fitted_ = True
        return self

    def predict_proba(self, base_probs: dict) -> np.ndarray:
        X = _logit_features([base_probs[n] for n in self.base_names])
        proba = self.meta.predict_proba(X)
        # sklearn orders columns by sorted classes present; align to [H,D,A]
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self.meta.classes_):
            out[:, cls] = proba[:, col]
        return out

    @property
    def influence(self):
        """Relative per-base influence = normalised mean |coef| across classes.

        NOT mixture weights. A multinomial-logistic stacker on log-probabilities
        does not decompose into a convex combination of the base probability
        vectors, so these numbers must not be presented as "X% market + Y% GBM"
        (fix.md §9). They are a coarse feature-importance summary only; they sum
        to 1 by construction of the normalisation, not because they are
        probabilities over base models.
        """
        if not self.fitted_:
            return {}
        coef = np.abs(self.meta.coef_)            # (n_classes, 3*n_base)
        per_base = coef.reshape(coef.shape[0], len(self.base_names), 3)
        strength = per_base.mean(axis=(0, 2))
        strength = strength / strength.sum()
        return dict(zip(self.base_names, strength.round(3)))

    # Backwards-compatible alias; prefer `.influence`.
    weights = influence

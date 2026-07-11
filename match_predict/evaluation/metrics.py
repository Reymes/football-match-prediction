"""Probabilistic evaluation metrics — the ones that matter for calibrated
forecasting, not just accuracy.

  * log_loss              — proper scoring rule; punishes confident wrong calls.
  * brier_score           — multiclass Brier (mean squared prob error).
  * ranked_probability_score — respects the ordinal H>D>A structure; the
                            standard metric for football 1X2 forecasts.
  * poisson_deviance      — goodness-of-fit for the goal-rate (xG) predictions.
  * accuracy              — reported, but never the optimisation target.

All take probabilities in [H, D, A] column order and integer labels 0/1/2.
"""
from __future__ import annotations

import numpy as np
from ..calibration.calibrate import expected_calibration_error

_EPS = 1e-15


def _clip(p):
    p = np.clip(p, _EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(proba, y):
    p = _clip(np.asarray(proba, float))
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier_score(proba, y):
    p = np.asarray(proba, float)
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - Y) ** 2, axis=1)))


def ranked_probability_score(proba, y):
    """RPS for ordered outcomes (H, D, A). Lower is better."""
    p = np.asarray(proba, float)
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), y] = 1.0
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(Y, axis=1)
    k = p.shape[1]
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / (k - 1)))


def poisson_deviance(y_true, lam):
    """Mean Poisson deviance between observed goals and predicted rate."""
    y = np.asarray(y_true, float)
    lam = np.clip(np.asarray(lam, float), _EPS, None)
    # y*log(y/lam) -> 0 as y -> 0; compute only on positive y to avoid log(0).
    ratio = np.divide(y, lam, out=np.ones_like(y), where=y > 0)
    term = np.where(y > 0, y * np.log(ratio), 0.0)
    return float(np.mean(2 * (term - (y - lam))))


def accuracy(proba, y):
    return float(np.mean(np.asarray(proba).argmax(axis=1) == y))


def evaluate_proba(proba, y, n_bins=15) -> dict:
    """One-stop scorecard for a set of 1X2 probabilities."""
    proba = np.asarray(proba, float)
    y = np.asarray(y)
    return {
        "log_loss": round(log_loss(proba, y), 4),
        "brier": round(brier_score(proba, y), 4),
        "rps": round(ranked_probability_score(proba, y), 4),
        "ece": round(expected_calibration_error(proba, y, n_bins), 4),
        "accuracy": round(accuracy(proba, y), 4),
        "n": int(len(y)),
    }

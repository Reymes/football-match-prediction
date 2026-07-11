"""Per-market historical validation profiles (bet.md §10, §18).

A market may only be evaluated if it carries an out-of-time validation profile
describing how well its probabilities behaved on data the thresholds were NOT
tuned on. The decision layer cannot repair a poor probability model, so these
profiles gate everything downstream (bet.md §18).

`MarketValidationProfile` holds, per market and (optionally) per probability
band:
  * forecast-quality metrics (log loss, Brier, calibration slope/intercept, ECE);
  * the sample size behind them;
  * whether the market passed the quality bar.

`build_profiles_from_backtest` turns a WalkForwardBacktest result into 1X2
profiles measured strictly out-of-time, so we never present tuned-in numbers as
validation. Markets without a real profile default to "unvalidated" and are
rejected by eligibility with INSUFFICIENT_HISTORICAL_SAMPLE.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BandProfile:
    lo: float
    hi: float
    n_samples: int
    calibration_error: float       # |empirical - predicted| in this band
    empirical_rate: float
    predicted_rate: float


@dataclass
class MarketValidationProfile:
    market: str
    n_samples: int = 0
    log_loss: float | None = None
    brier: float | None = None
    ece: float | None = None
    calibration_slope: float | None = None
    calibration_intercept: float | None = None
    passed_quality: bool = False
    bands: list = field(default_factory=list)     # list[BandProfile]
    notes: str = ""

    def band_for(self, p: float) -> BandProfile | None:
        for b in self.bands:
            if b.lo <= p < b.hi or (p == 1.0 and b.hi == 1.0):
                return b
        return None

    def calibration_error_at(self, p: float, default: float = 0.02) -> float:
        b = self.band_for(p)
        return b.calibration_error if b is not None else default

    def samples_at(self, p: float) -> int:
        b = self.band_for(p)
        return b.n_samples if b is not None else self.n_samples

    def to_dict(self) -> dict:
        return {
            "market": self.market, "n_samples": self.n_samples,
            "log_loss": self.log_loss, "brier": self.brier, "ece": self.ece,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
            "passed_quality": self.passed_quality, "notes": self.notes,
            "bands": [b.__dict__ for b in self.bands],
        }


def calibration_line(prob_win: np.ndarray, y_win: np.ndarray):
    """Logistic-style calibration slope/intercept via a simple linear fit of the
    binned empirical rate on the predicted rate (out-of-time diagnostic).

    A well-calibrated model has slope ~1 and intercept ~0.
    """
    prob_win = np.asarray(prob_win, float)
    y_win = np.asarray(y_win, float)
    if len(prob_win) < 10 or prob_win.std() < 1e-6:
        return None, None
    # bin into deciles, fit empirical vs predicted
    edges = np.quantile(prob_win, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None, None
    xs, ys = [], []
    for i in range(len(edges) - 1):
        m = (prob_win >= edges[i]) & (prob_win <= edges[i + 1])
        if m.sum() >= 5:
            xs.append(prob_win[m].mean())
            ys.append(y_win[m].mean())
    if len(xs) < 3:
        return None, None
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _bands(prob_win: np.ndarray, y_win: np.ndarray,
           edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list:
    prob_win = np.asarray(prob_win, float)
    y_win = np.asarray(y_win, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob_win >= lo) & (prob_win < hi) if hi < 1.0 else \
            (prob_win >= lo) & (prob_win <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        emp = float(y_win[m].mean())
        pred = float(prob_win[m].mean())
        out.append(BandProfile(lo=lo, hi=hi, n_samples=n,
                               calibration_error=abs(emp - pred),
                               empirical_rate=emp, predicted_rate=pred))
    return out


def build_1x2_profile_from_probs(proba: np.ndarray, y: np.ndarray,
                                 market: str = "match_winner",
                                 max_log_loss: float = 1.02,
                                 max_ece: float = 0.06,
                                 min_samples: int = 300) -> MarketValidationProfile:
    """Build a 1X2 validation profile from out-of-time (proba, y) pairs.

    Probabilities are flattened to per-outcome win/no-win events so calibration
    is measured on the actual probability the model attaches to what happened.
    """
    from ..evaluation.metrics import log_loss, brier_score, \
        expected_calibration_error

    proba = np.asarray(proba, float)
    y = np.asarray(y, int)
    n = len(y)
    if n == 0:
        return MarketValidationProfile(market=market, notes="no samples")

    # per-outcome one-vs-rest for calibration diagnostics
    prob_win = proba.reshape(-1)
    y_win = np.zeros_like(proba)
    y_win[np.arange(n), y] = 1.0
    y_win = y_win.reshape(-1)

    ll = log_loss(proba, y)
    br = brier_score(proba, y)
    ece = expected_calibration_error(proba, y)
    slope, intercept = calibration_line(prob_win, y_win)
    bands = _bands(prob_win, y_win)

    passed = (n >= min_samples and ll <= max_log_loss and ece <= max_ece)
    return MarketValidationProfile(
        market=market, n_samples=n, log_loss=round(ll, 4),
        brier=round(br, 4), ece=round(ece, 4),
        calibration_slope=(round(slope, 3) if slope is not None else None),
        calibration_intercept=(round(intercept, 3) if intercept is not None else None),
        passed_quality=bool(passed), bands=bands,
        notes=("passed out-of-time quality bar" if passed
               else "FAILED quality bar — market should stay disabled"))


def profile_from_dict(d: dict) -> MarketValidationProfile:
    """Reconstruct a `MarketValidationProfile` from its `to_dict()` form."""
    bands = [BandProfile(**b) for b in d.get("bands", [])]
    return MarketValidationProfile(
        market=d["market"], n_samples=int(d.get("n_samples", 0)),
        log_loss=d.get("log_loss"), brier=d.get("brier"), ece=d.get("ece"),
        calibration_slope=d.get("calibration_slope"),
        calibration_intercept=d.get("calibration_intercept"),
        passed_quality=bool(d.get("passed_quality", False)),
        bands=bands, notes=d.get("notes", ""))


def save_profiles(profiles: dict, path: str, meta: dict | None = None) -> None:
    """Persist `{market: MarketValidationProfile}` (or dict form) to JSON.

    The written file carries the profiles under a ``profiles`` key plus optional
    ``meta`` (the out-of-time window etc.) so a served profile is auditable.
    """
    payload = {m: (p.to_dict() if hasattr(p, "to_dict") else p)
               for m, p in profiles.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta or {}, "profiles": payload}, fh, indent=2)


def load_profiles(path: str) -> dict:
    """Load `{market: MarketValidationProfile}` from a saved file.

    Returns an empty dict when the file is absent — the decision layer then
    treats every market as unvalidated and safely rejects (INSUFFICIENT_
    HISTORICAL_SAMPLE), so a missing profile file never crashes serving.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    raw = raw.get("profiles", raw)
    return {m: profile_from_dict(d) for m, d in raw.items()}


def build_profiles_from_backtest(result, min_samples: int = 300) -> dict:
    """Turn a WalkForwardBacktest result into a {market: profile} dict.

    Only the 1X2 (match_winner) market is derived here because that is what the
    walk-forward backtest scores directly. Score-derived markets (O/U, BTTS,
    correct score) require their own dedicated evaluation before enabling and
    default to unvalidated until backtest_decisions produces them.
    """
    y = result.test_frame["y"].to_numpy()
    proba = result.probas.get("ensemble_cal", result.probas.get("ensemble"))
    prof = build_1x2_profile_from_probs(proba, y, market="match_winner",
                                        min_samples=min_samples)
    return {"match_winner": prof}

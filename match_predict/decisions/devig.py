"""Margin removal (de-vigging) for bookmaker odds (bet.md §5).

Given the decimal odds for a set of mutually-exclusive, collectively-exhaustive
outcomes (e.g. 1X2, or over/under, or BTTS yes/no), turn them into margin-free
("fair") probabilities that sum to one. Three methods are supported and can be
compared:

  * normalized  — proportional to raw implied (1/odds), the simplest unbiased
                  baseline (matches features/build.py:market_implied_probs).
  * power       — raise implied probabilities to a common exponent k so they
                  sum to one; captures the favourite-longshot bias better.
  * shin        — Shin's model of insider trading; solves for z (the insider
                  fraction) so probabilities sum to one. Preferred for 2-3 way
                  markets (bet.md §5).

The offered odds themselves are never modified here; only the fair probability
is produced. Expected value always uses the ORIGINAL offered odds.
"""
from __future__ import annotations

import numpy as np


def _implied(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError(f"invalid decimal odds for de-vig: {odds!r}")
    return 1.0 / o


def overround(odds) -> float:
    """Bookmaker margin: sum of raw implied probabilities minus one."""
    return float(_implied(odds).sum() - 1.0)


def devig_normalized(odds) -> np.ndarray:
    imp = _implied(odds)
    return imp / imp.sum()


def devig_power(odds, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Solve sum(imp_i ** k) == 1 for k > 0 by bisection, then normalize."""
    imp = _implied(odds)
    # k in (0, large); f(k) = sum(imp**k) is monotone decreasing in k.
    lo, hi = 1e-6, 100.0

    def f(k):
        return float(np.sum(imp ** k)) - 1.0

    # sum(imp) >= 1 (overround), so f(1) >= 0; need f(hi) < 0.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val) < tol:
            break
        if val > 0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    p = imp ** k
    return p / p.sum()


def devig_shin(odds, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Shin (1992/1993) de-vig. Solves for insider proportion z in [0,1).

    p_i = ( sqrt(z^2 + 4(1-z) * imp_i^2 / S) - z ) / (2(1-z))
    where S = sum(imp) is the booksum. z is chosen so the p_i sum to one.
    Falls back to normalized for degenerate (single-outcome) inputs.
    """
    imp = _implied(odds)
    S = float(imp.sum())
    if imp.size < 2:
        return imp / S

    def probs(z):
        if z <= 0:
            return imp / S
        inner = z * z + 4.0 * (1.0 - z) * (imp ** 2) / S
        return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))

    def g(z):
        return float(probs(z).sum()) - 1.0

    lo, hi = 0.0, 0.999
    # g is monotone; g(0) = S - 1 >= 0.  Ensure a sign change.
    if g(hi) > 0:
        return imp / S
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = g(mid)
        if abs(val) < tol:
            break
        if val > 0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    p = probs(z)
    return p / p.sum()


_METHODS = {
    "normalized": devig_normalized,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds, method: str = "shin") -> np.ndarray:
    """Dispatch to a named de-vig method. Result sums to one."""
    m = _METHODS.get(method)
    if m is None:
        raise ValueError(f"unknown de-vig method {method!r}; "
                         f"choose from {sorted(_METHODS)}")
    return m(odds)


def compare_methods(odds) -> dict:
    """Return every method's fair probabilities plus the overround."""
    out = {"overround": overround(odds)}
    for name, fn in _METHODS.items():
        out[name] = fn(odds).tolist()
    return out

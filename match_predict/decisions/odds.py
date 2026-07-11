"""Bookmaker odds validation and timestamp safety (bet.md §5, §3).

Responsibilities:
  * validate decimal odds (finite, > 1.0);
  * convert offered odds -> raw implied probability (1 / decimal);
  * check that a price's timestamp is at or before the decision cutoff and
    not staler than the configured limit.

The offered odds are ALWAYS preserved for expected-value maths; the margin-free
market probability is a separate quantity computed in devig.py. Never confuse
"1 / odds" (offered implied, still carrying the bookmaker margin) with the fair
market probability (bet.md §5).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .schema import RejectionCode


def valid_decimal_odds(odds) -> bool:
    """A decimal price is valid iff it is finite and strictly greater than 1."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return False
    return np.isfinite(o) and o > 1.0


def offered_implied_probability(odds: float) -> float:
    """Raw implied probability = 1 / decimal odds (still includes margin)."""
    if not valid_decimal_odds(odds):
        raise ValueError(f"invalid decimal odds: {odds!r}")
    return 1.0 / float(odds)


def break_even_probability(odds: float) -> float:
    """Probability at which a bet at these odds has zero expected value.

    Identical to the offered implied probability, exposed under an explicit
    name because that is how it is used in the decision report (bet.md §11).
    """
    return offered_implied_probability(odds)


def _parse_ts(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def check_odds_timestamp(odds_timestamp, decision_cutoff,
                         max_staleness_hours: float,
                         require_timestamp: bool = True) -> list:
    """Timestamp-safety gate for a single price (bet.md §3, §5).

    Returns a list of RejectionCode; empty means the price is usable.
      * ODDS_IN_FUTURE   — captured strictly after the decision cutoff.
      * ODDS_STALE        — missing (when required) or older than the limit.
    """
    reasons: list = []
    cutoff = _parse_ts(decision_cutoff)
    ts = _parse_ts(odds_timestamp)

    if ts is None:
        if require_timestamp:
            reasons.append(RejectionCode.ODDS_STALE)
        return reasons
    if cutoff is not None:
        if ts > cutoff:
            reasons.append(RejectionCode.ODDS_IN_FUTURE)
        elif (cutoff - ts).total_seconds() > max_staleness_hours * 3600.0:
            reasons.append(RejectionCode.ODDS_STALE)
    return reasons


def validate_price(quote, decision_cutoff, max_staleness_hours: float,
                   require_timestamp: bool = True) -> list:
    """Full validation of one PriceQuote-like object. Returns RejectionCodes."""
    reasons: list = []
    if not quote.available:
        reasons.append(RejectionCode.MISSING_ODDS)
    if not valid_decimal_odds(quote.offered_odds):
        reasons.append(RejectionCode.INVALID_ODDS)
    reasons.extend(check_odds_timestamp(
        quote.odds_timestamp, decision_cutoff, max_staleness_hours,
        require_timestamp))
    return reasons

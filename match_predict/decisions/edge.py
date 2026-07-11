"""Edge and expected-value maths (bet.md §6, §7).

Definitions (kept deliberately explicit so percentage POINTS are never confused
with percentage CHANGE, per bet.md §6):

    probability_edge      = model_probability - market_fair_probability
    raw_expected_value    = model_probability        * offered_odds - 1
    conservative_ev       = conservative_probability * offered_odds - 1
    break_even_probability = 1 / offered_odds

A positive expected value does NOT mean the selection will win; it means the
model estimates positive expectation IF its probability is reliable.
"""
from __future__ import annotations


def probability_edge(model_probability: float,
                     market_fair_probability: float) -> float:
    """Signed edge in probability POINTS (not percentage change)."""
    return float(model_probability) - float(market_fair_probability)


def expected_value(probability: float, offered_odds: float) -> float:
    """Expected return per unit stake at the given probability & offered odds."""
    return float(probability) * float(offered_odds) - 1.0


def raw_expected_value(model_probability: float, offered_odds: float) -> float:
    return expected_value(model_probability, offered_odds)


def conservative_expected_value(conservative_probability: float,
                                offered_odds: float) -> float:
    return expected_value(conservative_probability, offered_odds)


def break_even_probability(offered_odds: float) -> float:
    """The probability that makes expected value zero at these odds."""
    return 1.0 / float(offered_odds)


def edge_summary(model_probability: float, market_fair_probability: float,
                 conservative_probability: float, offered_odds: float) -> dict:
    """One-shot bundle of every edge/EV quantity for a selection."""
    return {
        "probability_edge": probability_edge(model_probability,
                                             market_fair_probability),
        "raw_expected_value": raw_expected_value(model_probability, offered_odds),
        "conservative_expected_value": conservative_expected_value(
            conservative_probability, offered_odds),
        "break_even_probability": break_even_probability(offered_odds),
    }

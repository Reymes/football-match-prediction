"""Probability & odds tests (bet.md §25 - Probability and odds)."""
import numpy as np
import pytest

from match_predict.decisions import devig, edge
from match_predict.decisions.odds import (
    valid_decimal_odds, offered_implied_probability, break_even_probability)
from match_predict.decisions.eligibility import probabilities_valid


def test_valid_probability_vectors():
    assert probabilities_valid({"H": 0.5, "D": 0.3, "A": 0.2})
    assert not probabilities_valid({"H": 0.5, "D": 0.3, "A": 0.5})   # sums to 1.3
    assert not probabilities_valid({"H": -0.1, "D": 0.6, "A": 0.5})  # negative
    assert not probabilities_valid({"H": float("nan"), "D": 0.5, "A": 0.5})


def test_invalid_odds_rejected():
    assert not valid_decimal_odds(1.0)      # no payout
    assert not valid_decimal_odds(0.5)
    assert not valid_decimal_odds(-2.0)
    assert not valid_decimal_odds("abc")
    assert not valid_decimal_odds(float("inf"))
    assert valid_decimal_odds(1.01)
    assert valid_decimal_odds(3.5)


def test_overround_removal_sums_to_one():
    odds = [2.1, 3.5, 3.8]
    assert devig.overround(odds) > 0            # book has margin
    for method in ("normalized", "power", "shin"):
        p = devig.devig(odds, method)
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p > 0)


def test_shin_corrects_favourite_longshot_bias():
    # Shin's insider-trading model corrects the favourite-longshot bias: relative
    # to naive normalization it assigns MORE probability to the favourite and
    # LESS to the longshot. (The margin is loaded disproportionately onto the
    # longshot, so removing it fairly lifts the favourite.)
    odds = [1.2, 7.0, 15.0]                     # clear favourite + longshot
    norm = devig.devig(odds, "normalized")
    shin = devig.devig(odds, "shin")
    assert shin[0] >= norm[0] - 1e-9            # favourite: at least as high
    assert shin[-1] <= norm[-1] + 1e-9          # longshot: at least as low


def test_market_probability_vs_offered_implied():
    # offered implied still carries the margin; fair (de-vigged) is smaller.
    # Use a book with a genuine overround (booksum > 1); [2.0,4.0,4.0] has none.
    odds = [1.9, 3.8, 3.8]
    assert devig.overround(odds) > 0
    offered = offered_implied_probability(odds[0])
    fair = devig.devig(odds, "shin")[0]
    assert offered > fair                       # margin removed


def test_expected_value_matches_worked_example():
    # bet.md §6 worked example: p=0.31, odds=3.80 -> EV +17.8%
    ev = edge.raw_expected_value(0.31, 3.80)
    assert ev == pytest.approx(0.178, abs=1e-3)


def test_probability_edge_in_points():
    e = edge.probability_edge(0.31, 0.25)
    assert e == pytest.approx(0.06, abs=1e-9)   # +6 percentage POINTS


def test_conservative_ev_lower_than_raw():
    raw = edge.raw_expected_value(0.31, 3.80)
    cons = edge.conservative_expected_value(0.29, 3.80)
    assert cons < raw


def test_break_even_probability():
    assert break_even_probability(4.0) == pytest.approx(0.25)
    assert edge.break_even_probability(2.0) == pytest.approx(0.5)


def test_devig_rejects_invalid_odds():
    with pytest.raises(ValueError):
        devig.devig([1.0, 3.0, 4.0], "shin")

"""Betting: placing, balance transactions, and settlement against results."""
import pandas as pd
import pytest

from match_predict import betting
from match_predict.store import Store, STARTING_BALANCE


FX = {
    "match_id": "England-PL|2025-2026|20250815|Liverpool|Arsenal|UPCOMING",
    "league": "England-PL", "league_label": "England · Premier League",
    "date": "2025-08-15", "home_team": "Liverpool", "away_team": "Arsenal",
    "odds_h": 2.10, "odds_d": 3.40, "odds_a": 3.50,
}


def _store(tmp_path):
    return Store(str(tmp_path / "bet.db"))


def _result(fthg, ftag):
    return pd.DataFrame([{"league": FX["league"], "date": pd.Timestamp(FX["date"]),
                          "home_team": FX["home_team"], "away_team": FX["away_team"],
                          "fthg": fthg, "ftag": ftag}])


def test_outcome_of():
    assert betting.outcome_of(2, 0) == "H"
    assert betting.outcome_of(1, 1) == "D"
    assert betting.outcome_of(0, 3) == "A"


def test_bet_from_fixture_uses_feed_odds():
    b = betting.bet_from_fixture(FX, "A", 10)
    assert b["odds"] == 3.50 and b["sel_label"] == "Arsenal"


def test_bet_from_fixture_rejects_missing_odds():
    fx = dict(FX, odds_d=None)
    with pytest.raises(ValueError):
        betting.bet_from_fixture(fx, "D", 5)


def test_place_bet_debits_balance(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.bet_from_fixture(FX, "H", 20))
    assert s.wallet()["balance"] == STARTING_BALANCE - 20
    assert s.portfolio()["staked_open"] == 20
    assert s.portfolio()["equity"] == STARTING_BALANCE   # stake still "on the table"


def test_cannot_overspend(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.place_bet(betting.bet_from_fixture(FX, "H", STARTING_BALANCE + 1))


def test_settlement_win_pays_stake_times_odds(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.bet_from_fixture(FX, "H", 10))     # backing Liverpool
    summary = betting.settle_open_bets(s, _result(2, 0))   # home win
    assert summary == {"settled": 1, "won": 1, "lost": 0, "void": 0,
                       "portfolio": s.portfolio()}
    b = s.all_bets()[0]
    assert b["status"] == "won" and b["payout"] == pytest.approx(21.0)
    assert b["result"] == "2-0"
    # balance = 1000 - 10 + 21
    assert s.wallet()["balance"] == pytest.approx(STARTING_BALANCE + 11)


def test_settlement_loss_keeps_debit(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.bet_from_fixture(FX, "H", 10))
    betting.settle_open_bets(s, _result(0, 2))             # away win -> lost
    b = s.all_bets()[0]
    assert b["status"] == "lost" and b["payout"] == 0
    assert s.wallet()["balance"] == pytest.approx(STARTING_BALANCE - 10)
    p = s.portfolio()
    assert p["pnl"] == pytest.approx(-10) and p["win_rate"] == 0.0


def test_unplayed_match_stays_open(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.bet_from_fixture(FX, "H", 10))
    other = pd.DataFrame([{"league": "Spain-LL", "date": pd.Timestamp("2025-08-15"),
                           "home_team": "X", "away_team": "Y", "fthg": 1, "ftag": 0}])
    summary = betting.settle_open_bets(s, other)
    assert summary["settled"] == 0
    assert s.open_bets()[0]["status"] == "open"


def test_reset_wallet(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.bet_from_fixture(FX, "H", 10))
    s.reset_wallet()
    assert s.wallet()["balance"] == STARTING_BALANCE and s.all_bets() == []


# --------------------------------------------------------------------------- #
# Multi-market pricing + settlement                                            #
# --------------------------------------------------------------------------- #
FX_OU = dict(FX, odds_over25=1.90, odds_under25=1.95)

PRED = {
    "prob_home": 0.45, "prob_draw": 0.28, "prob_away": 0.27,
    "over_under": {"1.5": 0.75, "2.5": 0.55},
    "btts_yes": 0.60,
    "top_scores": [{"score": "2-1", "prob": 0.10}, {"score": "1-1", "prob": 0.09}],
}


def test_ou_2_5_uses_feed_odds():
    b = betting.build_bet(FX_OU, PRED, "OU", "O2.5", 10)
    assert b["market"] == "OU" and b["priced_by"] == "feed"
    assert b["odds"] == 1.90 and b["model_prob"] == pytest.approx(0.55)


def test_ou_other_line_is_model_priced():
    b = betting.build_bet(FX_OU, PRED, "OU", "U1.5", 10)
    assert b["priced_by"] == "model"
    # under 1.5 = 1 - 0.75 = 0.25 -> fair 1/0.25 = 4.0
    assert b["odds"] == pytest.approx(4.0) and b["model_prob"] == pytest.approx(0.25)


def test_btts_is_model_priced_fair_odds():
    b = betting.build_bet(FX_OU, PRED, "BTTS", "YES", 5)
    assert b["priced_by"] == "model" and b["model_prob"] == pytest.approx(0.60)
    assert b["odds"] == pytest.approx(round(1 / 0.60, 2))


def test_cs_is_model_priced_from_top_scores():
    b = betting.build_bet(FX_OU, PRED, "CS", "2-1", 5)
    assert b["selection"] == "2-1" and b["priced_by"] == "model"
    assert b["odds"] == pytest.approx(10.0)


def test_model_market_without_prediction_raises():
    with pytest.raises(ValueError):
        betting.build_bet(FX_OU, None, "BTTS", "YES", 5)


def test_settle_over_under():
    assert betting.settle_selection("OU", "O2.5", 2, 1) == "won"   # total 3
    assert betting.settle_selection("OU", "U2.5", 2, 1) == "lost"
    assert betting.settle_selection("OU", "O1.5", 1, 0) == "lost"  # total 1
    assert betting.settle_selection("OU", "U1.5", 0, 0) == "won"


def test_settle_btts():
    assert betting.settle_selection("BTTS", "YES", 1, 1) == "won"
    assert betting.settle_selection("BTTS", "YES", 2, 0) == "lost"
    assert betting.settle_selection("BTTS", "NO", 2, 0) == "won"


def test_settle_correct_score():
    assert betting.settle_selection("CS", "2-1", 2, 1) == "won"
    assert betting.settle_selection("CS", "2-1", 1, 1) == "lost"


def test_settle_open_bets_across_markets(tmp_path):
    s = _store(tmp_path)
    s.place_bet(betting.build_bet(FX_OU, PRED, "OU", "O2.5", 10))   # total 3 -> won
    s.place_bet(betting.build_bet(FX_OU, PRED, "BTTS", "YES", 10))  # 2-1 -> won
    s.place_bet(betting.build_bet(FX_OU, PRED, "CS", "1-1", 10))    # 2-1 -> lost
    summary = betting.settle_open_bets(s, _result(2, 1))
    assert summary["settled"] == 3 and summary["won"] == 2 and summary["lost"] == 1

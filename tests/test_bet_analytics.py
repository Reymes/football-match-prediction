"""Paper-bet ledger analytics: realized return kept separate from forecast
quality, with drawdown, losing runs, odds bands and calibration (bet.md §16/§18)."""
import pytest

from match_predict import bet_analytics as ba
from match_predict.store import Store


def _bet(store, *, odds, stake, model_prob=None):
    return store.place_bet({
        "match_id": f"L|2025|1|{odds}|H", "league": "England-PL",
        "home": "A", "away": "B", "selection": "H",
        "odds": odds, "stake": stake, "model_prob": model_prob,
    })["id"]


def _ledger(tmp_path):
    """Won, Lost, Lost, Won — a mixed sequence with a known equity curve."""
    s = Store(str(tmp_path / "a.db"))
    b1 = _bet(s, odds=2.0, stake=10, model_prob=0.5)
    b2 = _bet(s, odds=3.0, stake=10, model_prob=0.4)
    b3 = _bet(s, odds=4.0, stake=10, model_prob=0.3)
    b4 = _bet(s, odds=1.5, stake=10, model_prob=0.7)
    s.settle_bet(b1, won=True, result="1-0")
    s.settle_bet(b2, won=False, result="0-1")
    s.settle_bet(b3, won=False, result="0-2")
    s.settle_bet(b4, won=True, result="2-1")
    return s


def test_empty_ledger(tmp_path):
    r = ba.performance_report(Store(str(tmp_path / "e.db")).all_bets())
    assert r["n_bets_total"] == 0
    assert r["realized_return"]["roi"] is None
    assert r["forecast_quality"] is None
    assert r["meaningful_sample"] is False


def test_realized_return(tmp_path):
    r = ba.report_from_store(_ledger(tmp_path))["realized_return"]
    assert r["n_settled"] == 4 and r["won"] == 2 and r["lost"] == 2
    assert r["win_rate"] == 0.5
    assert r["staked"] == 40.0 and r["returned"] == 35.0
    assert r["pnl"] == -5.0
    assert r["roi"] == pytest.approx(-0.125)


def test_drawdown_and_losing_run(tmp_path):
    r = ba.report_from_store(_ledger(tmp_path))["realized_return"]
    # cumulative pnl: +10, 0, -10, -5 -> peak 10, trough -10 -> drawdown 20
    assert r["max_drawdown"] == pytest.approx(20.0)
    assert r["longest_losing_run"] == 2


def test_return_by_odds_band(tmp_path):
    bands = {b["band"]: b for b in ba.report_from_store(_ledger(tmp_path))["by_odds_band"]}
    assert bands["1.5-2.0"]["pnl"] == pytest.approx(5.0)     # odds 1.5 winner
    assert bands["3.0-5.0"]["n"] == 2 and bands["3.0-5.0"]["won"] == 0


def test_forecast_quality_calibration(tmp_path):
    fq = ba.report_from_store(_ledger(tmp_path))["forecast_quality"]
    assert fq["n_scored"] == 4
    # brier = mean((p-won)^2) for p=[.5,.4,.3,.7], won=[1,0,0,1]
    assert fq["brier"] == pytest.approx(0.1475)
    assert fq["avg_pred_win_prob"] == pytest.approx(0.475)
    assert fq["empirical_win_rate"] == pytest.approx(0.5)
    assert sum(row["n"] for row in fq["reliability"]) == 4


def test_forecast_quality_excludes_unscored_bets(tmp_path):
    s = Store(str(tmp_path / "u.db"))
    s.settle_bet(_bet(s, odds=2.0, stake=10, model_prob=0.6), won=True)
    s.settle_bet(_bet(s, odds=2.0, stake=10, model_prob=None), won=True)
    r = ba.report_from_store(s)
    assert r["realized_return"]["n_settled"] == 2          # both count for return
    assert r["forecast_quality"]["n_scored"] == 1          # only the scored one


def test_void_excluded_from_return(tmp_path):
    s = Store(str(tmp_path / "v.db"))
    s.settle_bet(_bet(s, odds=2.0, stake=10, model_prob=0.5), won=False, void=True)
    r = ba.performance_report(s.all_bets())
    assert r["n_void"] == 1 and r["realized_return"]["n_settled"] == 0


def test_api_bet_report_smoke():
    """The /api/bet_report route returns the report shape (read-only)."""
    try:
        import app as webapp
    except Exception:                              # noqa: BLE001 - optional web deps
        pytest.skip("app import unavailable in this environment")
    resp = webapp.app.test_client().get("/api/bet_report")
    assert resp.status_code == 200
    data = resp.get_json()
    assert {"realized_return", "forecast_quality", "meaningful_sample"} <= data.keys()

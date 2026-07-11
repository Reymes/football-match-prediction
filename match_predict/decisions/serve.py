"""Bridge from a live `MatchPrediction` to the advisory decision engine.

This is the single place that turns pipeline output into a `MatchDecision`,
so the CLI research script and the Flask app never duplicate the reconcile /
de-vig / evaluate wiring (rule: reuse `features/build.py`-style single source
of truth, here for the decision layer).

Leakage note: `feed_odds_1x2` / `feed_odds_ou25` must be the pre-kickoff market
line already carried by the fixture feed (never client-supplied, never fetched
live from a website here). The PURE view is built from the Dixon-Coles score
matrix alone and never sees bookmaker odds; only the MARKET view consumes them.
"""
from __future__ import annotations

import pandas as pd

from ..models.markets import derive_markets, score_matrix
from ..pipeline.predict import reconcile_matrix_to_1x2
from .bet_modes import evaluate_mode
from .decision_engine import build_views, evaluate_match
from .exposure import ExposureLedger
from .schema import MatchDecision, load_config


def decide_for_prediction(
    mp,
    *,
    feed_odds_1x2: list | None = None,
    feed_odds_ou25: dict | None = None,
    config: dict | None = None,
    market_profiles: dict | None = None,
    ledger=None,
    decision_time: str | None = None,
    horizon: str | None = None,
    odds_timestamp: str | None = None,
    data_quality: float = 1.0,
    default_rho: float = -0.045,
) -> MatchDecision:
    """Evaluate one `MatchPrediction` with the advisory decision engine.

    `feed_odds_1x2` is `[odds_h, odds_d, odds_a]`; `feed_odds_ou25` is
    `{"OVER": odds, "UNDER": odds}`. Both are optional — missing odds fall
    through to the engine's own MISSING_ODDS rejection (an honest "no bet"),
    they never crash this function.
    """
    config = config or load_config()

    lam = mp.model_rate_home if mp.model_rate_home is not None else mp.exp_goals_home
    mu = mp.model_rate_away if mp.model_rate_away is not None else mp.exp_goals_away
    rho = getattr(mp, "rho", None) or default_rho

    hybrid = {"H": mp.prob_home, "D": mp.prob_draw, "A": mp.prob_away}
    pure_matrix = score_matrix(lam, mu, rho=rho)
    pure_book = derive_markets(pure_matrix, lam, mu)
    pure = {"H": pure_book.p_home, "D": pure_book.p_draw, "A": pure_book.p_away}

    views = build_views(pure, hybrid, outcome_odds_1x2=feed_odds_1x2,
                        devig_method=config.get("devig_method", "shin"))

    # score-derived markets (O/U, BTTS, correct score) all read off ONE matrix
    # reconciled to the hybrid HDA — the same approach evaluate_prices.py and
    # research_decisions.py use, so every reported market stays mutually consistent.
    reconciled_matrix = reconcile_matrix_to_1x2(
        pure_matrix, [hybrid["H"], hybrid["D"], hybrid["A"]])
    book = derive_markets(reconciled_matrix, lam, mu)

    prices: dict = {}
    if feed_odds_1x2 is not None and all(o is not None for o in feed_odds_1x2):
        h, d, a = feed_odds_1x2
        prices["match_winner"] = {"H": h, "D": d, "A": a}
    if feed_odds_ou25 is not None and any(
            feed_odds_ou25.get(k) is not None for k in ("OVER", "UNDER")):
        prices["over_under_2_5"] = {k: v for k, v in feed_odds_ou25.items()
                                    if v is not None}

    fixture_id = f"{mp.league}|{mp.date}|{mp.home_team}|{mp.away_team}"
    return evaluate_match(
        fixture_id=fixture_id, league=mp.league,
        home_team=mp.home_team, away_team=mp.away_team, kickoff=mp.date,
        views=views, market_book=book, prices=prices, config=config,
        market_profiles=market_profiles, data_quality=data_quality,
        decision_time=decision_time, horizon=horizon,
        odds_timestamp=odds_timestamp, ledger=ledger)


def pure_probs_for_prediction(mp, default_rho: float = -0.045) -> dict:
    """Pure-model probability per selection key, read off the pure score matrix.

    Mirrors `candidate_selections` selection keys ("H"/"D"/"A", "OVER_2.5",
    "BTTS_YES", "2-1", …) so the bet-mode layer can check whether the PURE model
    (no bookmaker odds) independently beats the market for any market — not just
    1X2 (bet-funcuanlty §12 model-support count).
    """
    lam = mp.model_rate_home if mp.model_rate_home is not None else mp.exp_goals_home
    mu = mp.model_rate_away if mp.model_rate_away is not None else mp.exp_goals_away
    rho = getattr(mp, "rho", None) or default_rho
    book = derive_markets(score_matrix(lam, mu, rho=rho), lam, mu)
    probs = {"H": book.p_home, "D": book.p_draw, "A": book.p_away}
    ou = book.over_under.get(2.5, {})
    if ou.get("over") is not None:
        probs["OVER_2.5"] = ou["over"]
    if ou.get("under") is not None:
        probs["UNDER_2.5"] = ou["under"]
    if book.btts.get("yes") is not None:
        probs["BTTS_YES"] = book.btts["yes"]
    if book.btts.get("no") is not None:
        probs["BTTS_NO"] = book.btts["no"]
    for (i, j), pr in book.correct_score:
        probs[f"{i}-{j}"] = pr
    return probs


def _match_key(league, date, home, away) -> str:
    return f"{league}|{pd.to_datetime(date):%Y%m%d}|{home}|{away}"


def _odds_maps(fx: pd.DataFrame, odds_totals: pd.DataFrame | None):
    """Build the per-match 1X2 and O/U 2.5 feed-odds lookups (never client odds)."""
    odds_1x2 = {
        _match_key(r.league, r.date, r.home_team, r.away_team):
            [r.odds_h, r.odds_d, r.odds_a]
        for r in fx.itertuples()
        if pd.notna(getattr(r, "odds_h", None)) and pd.notna(getattr(r, "odds_d", None))
        and pd.notna(getattr(r, "odds_a", None))
    }
    ou25 = {}
    if odds_totals is not None and not odds_totals.empty:
        ou25 = {
            _match_key(r.league, r.date, r.home_team, r.away_team):
                {"OVER": r.odds_over25 if pd.notna(r.odds_over25) else None,
                 "UNDER": r.odds_under25 if pd.notna(r.odds_under25) else None}
            for r in odds_totals.itertuples()
        }
    return odds_1x2, ou25


def decide_for_fixtures(pred, fx: pd.DataFrame, *, config: dict | None = None,
                        odds_totals: pd.DataFrame | None = None,
                        market_profiles: dict | None = None,
                        odds_timestamp: str | None = None,
                        decision_time: str | None = None) -> list:
    """Score `fx` with `pred.predict_fixtures` and decide every covered match.

    Single shared path for the CLI research script and the Flask dashboard:
    never retrains, never duplicates feature/matrix/de-vig logic, and always
    reads odds from the fixture feed (`fx.odds_h/d/a`), never a client. Rows
    the model doesn't cover (unknown league) are simply skipped by
    `pred.predict_fixtures`. Returns a list of `MatchDecision`.

    `odds_timestamp` is the verifiable observed-at time of the batch fixture
    feed (its last sync / file mtime); `decision_time` is the cutoff (now).
    Passing them lets the engine judge staleness honestly instead of rejecting
    every price for a missing timestamp — the feed is a pre-kickoff snapshot, so
    its sync time is a legitimate (non-leaking) capture time (bet.md §3, §5).
    """
    config = config or load_config()
    if fx.empty:
        return []

    odds_1x2_by_key, ou25_by_key = _odds_maps(fx, odds_totals)
    ledger = ExposureLedger(config["exposure"], config["staking"])
    decisions = []
    for mp in pred.predict_fixtures(fx):
        key = _match_key(mp.league, mp.date, mp.home_team, mp.away_team)
        decisions.append(decide_for_prediction(
            mp, feed_odds_1x2=odds_1x2_by_key.get(key),
            feed_odds_ou25=ou25_by_key.get(key),
            config=config, market_profiles=market_profiles,
            ledger=ledger, default_rho=pred.default_rho,
            odds_timestamp=odds_timestamp, decision_time=decision_time))
    return decisions


def modes_for_fixtures(pred, fx: pd.DataFrame, modes, *,
                       config: dict | None = None,
                       odds_totals: pd.DataFrame | None = None,
                       market_profiles: dict | None = None,
                       odds_timestamp: str | None = None,
                       decision_time: str | None = None) -> dict:
    """Run the requested bet MODE(S) over `fx` and return `{mode: result}`.

    Builds each fixture's `MatchDecision` exactly once (same path as
    `decide_for_fixtures`), captures the PURE-model probability per selection so
    the mode layer can count independent model support, then re-thresholds under
    each mode profile. Never retrains, never fetches or accepts client odds.
    """
    config = config or load_config()
    if isinstance(modes, str):
        modes = [modes]
    if fx.empty:
        return {m: {"mode": m, "selections": [], "summary": {}} for m in modes}

    odds_1x2_by_key, ou25_by_key = _odds_maps(fx, odds_totals)
    ledger = ExposureLedger(config["exposure"], config["staking"])
    decisions, pure_by_fixture = [], {}
    for mp in pred.predict_fixtures(fx):
        key = _match_key(mp.league, mp.date, mp.home_team, mp.away_team)
        dec = decide_for_prediction(
            mp, feed_odds_1x2=odds_1x2_by_key.get(key),
            feed_odds_ou25=ou25_by_key.get(key),
            config=config, market_profiles=market_profiles,
            ledger=ledger, default_rho=pred.default_rho,
            odds_timestamp=odds_timestamp, decision_time=decision_time)
        decisions.append(dec)
        pure_by_fixture[dec.fixture_id] = pure_probs_for_prediction(
            mp, pred.default_rho)

    return {m: evaluate_mode(decisions, m, config, market_profiles,
                             pure_probs_by_fixture=pure_by_fixture)
            for m in modes}

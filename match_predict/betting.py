"""Paper-betting logic across markets: price a selection, settle it on results.

The wallet, bet rows and balance transactions live in ``store.py``; this module
holds the *domain* logic — pricing a chosen market selection and turning a
played match into a win/loss/void — and stays deliberately small.

Markets and how each is priced:

  * ``1X2``  home / draw / away         — priced at the **fixture feed's** odds.
  * ``OU``   over / under 2.5 goals     — priced at the **fixture feed's** odds.
  * ``BTTS`` both teams to score yes/no — the feed carries no line for this, so
             it is priced at the **model's own fair odds** (1/p), labelled as
             such. This is honest paper pricing, never a market quote.
  * ``CS``   exact correct score        — model-priced fair odds, as above.

Odds are always resolved *server-side* (from the feed or from our own model
probability) — never taken from the client — so a bet is priced at either the
real market line or our transparent fair line. No leverage, no partial
cash-out, no accumulators: one stake, one outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

SELECTIONS = ("H", "D", "A")   # kept for the 1X2 market / back-compat
OU_LINE = 2.5

# Which markets read a real price off the feed vs. our own fair line. Surfaced
# in the API/UI so a model-priced bet is never mistaken for a market quote.
FEED_PRICED = ("1X2", "OU")
MODEL_PRICED = ("BTTS", "CS")


def outcome_of(fthg: float, ftag: float) -> str:
    """Full-time 1X2 outcome from a final score."""
    if fthg > ftag:
        return "H"
    if fthg < ftag:
        return "A"
    return "D"


def _key(league, date, home, away) -> tuple:
    d = pd.to_datetime(date, errors="coerce")
    ds = f"{d:%Y-%m-%d}" if pd.notna(d) else str(date)
    return (str(league), ds, str(home), str(away))


def results_index(results: pd.DataFrame) -> dict:
    """Map (league, date, home, away) -> (fthg, ftag) for finished matches."""
    idx = {}
    if results is None or results.empty:
        return idx
    df = results.dropna(subset=["fthg", "ftag"])
    for r in df.itertuples():
        idx[_key(r.league, r.date, r.home_team, r.away_team)] = (
            int(r.fthg), int(r.ftag))
    return idx


# --------------------------------------------------------------------------- #
# Settlement — one small resolver per market, dispatched by ``market`` code.   #
# Each returns "won" | "lost" | "void" from a final score.                     #
# --------------------------------------------------------------------------- #
def _parse_score(sel: str) -> tuple[int, int]:
    i, j = sel.split("-")
    return int(i), int(j)


def _settle_1x2(selection: str, fthg: int, ftag: int) -> str:
    return "won" if outcome_of(fthg, ftag) == selection.upper() else "lost"


def _settle_ou(selection: str, fthg: int, ftag: int) -> str:
    side, line = selection[0].upper(), float(selection[1:])
    total = fthg + ftag
    if total == line:                      # exact push (only whole lines)
        return "void"
    over = total > line
    hit = (side == "O" and over) or (side == "U" and not over)
    return "won" if hit else "lost"


def _settle_btts(selection: str, fthg: int, ftag: int) -> str:
    both = fthg > 0 and ftag > 0
    want_yes = selection.upper().startswith("Y")
    return "won" if both == want_yes else "lost"


def _settle_cs(selection: str, fthg: int, ftag: int) -> str:
    i, j = _parse_score(selection)
    return "won" if (fthg == i and ftag == j) else "lost"


_SETTLERS: dict[str, Callable[[str, int, int], str]] = {
    "1X2": _settle_1x2, "OU": _settle_ou, "BTTS": _settle_btts, "CS": _settle_cs,
}


def settle_selection(market: str, selection: str, fthg: int, ftag: int) -> str:
    """Resolve one bet to won/lost/void. Unknown markets void (stake refunded)."""
    fn = _SETTLERS.get((market or "1X2").upper())
    if fn is None:
        return "void"
    try:
        return fn(selection, fthg, ftag)
    except (ValueError, IndexError):       # malformed selection -> refund
        return "void"


def settle_open_bets(store, results: pd.DataFrame) -> dict:
    """Settle every open bet whose match now has a result. Returns a summary."""
    idx = results_index(results)
    settled = won = lost = void = 0
    for b in store.open_bets():
        res = idx.get(_key(b["league"], b["match_date"], b["home"], b["away"]))
        if res is None:
            continue                       # not played / not synced yet
        fthg, ftag = res
        status = settle_selection(b.get("market", "1X2"), b["selection"],
                                  fthg, ftag)
        store.settle_bet(b["id"], won=(status == "won"),
                         void=(status == "void"), result=f"{fthg}-{ftag}")
        settled += 1
        won += int(status == "won")
        lost += int(status == "lost")
        void += int(status == "void")
    return {"settled": settled, "won": won, "lost": lost, "void": void,
            "portfolio": store.portfolio()}


# --------------------------------------------------------------------------- #
# Pricing — turn a fixture (+ optional model prediction) into a placeable bet. #
# --------------------------------------------------------------------------- #
def _finite_odds(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 1.0 else None


def _fair_odds(prob) -> float | None:
    """Vig-free fair price from our own probability (1/p)."""
    if prob is None:
        return None
    p = float(prob)
    return round(1.0 / p, 2) if 0.0 < p < 1.0 else None


@dataclass
class Quote:
    odds: float
    model_prob: float | None
    label: str
    priced_by: str          # "feed" | "model"


def _quote_1x2(fx: dict, pred: dict | None, sel: str) -> Quote:
    sel = sel.upper()
    if sel not in SELECTIONS:
        raise ValueError("selection must be H, D or A")
    odds = _finite_odds(fx.get({"H": "odds_h", "D": "odds_d",
                                "A": "odds_a"}[sel]))
    prob = None
    if pred:
        prob = {"H": pred.get("prob_home"), "D": pred.get("prob_draw"),
                "A": pred.get("prob_away")}[sel]
    label = {"H": fx["home_team"], "D": "Draw", "A": fx["away_team"]}[sel]
    return Quote(odds, prob, label, "feed")


def _quote_ou(fx: dict, pred: dict | None, sel: str) -> Quote:
    side, line = sel[0].upper(), float(sel[1:])
    prob = None
    if pred:
        over = (pred.get("over_under") or {}).get(f"{line:g}")
        if over is not None:
            prob = float(over) if side == "O" else 1.0 - float(over)
    # The feed only quotes the 2.5 line; other totals fall back to fair odds.
    feed = (_finite_odds(fx.get("odds_over25" if side == "O" else "odds_under25"))
            if line == OU_LINE else None)
    odds, priced_by = (feed, "feed") if feed is not None else (_fair_odds(prob), "model")
    label = f"{'Over' if side == 'O' else 'Under'} {line:g} goals"
    return Quote(odds, prob, label, priced_by)


def _quote_btts(fx: dict, pred: dict | None, sel: str) -> Quote:
    want_yes = sel.upper().startswith("Y")
    prob = None
    if pred and pred.get("btts_yes") is not None:
        y = float(pred["btts_yes"])
        prob = y if want_yes else 1.0 - y
    label = "Both teams to score" if want_yes else "Not both teams to score"
    return Quote(_fair_odds(prob), prob, label, "model")


def _quote_cs(fx: dict, pred: dict | None, sel: str) -> Quote:
    prob = None
    if pred:
        for s in pred.get("top_scores", []):
            if s.get("score") == sel:
                prob = float(s["prob"])
                break
    return Quote(_fair_odds(prob), prob, f"Correct score {sel}", "model")


_QUOTERS: dict[str, Callable[[dict, dict | None, str], Quote]] = {
    "1X2": _quote_1x2, "OU": _quote_ou, "BTTS": _quote_btts, "CS": _quote_cs,
}


def quote_selection(fx: dict, pred: dict | None, market: str, selection: str) -> Quote:
    """Price a market selection server-side. Raises ValueError if unavailable."""
    market = (market or "1X2").upper()
    quoter = _QUOTERS.get(market)
    if quoter is None:
        raise ValueError(f"unknown market '{market}'")
    q = quoter(fx, pred, selection)
    if q.odds is None:
        if market in MODEL_PRICED:
            raise ValueError("no model price yet — train a model to bet this market")
        raise ValueError("no market odds available for this selection")
    return q


def build_bet(fx: dict, pred: dict | None, market: str, selection: str,
              stake: float) -> dict:
    """Assemble a ``store.place_bet`` payload from a fixture + market selection.

    Odds come from the fixture feed (1X2, OU) or from our own model probability
    (BTTS, CS) — never from the client — so a bet is always priced at the real
    market line or our transparent fair line.
    """
    market = (market or "1X2").upper()
    q = quote_selection(fx, pred, market, selection)
    return {
        "match_id": fx["match_id"],
        "league": fx["league"],
        "league_label": fx.get("league_label"),
        "match_date": fx.get("date") or fx.get("match_date"),
        "home": fx["home_team"], "away": fx["away_team"],
        "market": market, "selection": str(selection).upper() if market != "CS"
        else str(selection),
        "sel_label": q.label, "odds": round(float(q.odds), 3),
        "stake": float(stake), "model_prob": q.model_prob,
        "priced_by": q.priced_by,
    }


# Back-compat shim: the original 1X2-only entry point still used by older tests.
_SEL_ODDS = {"H": "odds_h", "D": "odds_d", "A": "odds_a"}


def bet_from_fixture(fx_row: dict, selection: str, stake: float,
                     model_prob: float | None = None) -> dict:
    """Assemble a 1X2 store payload from a fixture row + selection (legacy)."""
    pred = None
    if model_prob is not None and selection.upper() in SELECTIONS:
        key = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}[selection.upper()]
        pred = {key: model_prob}
    bet = build_bet(fx_row, pred, "1X2", selection, stake)
    bet["model_prob"] = model_prob if model_prob is not None else bet["model_prob"]
    return bet

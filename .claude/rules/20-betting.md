# Betting rules

The betting layer is **paper-money only** and must stay that way.

- €1000 play balance (`store.py`). Markets (`betting.py`): `1X2`, `OU`
  (over/under, any line), `BTTS`, `CS` (correct score). AH & team totals are
  displayed but not bettable (settlement complexity).
- Odds are **resolved server-side, never from the client.** `1X2` and O/U 2.5
  are priced at the **feed's** real line; markets the feed does not quote
  (`BTTS`, `CS`, non-2.5 O/U lines) are priced at the model's own **fair odds**
  (1/p) and clearly labelled "fair" in the UI — a transparent paper price, never
  presented as a bookmaker quote.
- Auto-settles against real results on each data sync (stake×odds on win,
  stake returned on void). `settle_selection()` dispatches per market.
- **Forbidden:** real bet placement, bookmaker-account integration, staking
  systems, martingale, loss-chasing, doubling, accumulator-as-leverage.
- Never describe a selection as a guaranteed win or a certain score.

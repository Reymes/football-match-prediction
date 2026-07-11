"""Tiny SQLite store: sync audit log + fixture-prediction cache.

Everything the platform needs is still derivable from the CSVs and the trained
bundle; this DB is a *convenience layer* so the web UI is fast and informative:

  * ``sync_runs``     — one row per data/fixtures sync, for the "last updated"
                        panel and a short history.
  * ``fixture_cache`` — a scored fixture keyed by (trained_through, match_id).
                        Predicting a full slate rebuilds features over the whole
                        history, so we cache the JSON result and invalidate it
                        automatically whenever the model is retrained (the
                        ``trained_through`` stamp changes).

Single file on the project root (``matchpredict.db`` by default). Pure stdlib.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

DEFAULT_DB = os.environ.get("MATCHPREDICT_DB", "matchpredict.db")
STARTING_BALANCE = float(os.environ.get("MATCHPREDICT_BALANCE", "1000"))
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(path: str):
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class Store:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self._init()

    def _init(self):
        with _LOCK, _connect(self.path) as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    ok       INTEGER NOT NULL DEFAULT 1,
                    files    INTEGER DEFAULT 0,
                    fixtures INTEGER DEFAULT 0,
                    seasons  INTEGER DEFAULT 0,
                    detail   TEXT
                );
                CREATE TABLE IF NOT EXISTS fixture_cache (
                    trained_through TEXT NOT NULL,
                    match_id        TEXT NOT NULL,
                    payload         TEXT NOT NULL,
                    ts              TEXT NOT NULL,
                    PRIMARY KEY (trained_through, match_id)
                );
                CREATE TABLE IF NOT EXISTS wallet (
                    id       INTEGER PRIMARY KEY CHECK (id = 1),
                    balance  REAL NOT NULL,
                    starting REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bets (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    placed_at  TEXT NOT NULL,
                    match_id   TEXT NOT NULL,
                    league     TEXT NOT NULL,
                    league_label TEXT,
                    match_date TEXT,
                    home       TEXT NOT NULL,
                    away       TEXT NOT NULL,
                    market     TEXT NOT NULL DEFAULT '1X2',
                    selection  TEXT NOT NULL,          -- H | D | A
                    sel_label  TEXT,
                    odds       REAL NOT NULL,
                    stake      REAL NOT NULL,
                    model_prob REAL,                   -- our prob at bet time (if known)
                    status     TEXT NOT NULL DEFAULT 'open',  -- open|won|lost|void
                    payout     REAL DEFAULT 0,
                    result     TEXT,                   -- final score, e.g. '2-1'
                    settled_at TEXT
                );
                """
            )
            row = c.execute("SELECT balance FROM wallet WHERE id=1").fetchone()
            if row is None:
                c.execute("INSERT INTO wallet (id, balance, starting) VALUES (1,?,?)",
                          (STARTING_BALANCE, STARTING_BALANCE))

    # ---------------------------------------------------------------- sync log
    def record_sync(self, kind: str, *, ok: bool = True, files: int = 0,
                    fixtures: int = 0, seasons: int = 0, detail: dict | None = None):
        with _LOCK, _connect(self.path) as c:
            c.execute(
                "INSERT INTO sync_runs (ts, kind, ok, files, fixtures, seasons, detail)"
                " VALUES (?,?,?,?,?,?,?)",
                (_now(), kind, int(ok), files, fixtures, seasons,
                 json.dumps(detail or {})),
            )

    def last_sync(self, kind: str | None = None) -> dict | None:
        q = "SELECT * FROM sync_runs"
        args: tuple = ()
        if kind:
            q += " WHERE kind=?"
            args = (kind,)
        q += " ORDER BY id DESC LIMIT 1"
        with _connect(self.path) as c:
            row = c.execute(q, args).fetchone()
        return _row_to_sync(row) if row else None

    def recent_syncs(self, limit: int = 12) -> list[dict]:
        with _connect(self.path) as c:
            rows = c.execute(
                "SELECT * FROM sync_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_sync(r) for r in rows]

    # ------------------------------------------------------------ pred cache
    def get_predictions(self, trained_through: str, match_ids: list[str]) -> dict:
        if not match_ids:
            return {}
        out = {}
        with _connect(self.path) as c:
            # chunk to stay under SQLite's variable limit
            for i in range(0, len(match_ids), 400):
                chunk = match_ids[i:i + 400]
                marks = ",".join("?" * len(chunk))
                rows = c.execute(
                    f"SELECT match_id, payload FROM fixture_cache "
                    f"WHERE trained_through=? AND match_id IN ({marks})",
                    (trained_through, *chunk),
                ).fetchall()
                for r in rows:
                    out[r["match_id"]] = json.loads(r["payload"])
        return out

    def put_predictions(self, trained_through: str, preds: list[dict]):
        if not preds:
            return
        with _LOCK, _connect(self.path) as c:
            c.executemany(
                "INSERT OR REPLACE INTO fixture_cache "
                "(trained_through, match_id, payload, ts) VALUES (?,?,?,?)",
                [(trained_through, p["match_id"], json.dumps(p), _now())
                 for p in preds if p.get("match_id")],
            )

    def clear_cache(self, trained_through: str | None = None):
        with _LOCK, _connect(self.path) as c:
            if trained_through:
                c.execute("DELETE FROM fixture_cache WHERE trained_through=?",
                          (trained_through,))
            else:
                c.execute("DELETE FROM fixture_cache")

    # ------------------------------------------------------------- wallet
    def wallet(self) -> dict:
        with _connect(self.path) as c:
            row = c.execute("SELECT balance, starting FROM wallet WHERE id=1").fetchone()
        return {"balance": round(row["balance"], 2),
                "starting": round(row["starting"], 2)}

    def _adjust_balance(self, c, delta: float):
        c.execute("UPDATE wallet SET balance = balance + ? WHERE id=1", (delta,))

    def reset_wallet(self, starting: float | None = None):
        """Wipe all bets and restore the opening balance."""
        s = STARTING_BALANCE if starting is None else float(starting)
        with _LOCK, _connect(self.path) as c:
            c.execute("DELETE FROM bets")
            c.execute("UPDATE wallet SET balance=?, starting=? WHERE id=1", (s, s))

    # --------------------------------------------------------------- bets
    def place_bet(self, bet: dict) -> dict:
        """Insert an open bet and debit the stake. Raises ValueError if broke.

        ``bet`` must carry: match_id, league, home, away, selection, odds,
        stake (and optionally league_label, match_date, sel_label, model_prob).
        """
        stake = float(bet["stake"])
        odds = float(bet["odds"])
        if stake <= 0:
            raise ValueError("stake must be positive")
        if odds <= 1.0:
            raise ValueError("odds must be greater than 1.0")
        with _LOCK, _connect(self.path) as c:
            bal = c.execute("SELECT balance FROM wallet WHERE id=1").fetchone()["balance"]
            if stake > bal + 1e-9:
                raise ValueError(f"insufficient balance (€{bal:.2f} available)")
            cur = c.execute(
                "INSERT INTO bets (placed_at, match_id, league, league_label, "
                "match_date, home, away, market, selection, sel_label, odds, "
                "stake, model_prob, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'open')",
                (_now(), bet["match_id"], bet["league"], bet.get("league_label"),
                 bet.get("match_date"), bet["home"], bet["away"],
                 bet.get("market", "1X2"), bet["selection"], bet.get("sel_label"),
                 odds, stake, bet.get("model_prob")),
            )
            self._adjust_balance(c, -stake)
            bet_id = cur.lastrowid
        return self.get_bet(bet_id)

    def get_bet(self, bet_id: int) -> dict | None:
        with _connect(self.path) as c:
            row = c.execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()
        return dict(row) if row else None

    def open_bets(self) -> list[dict]:
        with _connect(self.path) as c:
            rows = c.execute("SELECT * FROM bets WHERE status='open' "
                             "ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def all_bets(self, limit: int = 200) -> list[dict]:
        with _connect(self.path) as c:
            rows = c.execute("SELECT * FROM bets ORDER BY "
                             "CASE status WHEN 'open' THEN 0 ELSE 1 END, id DESC "
                             "LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def settle_bet(self, bet_id: int, *, won: bool, void: bool = False,
                   result: str | None = None):
        """Mark a bet won/lost/void and credit any return atomically."""
        with _LOCK, _connect(self.path) as c:
            b = c.execute("SELECT * FROM bets WHERE id=? AND status='open'",
                          (bet_id,)).fetchone()
            if b is None:
                return
            if void:
                status, payout = "void", b["stake"]
            elif won:
                status, payout = "won", b["stake"] * b["odds"]
            else:
                status, payout = "lost", 0.0
            c.execute("UPDATE bets SET status=?, payout=?, result=?, settled_at=? "
                      "WHERE id=?", (status, payout, result, _now(), bet_id))
            if payout:
                self._adjust_balance(c, payout)

    def portfolio(self) -> dict:
        w = self.wallet()
        with _connect(self.path) as c:
            rows = c.execute("SELECT status, stake, payout FROM bets").fetchall()
        n_open = staked_open = 0
        settled = won = staked_settled = returned = 0
        for r in rows:
            if r["status"] == "open":
                n_open += 1
                staked_open += r["stake"]
            elif r["status"] != "void":
                settled += 1
                staked_settled += r["stake"]
                returned += r["payout"]
                won += 1 if r["status"] == "won" else 0
        pnl = returned - staked_settled
        return {
            **w,
            "equity": round(w["balance"] + staked_open, 2),
            "open_bets": n_open, "staked_open": round(staked_open, 2),
            "settled_bets": settled, "won": won,
            "win_rate": round(won / settled, 3) if settled else None,
            "staked_settled": round(staked_settled, 2),
            "returned": round(returned, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / staked_settled, 3) if staked_settled else None,
        }


def _row_to_sync(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["detail"] = json.loads(d.get("detail") or "{}")
    except Exception:  # noqa: BLE001
        d["detail"] = {}
    d["ok"] = bool(d.get("ok", 1))
    return d

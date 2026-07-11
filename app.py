#!/usr/bin/env python
"""Flask web UI for the football match predictor — everything from the browser.

  * Predict     — single match (pick league/teams/date + optional odds) or CSV upload.
  * Models      — dashboard of available models, the data they used, and their
                  metrics; buttons to RETRAIN the serving bundle and to run the
                  honest walk-forward EVALUATION, both as live background jobs.

The trained Predictor bundle in artifacts/ is loaded at startup (if present) and
hot-reloaded in place whenever a retrain finishes — no server restart needed.

Run:  python app.py   →  http://127.0.0.1:5001
"""
from __future__ import annotations

import io
import os
import threading
import traceback
import warnings
from collections import deque
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

warnings.filterwarnings("ignore")

from match_predict.pipeline import Predictor
from match_predict.pipeline.training import (
    train_and_save, evaluate_walk_forward, load_model_card,
)
from match_predict.data import (
    sync_all, sync_latest, sync_fixtures, parse_fixtures, parse_fixture_totals_odds,
)
from match_predict.pipeline.predictor import infer_season
from match_predict.store import Store
from match_predict.viz import team_badge_svg, league_badge_svg, slugify
from match_predict import betting
from match_predict.bet_analytics import report_from_store
from match_predict.data import load_all
from match_predict.decisions import (
    decide_for_fixtures, load_config, load_profiles, modes_for_fixtures,
    summarize_day)

ARTIFACTS = os.environ.get("ARTIFACTS", "artifacts")
TRAIN_CACHE = os.environ.get("TRAIN_CACHE")  # optional feature-cache path
DATA_ROOT = os.environ.get("DATA_ROOT", "football-data")
DATA_CACHE = os.environ.get("DATA_CACHE", ".data-cache")
FIXTURES_PATH = os.path.join(DATA_ROOT, "fixtures.csv")
LOGO_DIR = os.path.join("static", "logos")   # optional real crests
STORE = Store()
# Display names for every league label the ingest layer can emit
# (see match_predict/data/schema.py:LEAGUE_BY_DIV). Any league found in the
# data but missing here still shows up in the UI under its raw label, so new
# divisions added upstream degrade gracefully instead of disappearing.
LEAGUE_LABELS = {
    "England-PL": "England · Premier League",
    "England-Champ": "England · Championship",
    "England-L1": "England · League One",
    "England-L2": "England · League Two",
    "England-NL": "England · National League",
    "Scotland-PR": "Scotland · Premiership",
    "Scotland-Champ": "Scotland · Championship",
    "Scotland-L1": "Scotland · League One",
    "Scotland-L2": "Scotland · League Two",
    "Germany-BL": "Germany · Bundesliga",
    "Germany-BL2": "Germany · 2. Bundesliga",
    "Italy-SA": "Italy · Serie A",
    "Italy-SB": "Italy · Serie B",
    "Spain-LL": "Spain · La Liga",
    "Spain-LL2": "Spain · La Liga 2",
    "France-L1": "France · Ligue 1",
    "France-L2": "France · Ligue 2",
    "Netherlands-ED": "Netherlands · Eredivisie",
    "Belgium-PL": "Belgium · Pro League",
    "Portugal-PL": "Portugal · Primeira Liga",
    "Turkey-SL": "Turkey · Süper Lig",
    "Greece-SL": "Greece · Super League",
}


def _label(league: str) -> str:
    """Pretty display name for a league label, falling back to the raw code."""
    return LEAGUE_LABELS.get(league, league)


def _available_leagues(pred) -> dict:
    """Ordered {league: pretty label} for leagues actually present in the model.

    Driven by the trained history rather than the static catalog, so any league
    added to the data (a new season file, a new division) appears automatically
    once the model is retrained — no code change required.
    """
    present = sorted(pd.unique(pred.history["league"].dropna()).tolist())
    return {lg: _label(lg) for lg in present}

app = Flask(__name__)

# ---- live model state (hot-reloadable) --------------------------------------
_STATE_LOCK = threading.Lock()
PREDICTOR: Predictor | None = None
TEAMS: dict = {}


def _teams_by_league(pred) -> dict:
    h = pred.history
    recent = h.sort_values("date").groupby("league").tail(760)  # ~2 seasons
    out = {}
    for lg in _available_leagues(pred):
        teams = pd.unique(pd.concat([
            recent.loc[recent.league == lg, "home_team"],
            recent.loc[recent.league == lg, "away_team"]]).dropna())
        out[lg] = sorted(map(str, teams))
    return out


def load_predictor() -> bool:
    """(Re)load the bundle into the global state. Returns True on success."""
    global PREDICTOR, TEAMS
    if not os.path.exists(os.path.join(ARTIFACTS, "models.joblib")):
        return False
    pred = Predictor.load(ARTIFACTS)
    teams = _teams_by_league(pred)
    with _STATE_LOCK:
        PREDICTOR, TEAMS = pred, teams
    return True


load_predictor()  # ok if it fails; the UI will offer a Train button


# ---- background job manager (one job at a time) -----------------------------
class JobManager:
    def __init__(self, maxlog=400):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.kind: str | None = None
        self.state = "idle"          # idle | running | done | error
        self.log = deque(maxlen=maxlog)
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self.pct: int | None = None      # 0..100 for a determinate bar; None = spinner

    def is_running(self) -> bool:
        return self.state == "running"

    def _progress(self, msg: str, pct: int | None = None):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {msg}")
        if pct is not None:
            self.pct = max(0, min(100, int(pct)))

    def start(self, kind: str, target) -> bool:
        with self.lock:
            if self.is_running():
                return False
            self.kind, self.state = kind, "running"
            self.error = None
            self.log.clear()
            self.pct = None
            self.started_at = datetime.now().isoformat(timespec="seconds")
            self.finished_at = None
        self.thread = threading.Thread(target=self._run, args=(target,), daemon=True)
        self.thread.start()
        return True

    def _run(self, target):
        try:
            target(self._progress)
            self.state = "done"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self._progress(f"ERROR: {self.error}")
            self._progress(traceback.format_exc().splitlines()[-1])
            self.state = "error"
        finally:
            self.finished_at = datetime.now().isoformat(timespec="seconds")

    def status(self) -> dict:
        return {
            "kind": self.kind, "state": self.state,
            "running": self.is_running(), "pct": self.pct,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "error": self.error, "log": list(self.log),
        }


JOBS = JobManager()


# ---- serialization ----------------------------------------------------------
def serialize(p) -> dict:
    ou = {str(k): round(v["over"], 4) for k, v in p.over_under.items()}
    ah = {str(k): {"home": round(v["home"], 4), "push": round(v["push"], 4),
                   "away": round(v["away"], 4)} for k, v in p.asian_handicap.items()}
    tt = {side: {str(k): round(v["over"], 4) for k, v in d.items()}
          for side, d in p.team_totals.items()}
    used_market = any("Market implies" in r for r in p.reasons)
    return {
        "league": p.league, "league_label": _label(p.league),
        "date": p.date, "home_team": p.home_team, "away_team": p.away_team,
        "prob_home": round(p.prob_home, 4), "prob_draw": round(p.prob_draw, 4),
        "prob_away": round(p.prob_away, 4),
        "xg_home": round(p.exp_goals_home, 2), "xg_away": round(p.exp_goals_away, 2),
        "model_rate_home": (round(p.model_rate_home, 2)
                            if p.model_rate_home is not None else None),
        "model_rate_away": (round(p.model_rate_away, 2)
                            if p.model_rate_away is not None else None),
        "top_scores": [{"score": f"{i}-{j}", "prob": round(pr, 4)}
                       for (i, j), pr in p.top_scores[:12]],
        "btts_yes": round(p.btts.get("yes", 0), 4),
        "over_under": ou, "asian_handicap": ah, "team_totals": tt,
        "confidence": round(p.confidence, 3), "uncertainty": round(p.uncertainty, 3),
        "reasons": p.reasons, "used_market": used_market,
    }


def _predict_df(fixtures: pd.DataFrame):
    with _STATE_LOCK:
        pred = PREDICTOR
    if pred is None:
        raise RuntimeError("no trained model loaded — train one from the Models tab")
    return [serialize(p) for p in pred.predict_fixtures(fixtures)]


# ---- logos ------------------------------------------------------------------
def _real_logo(kind: str, key: str) -> str | None:
    """Return a static URL if a real crest file was fetched, else None."""
    for ext in (".svg", ".png", ".webp"):
        rel = os.path.join(LOGO_DIR, kind, key + ext)
        if os.path.exists(rel):
            return "/" + rel.replace(os.sep, "/")
    return None


def _team_logo_url(name: str) -> str:
    key = slugify(name)
    return _real_logo("team", key) or f"/logo/team/{key}?name={_q(name)}"


def _league_logo_url(league: str) -> str:
    key = slugify(league)
    return _real_logo("league", key) or \
        f"/logo/league/{key}?league={_q(league)}&label={_q(_label(league))}"


def _q(s: str) -> str:
    from urllib.parse import quote
    return quote(str(s))


# ---- fixtures (upcoming matches) + prediction cache -------------------------
def _fixture_key(league, date, home, away) -> str:
    d = pd.to_datetime(date)
    return f"{league}|{infer_season(d)}|{d:%Y%m%d}|{home}|{away}|UPCOMING"


def _decorate(pred: dict) -> dict:
    """Attach a stable key and logo URLs to a serialized prediction/fixture."""
    pred = dict(pred)
    pred["match_id"] = _fixture_key(pred["league"], pred["date"],
                                    pred["home_team"], pred["away_team"])
    pred["home_logo"] = _team_logo_url(pred["home_team"])
    pred["away_logo"] = _team_logo_url(pred["away_team"])
    pred["league_logo"] = _league_logo_url(pred["league"])
    return pred


def load_fixtures_df() -> pd.DataFrame:
    return parse_fixtures(FIXTURES_PATH)


def _ou_odds_index() -> dict:
    """Map fixture key -> {odds_over25, odds_under25} from the feed's O/U 2.5."""
    tot = parse_fixture_totals_odds(FIXTURES_PATH)
    idx = {}
    for r in tot.itertuples():
        key = _fixture_key(r.league, r.date, r.home_team, r.away_team)
        idx[key] = {"odds_over25": _num(r.odds_over25),
                    "odds_under25": _num(r.odds_under25)}
    return idx


def fixture_row_by_key(match_id: str) -> dict | None:
    """Look up a single upcoming fixture (with feed odds) by its match key.

    1X2 and O/U 2.5 odds always come from the feed here, so a placed bet is
    priced at the real market line regardless of what a client sends.
    """
    fx = load_fixtures_df()
    ou = _ou_odds_index()
    for r in fx.itertuples():
        key = _fixture_key(r.league, r.date, r.home_team, r.away_team)
        if key == match_id:
            return {"match_id": key, "league": r.league,
                    "league_label": _label(r.league),
                    "date": f"{pd.to_datetime(r.date):%Y-%m-%d}",
                    "home_team": r.home_team, "away_team": r.away_team,
                    "odds_h": _num(r.odds_h), "odds_d": _num(r.odds_d),
                    "odds_a": _num(r.odds_a), **ou.get(key, {})}
    return None


def predict_fixtures_cached(fx: pd.DataFrame, use_cache: bool = True) -> dict:
    """Score a set of upcoming fixtures, reusing cached results per model.

    Returns predictable rows (with probabilities + logos) plus a list of
    fixtures whose league the current model does not cover.
    """
    with _STATE_LOCK:
        pred = PREDICTOR
        tt = pred.trained_through if pred else ""
    if pred is None:
        raise RuntimeError("no trained model loaded — train one from the Models tab")
    if fx.empty:
        return {"predictions": [], "uncovered": [], "trained_through": tt}

    covered_leagues = set(pred.dc_models)
    fx = fx.copy()
    fx["key"] = [_fixture_key(r.league, r.date, r.home_team, r.away_team)
                 for r in fx.itertuples()]
    # Live feed odds by key — attached fresh to every row (never cached, so a
    # re-sync's updated prices always win) and used for betting.
    ou = _ou_odds_index()
    odds_by_key = {r.key: {"odds_h": _num(r.odds_h), "odds_d": _num(r.odds_d),
                           "odds_a": _num(r.odds_a), **ou.get(r.key, {})}
                   for r in fx.itertuples()}
    coverable = fx[fx["league"].isin(covered_leagues)]
    uncovered = fx[~fx["league"].isin(covered_leagues)]

    cached = STORE.get_predictions(tt, coverable["key"].tolist()) if use_cache else {}
    missing = coverable[~coverable["key"].isin(cached)]

    fresh: list[dict] = []
    if not missing.empty:
        cols = ["league", "date", "home_team", "away_team",
                "odds_h", "odds_d", "odds_a"]
        for p in pred.predict_fixtures(missing[cols]):
            fresh.append(_decorate(serialize(p)))
        STORE.put_predictions(tt, fresh)

    by_key = {**cached, **{p["match_id"]: p for p in fresh}}
    preds = []
    for k in coverable["key"]:
        if k in by_key:
            preds.append({**by_key[k], **odds_by_key.get(k, {})})
    unc = [{"league": r.league, "league_label": _label(r.league),
            "league_logo": _league_logo_url(r.league),
            "date": f"{pd.to_datetime(r.date):%Y-%m-%d}",
            "match_id": r.key,
            "home_team": r.home_team, "away_team": r.away_team,
            "home_logo": _team_logo_url(r.home_team),
            "away_logo": _team_logo_url(r.away_team),
            **odds_by_key.get(r.key, {})}
           for r in uncovered.itertuples()]
    return {"predictions": preds, "uncovered": unc, "trained_through": tt}


def _num(v):
    try:
        f = float(v)
        return round(f, 3) if f == f else None   # NaN -> None
    except (TypeError, ValueError):
        return None


# ---- data status ------------------------------------------------------------
def _data_status() -> dict:
    import glob
    files = glob.glob(os.path.join(DATA_ROOT, "*", "*.csv"))
    seasons, leagues = set(), set()
    for f in files:
        base = os.path.basename(f)
        if "_" in base:
            leagues.add(base.rsplit("_", 1)[0])
            seasons.add(base.rsplit("_", 1)[1].replace(".csv", ""))
    fx = load_fixtures_df()
    fx_dates = pd.to_datetime(fx["date"], errors="coerce").dropna() if len(fx) else pd.Series([], dtype="datetime64[ns]")
    return {
        "n_files": len(files), "n_leagues": len(leagues),
        "n_seasons": len(seasons),
        "latest_season": (max(seasons) if seasons else None),
        "fixtures": int(len(fx)),
        "fixtures_leagues": sorted(fx["league"].dropna().unique().tolist()) if len(fx) else [],
        "fixtures_from": (f"{fx_dates.min():%Y-%m-%d}" if len(fx_dates) else None),
        "fixtures_to": (f"{fx_dates.max():%Y-%m-%d}" if len(fx_dates) else None),
        "last_sync": STORE.last_sync(),
        "recent_syncs": STORE.recent_syncs(6),
    }


# ---- routes -----------------------------------------------------------------
@app.route("/")
def index():
    with _STATE_LOCK:
        ready = PREDICTOR is not None
        trained_through = PREDICTOR.trained_through if ready else ""
        teams = TEAMS
        leagues = _available_leagues(PREDICTOR) if ready else LEAGUE_LABELS
    return render_template("index.html", leagues=leagues, teams=teams,
                           trained_through=trained_through, model_ready=ready)


@app.route("/api/health")
def api_health():
    with _STATE_LOCK:
        ready = PREDICTOR is not None
        trained_through = PREDICTOR.trained_through if ready else ""
    return jsonify({"model_ready": ready, "trained_through": trained_through,
                    "job": JOBS.status()["state"]})


@app.route("/api/teams")
def api_teams():
    with _STATE_LOCK:
        return jsonify(TEAMS)


@app.route("/api/models")
def api_models():
    """Model card + live bundle status for the dashboard."""
    card = load_model_card(ARTIFACTS)
    with _STATE_LOCK:
        ready = PREDICTOR is not None
        trained_through = PREDICTOR.trained_through if ready else ""
        n_history = int(len(PREDICTOR.history)) if ready else 0
        leagues = sorted(PREDICTOR.dc_models) if ready else []
    return jsonify({
        "model_ready": ready, "trained_through": trained_through,
        "n_history": n_history, "leagues": leagues,
        "league_labels": LEAGUE_LABELS, "card": card,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    d = request.get_json(force=True)
    try:
        row = {"league": d["league"], "date": d["date"],
               "home_team": d["home_team"], "away_team": d["away_team"]}
    except KeyError as e:
        return jsonify({"error": f"missing field {e}"}), 400
    if d.get("home_team") == d.get("away_team"):
        return jsonify({"error": "home and away team must differ"}), 400
    for k in ("odds_h", "odds_d", "odds_a"):
        v = d.get(k)
        row[k] = float(v) if v not in (None, "", "null") else None
    try:
        preds = _predict_df(pd.DataFrame([row]))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    if not preds:
        return jsonify({"error": "could not predict (unknown league?)"}), 400
    return jsonify(preds[0])


@app.route("/api/predict_csv", methods=["POST"])
def api_predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files["file"]
    try:
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8-sig")))
    except Exception as e:
        return jsonify({"error": f"could not read CSV: {e}"}), 400
    required = {"league", "date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        return jsonify({"error": f"CSV missing columns: {sorted(missing)}"}), 400
    try:
        preds = _predict_df(df)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 400
    return jsonify({"n_input": len(df), "n_predicted": len(preds),
                    "predictions": preds})


# ---- logos ------------------------------------------------------------------
@app.route("/logo/team/<key>")
def logo_team(key):
    name = request.args.get("name", key)
    return Response(team_badge_svg(name), mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/logo/league/<key>")
def logo_league(key):
    league = request.args.get("league", key)
    label = request.args.get("label")
    return Response(league_badge_svg(league, label), mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


# ---- data status + fixtures -------------------------------------------------
@app.route("/api/data/status")
def api_data_status():
    return jsonify(_data_status())


@app.route("/api/fixtures")
def api_fixtures():
    """Upcoming fixtures, scored where the model has the league."""
    league = request.args.get("league")
    fx = load_fixtures_df()
    if league:
        fx = fx[fx["league"] == league]
    if PREDICTOR is None:
        # No model yet: still list the fixtures (with odds) so users can bet.
        ou = _ou_odds_index()
        rows = []
        for r in fx.itertuples():
            row = {"league": r.league, "league_label": _label(r.league),
                   "date": f"{pd.to_datetime(r.date):%Y-%m-%d}",
                   "home_team": r.home_team, "away_team": r.away_team,
                   "odds_h": _num(r.odds_h), "odds_d": _num(r.odds_d),
                   "odds_a": _num(r.odds_a)}
            dec = _decorate(row)
            dec.update(ou.get(dec["match_id"], {}))
            rows.append(dec)
        return jsonify({"predictions": [], "uncovered": rows,
                        "trained_through": "", "n_fixtures": int(len(fx))})
    try:
        result = predict_fixtures_cached(fx)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    result["n_fixtures"] = int(len(fx))
    return jsonify(result)


@app.route("/api/decisions")
def api_decisions():
    """Advisory research view — Value Scan tab (bet.md §2, §22, §23).

    Reuses the live Predictor + fixture feed odds; never retrains, never
    accepts client-supplied odds, never places a bet. "No bet" is a normal,
    expected result — most selections will show NO_BET without a captured
    odds timestamp (the honest, conservative default).
    """
    league = request.args.get("league")
    with _STATE_LOCK:
        pred = PREDICTOR
    if pred is None:
        return jsonify({"error": "no trained model loaded — train one from "
                                 "the Models tab"}), 409
    fx = load_fixtures_df()
    if league:
        fx = fx[fx["league"] == league]
    if fx.empty:
        return jsonify({"decisions": [], "summary": summarize_day([]),
                        "n_fixtures": 0})

    cfg = load_config()
    ou25 = parse_fixture_totals_odds(FIXTURES_PATH)
    odds_ts = _feed_odds_timestamp()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    profiles = _market_profiles()
    decisions = decide_for_fixtures(pred, fx, config=cfg, odds_totals=ou25,
                                    market_profiles=profiles,
                                    odds_timestamp=odds_ts, decision_time=now)
    return jsonify({
        "decisions": [d.to_dict() for d in decisions],
        "summary": summarize_day(decisions),
        "n_fixtures": int(len(fx)),
        "odds_timestamp": odds_ts,
        "markets_validated": sorted(m for m, p in profiles.items()
                                    if p.passed_quality),
    })


_BET_MODE_ALIASES = {"smart": "smart", "high-return": "high_return",
                     "high_return": "high_return"}


@app.route("/api/bet-modes")
def api_bet_modes():
    """Smart Bet / High Return Bet decision modes (bet-funcuanlty §3, §13).

    `?mode=smart|high-return|compare` (default compare). Re-thresholds the SAME
    trained-model outputs used by the Value Scan; never retrains, never accepts
    client odds, never places a bet. "No bet" is a normal result and neither
    mode is forced to pick a selection. High-return selections usually LOSE more
    often than they win — the value is in the price, not the hit rate.
    """
    raw_mode = (request.args.get("mode") or "compare").lower()
    if raw_mode == "compare":
        modes = ["smart", "high_return"]
    elif raw_mode in _BET_MODE_ALIASES:
        modes = [_BET_MODE_ALIASES[raw_mode]]
    else:
        return jsonify({"error": f"unknown bet mode '{raw_mode}'"}), 400

    league = request.args.get("league")
    with _STATE_LOCK:
        pred = PREDICTOR
    if pred is None:
        return jsonify({"error": "no trained model loaded — train one from "
                                 "the Models tab"}), 409
    fx = load_fixtures_df()
    if league:
        fx = fx[fx["league"] == league]

    cfg = load_config()
    profiles = _market_profiles()
    if fx.empty:
        results = {m: {"mode": m, "selections": [], "summary": {}} for m in modes}
    else:
        ou25 = parse_fixture_totals_odds(FIXTURES_PATH)
        odds_ts = _feed_odds_timestamp()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = modes_for_fixtures(
            pred, fx, modes, config=cfg, odds_totals=ou25,
            market_profiles=profiles, odds_timestamp=odds_ts, decision_time=now)

    return jsonify({
        "compare": raw_mode == "compare",
        "modes": {m: {"summary": r["summary"],
                      "selections": [s.to_dict() for s in r["selections"]]}
                  for m, r in results.items()},
        "n_fixtures": int(len(fx)),
        "markets_validated": sorted(m for m, p in profiles.items()
                                    if p.passed_quality),
    })


MARKET_PROFILES_PATH = os.path.join("artifacts", "market_profiles.json")


def _market_profiles() -> dict:
    """Load persisted out-of-time validation profiles (empty if none built yet).

    Read fresh each call so `scripts/build_market_profiles.py` takes effect
    without a server restart; the file is tiny. No profile → the engine safely
    rejects every market as unvalidated (INSUFFICIENT_HISTORICAL_SAMPLE).
    """
    return load_profiles(MARKET_PROFILES_PATH)


def _feed_odds_timestamp() -> str | None:
    """Verifiable observed-at time of the batch fixture-odds feed.

    The football-data feed carries no per-row odds timestamp, but it is a
    pre-kickoff snapshot captured at sync time — a legitimate, non-leaking
    capture time. Prefer the recorded fixtures-sync time, else the file mtime.
    """
    last = STORE.last_sync("fixtures") or STORE.last_sync()
    if last and last.get("ts"):
        return last["ts"]
    try:
        mtime = os.path.getmtime(FIXTURES_PATH)
        return datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


@app.route("/api/bet_report")
def api_bet_report():
    """Honest paper-bet ledger analytics (bet.md §16/§18) — advisory only.

    Realized return (P&L, ROI, drawdown, longest losing run, return by odds
    band) is reported separately from forecast quality (calibration of the
    model probabilities actually staked). Nothing here places a bet.
    """
    return jsonify(report_from_store(STORE))


# ---- betting ----------------------------------------------------------------
def _settle_now(progress=None) -> dict:
    """Settle open bets against the freshest results on disk."""
    if progress:
        progress("settling open bets against latest results …", None)
    results = load_all(DATA_ROOT)
    summary = betting.settle_open_bets(STORE, results)
    if progress:
        progress(f"settled {summary['settled']} bets "
                 f"({summary['won']} won, {summary['lost']} lost)", None)
    return summary


@app.route("/api/wallet")
def api_wallet():
    return jsonify(STORE.portfolio())


@app.route("/api/wallet/reset", methods=["POST"])
def api_wallet_reset():
    STORE.reset_wallet()
    return jsonify(STORE.portfolio())


@app.route("/api/bets")
def api_bets():
    return jsonify({"bets": STORE.all_bets(), "wallet": STORE.portfolio()})


def _prediction_for(match_id: str) -> dict | None:
    """Serialized prediction for one fixture (for model-priced odds / prob)."""
    with _STATE_LOCK:
        pred = PREDICTOR
    if pred is None:
        return None
    fx = load_fixtures_df()
    if fx.empty:
        return None
    fx = fx.copy()
    fx["key"] = [_fixture_key(r.league, r.date, r.home_team, r.away_team)
                 for r in fx.itertuples()]
    one = fx[fx["key"] == match_id]
    if one.empty:
        return None
    preds = predict_fixtures_cached(one).get("predictions") or []
    return preds[0] if preds else None


@app.route("/api/bet", methods=["POST"])
def api_place_bet():
    d = request.get_json(force=True) or {}
    try:
        match_id = d["match_id"]
        market = str(d.get("market", "1X2"))
        selection = str(d["selection"])
        stake = float(d["stake"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "need match_id, market, selection, stake"}), 400
    fx = fixture_row_by_key(match_id)
    if fx is None:
        return jsonify({"error": "fixture not found — sync fixtures first"}), 404
    # Odds are resolved server-side (feed for 1X2/OU, model fair for BTTS/CS).
    prediction = _prediction_for(match_id)
    try:
        payload = betting.build_bet(fx, prediction, market, selection, stake)
        bet = STORE.place_bet(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"bet": bet, "wallet": STORE.portfolio()})


@app.route("/api/bets/settle", methods=["POST"])
def api_settle():
    def job(progress):
        _settle_now(progress)
        progress("settlement complete.", 100)
    if not JOBS.start("settle", job):
        return jsonify({"error": f"a {JOBS.kind} job is already running"}), 409
    return jsonify({"started": True, "kind": "settle"})


# ---- sync jobs --------------------------------------------------------------
def _sync_job(kind: str):
    def job(progress):
        if kind == "all":
            info = sync_all(DATA_ROOT, DATA_CACHE, progress)
            STORE.record_sync("all", files=info["files"],
                              seasons=len(info["seasons"]), detail=info)
        elif kind == "latest":
            info = sync_latest(DATA_ROOT, DATA_CACHE, progress)
            STORE.record_sync("latest", files=info["files"],
                              seasons=len(info["seasons"]), detail=info)
        else:  # fixtures (also refreshes the fixtures' season)
            info = sync_fixtures(DATA_ROOT, DATA_CACHE, progress)
            sub = info.get("season") or {}
            STORE.record_sync("fixtures", ok=not info.get("error"),
                              fixtures=info.get("fixtures", 0),
                              files=(sub.get("files", 0) if sub else 0),
                              detail=info)
        # Fresh results just landed — settle any open bets they resolve.
        try:
            _settle_now(progress)
        except Exception as e:              # noqa: BLE001 — never fail a sync on this
            progress(f"(bet settlement skipped: {type(e).__name__})", None)
        progress("sync complete.", 100)
    return job


@app.route("/api/sync", methods=["POST"])
def api_sync():
    d = request.get_json(silent=True) or {}
    kind = d.get("mode", "latest")
    if kind not in ("all", "latest", "fixtures"):
        return jsonify({"error": "mode must be all|latest|fixtures"}), 400
    if not JOBS.start(f"sync:{kind}", _sync_job(kind)):
        return jsonify({"error": f"a {JOBS.kind} job is already running"}), 409
    return jsonify({"started": True, "kind": f"sync:{kind}"})


# ---- training / evaluation jobs ---------------------------------------------
@app.route("/api/train", methods=["POST"])
def api_train():
    d = request.get_json(silent=True) or {}
    val_start = d.get("val_start", "2024-08-01")
    val_end = d.get("val_end", "2025-08-01")
    dc_window = int(d.get("dc_window_days", 900))

    def job(progress):
        train_and_save(out=ARTIFACTS, val_start=val_start, val_end=val_end,
                       dc_window_days=dc_window, cache=TRAIN_CACHE,
                       progress=progress)
        progress("hot-reloading served model …")
        load_predictor()
        STORE.clear_cache()          # new model -> old fixture scores are stale
        progress("model reloaded — new predictions are live.")

    if not JOBS.start("train", job):
        return jsonify({"error": f"a {JOBS.kind} job is already running"}), 409
    return jsonify({"started": True, "kind": "train"})


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    d = request.get_json(silent=True) or {}
    val_start = d.get("val_start", "2024-08-01")
    test_start = d.get("test_start", "2025-08-01")

    def job(progress):
        evaluate_walk_forward(out=ARTIFACTS, val_start=val_start,
                              test_start=test_start, cache=TRAIN_CACHE,
                              progress=progress)

    if not JOBS.start("evaluate", job):
        return jsonify({"error": f"a {JOBS.kind} job is already running"}), 409
    return jsonify({"started": True, "kind": "evaluate"})


@app.route("/api/job/status")
def api_job_status():
    return jsonify(JOBS.status())


if __name__ == "__main__":
    # Port 5000 is taken by macOS AirPlay Receiver (ControlCenter); default to 5001.
    port = int(os.environ.get("PORT", "5001"))
    # threaded=True so a long training job doesn't block status polling / predicts.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

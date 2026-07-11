# Football Match Prediction System — Architecture

**Purpose.** A modular, production-oriented system that turns historical match
and odds data into **well-calibrated probability distributions** for every major
pre-match betting market. The design goal (from `task.md`) is *not* to maximise
apparent accuracy — it is to produce probabilities that are statistically
honest and that hold up against a strong bookmaker baseline over long,
chronological out-of-time periods.

> **What is built vs. designed.** This repository ships a **runnable baseline**
> that implements the statistical core end-to-end on the included data
> (ingestion → features → Dixon-Coles + Elo + LightGBM → stacked ensemble →
> calibration → walk-forward evaluation → explained per-match output). Sections
> marked _(design)_ describe the production extensions (extra data sources,
> orchestration, registry, monitoring) with the exact integration points the
> code already leaves open. Every claim about the baseline is backed by the
> numbers in [`reports/`](reports/) and reproducible with the scripts below.

---

## 0. Headline result (out-of-time, 2025/26 season, 2,058 matches, 6 leagues)

| model | log-loss | brier | RPS | ECE | acc |
|---|---|---|---|---|---|
| **market** (de-vigged bookmaker) | **0.9693** | 0.5764 | 0.1943 | 0.0337 | 0.538 |
| dixon_coles | 0.9991 | 0.5961 | 0.2032 | 0.0283 | 0.512 |
| gbm (LightGBM) | 0.9730 | 0.5787 | 0.1945 | 0.0337 | 0.533 |
| ensemble (stacked) | 0.9716 | 0.5785 | 0.1951 | 0.0250 | 0.532 |
| **ensemble + calibration** | 0.9717 | 0.5786 | 0.1951 | **0.0245** | 0.532 |

**Reading of the result.** The pre-match market line is an exceptionally strong
forecaster. On this test season it has the **lowest (best) log-loss (0.9693)**;
the ensemble's is 0.9717 — **~0.0024 nats worse**, not a match. The ensemble's
top-label ECE is numerically lower (0.025 vs 0.034), but neither difference is
backed by a paired bootstrap confidence interval or significance test, so we do
**not** claim the ensemble beats, matches, or is better calibrated than the
market. Honest reading: without post-match information the ensemble is
*competitive with, but does not outperform,* the bookmaker line on the primary
proper scoring rule. The reported ensemble *influence* figures
(market 0.56 / GBM 0.31 / Dixon-Coles 0.13) are **normalised magnitudes of the
logistic meta-learner's coefficients, not literal mixture weights** — they show
the meta-learner anchoring on the market, but do not sum-to-one as probabilities
over base models.

> **Caveat (not yet implemented):** paired bootstrap CIs and a Diebold–Mariano
> test on the log-loss differences. Until those exist, treat all model-vs-market
> orderings on this table as *numerical only*, not statistically established.

---

## 1. Complete system architecture

```
                     ┌──────────────────────────────────────────────┐
                     │                DATA SOURCES                   │
   Football-Data ────┤  results · match stats · bookmaker odds       │
   FBref/Understat ..│  xG, shots, PPDA          (design)            │
   ClubElo/TM ......│  ratings, squad value      (design)           │
   Weather/Injury ..│  context enrichment        (design)           │
                     └───────────────┬──────────────────────────────┘
                                     ▼
                 ┌────────────────────────────────────┐
                 │  INGESTION  (data/ingest.py)        │  encoding + date
                 │  → canonical match schema           │  normalisation
                 └───────────────┬────────────────────┘
                                 ▼
                 ┌────────────────────────────────────┐
                 │  VALIDATION (data/validation.py)    │  hard invariants +
                 │  → pass/fail + quality report       │  quality warnings
                 └───────────────┬────────────────────┘
                                 ▼
                 ┌────────────────────────────────────┐
                 │  FEATURE ENGINEERING (features/)    │  ALL shifted to
                 │  Elo · form/EWMA · context · market │  pre-kickoff only
                 └───────────────┬────────────────────┘
                                 ▼
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
 ┌─────────────┐        ┌─────────────────┐       ┌────────────────┐
 │ Dixon-Coles │        │ LightGBM 1X2    │       │ Market (de-vig)│
 │ goal model  │        │ (Elo+form+mkt)  │       │ baseline       │
 │ → λ_h, λ_a  │        │ → P(H,D,A)      │       │ → P(H,D,A)     │
 │ → score mat │        └────────┬────────┘       └───────┬────────┘
 └──────┬──────┘                 │                         │
        │                        ▼                         │
        │           ┌────────────────────────────┐        │
        │           │ STACKED ENSEMBLE (ensemble/)│◄───────┘
        │           │ multinomial logit on logits │
        │           └────────────┬───────────────┘
        │                        ▼
        │           ┌────────────────────────────┐
        │           │ CALIBRATION (calibration/)  │  temperature / isotonic
        │           │ → calibrated P(H,D,A)       │  ECE, reliability
        │           └────────────┬───────────────┘
        ▼                        ▼
 ┌────────────────────────────────────────────────┐
 │ RECONCILE + MARKET DERIVATION (pipeline/,       │  one coherent joint
 │ models/markets.py): score matrix ⋈ ensemble 1X2 │  distribution →
 │ → CS · O/U · BTTS · AH · team totals            │  all markets
 └────────────────────┬───────────────────────────┘
                      ▼
 ┌────────────────────────────────────────────────┐
 │ OUTPUT + EXPLANATION (pipeline/predict.py)      │  SHAP + reasons +
 │ 1X2, xG, top scores, confidence, uncertainty    │  confidence/uncertainty
 └────────────────────────────────────────────────┘

 EVALUATION (evaluation/): walk-forward, out-of-time, vs market baseline
```

The modules are independent and communicate through plain pandas frames / numpy
arrays, so any stage can be replaced (e.g. swap LightGBM for CatBoost, or add a
Bayesian hierarchical goal model) without touching the others.

---

## 2. Folder structure

```
Match-predict/
├── football-data/            # training data: 6 leagues × ~30 seasons (1993–2025)
│   ├── england/ france/ germany/ italy/ spain/ portugal/
├── testing/                  # out-of-time hold-out: 2025/26 season, all leagues
├── match_predict/            # the package
│   ├── data/
│   │   ├── schema.py         # canonical columns + raw→canonical mapping
│   │   ├── ingest.py         # encoding/date-robust loader → tidy frame
│   │   └── validation.py     # data-quality report (errors + warnings)
│   ├── features/
│   │   ├── elo.py            # dynamic goal-aware Elo engine
│   │   ├── form.py           # rolling + EWMA form (shifted = leak-free)
│   │   ├── context.py        # rest, congestion, season phase, derby hook
│   │   └── build.py          # assembles FEATURE_COLUMNS + market-implied probs
│   ├── models/
│   │   ├── dixon_coles.py    # time-weighted Dixon-Coles MLE
│   │   ├── markets.py        # score matrix → every betting market
│   │   ├── gbm.py            # LightGBM 1X2 + SHAP
│   │   └── baselines.py      # market + base-rate baselines
│   ├── ensemble/
│   │   └── stacker.py        # stacked meta-learner (logit-space)
│   ├── calibration/
│   │   └── calibrate.py      # temperature/isotonic + ECE + reliability
│   ├── evaluation/
│   │   ├── metrics.py        # log-loss, Brier, RPS, Poisson deviance
│   │   └── backtest.py       # walk-forward, out-of-time engine
│   └── pipeline/
│       └── predict.py        # reconcile + full explained output
├── scripts/
│   ├── run_backtest.py       # scorecard vs market baseline
│   ├── predict_matches.py    # explained predictions for real fixtures
│   └── make_report.py        # writes reports/ artifacts
├── reports/                  # scorecard.csv, reliability.png, example_predictions.json
├── requirements.txt
├── ARCHITECTURE.md           # this document
└── README.md
```

---

## 3. Database schema

The baseline runs file-based (CSV in, pandas out). For production the same
canonical model maps directly to a relational store (Postgres/TimescaleDB) or a
columnar warehouse (DuckDB/BigQuery). Core tables:

```sql
-- Dimension: teams (stable ids across seasons; handles renames)
CREATE TABLE team (
    team_id     SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    league      TEXT NOT NULL,
    aliases     TEXT[],                     -- name canonicalisation
    city        TEXT, latitude FLOAT, longitude FLOAT   -- travel/derby features
);

-- Fact: one row per match (mirrors CANONICAL_COLUMNS in data/schema.py)
CREATE TABLE match (
    match_id    TEXT PRIMARY KEY,           -- league|season|date|home|away
    league      TEXT, season TEXT,
    kickoff     TIMESTAMPTZ,                -- date + time
    home_id     INT REFERENCES team, away_id INT REFERENCES team,
    fthg SMALLINT, ftag SMALLINT, ftr CHAR(1),
    hthg SMALLINT, htag SMALLINT,
    referee     TEXT,
    hs INT, "as" INT, hst INT, ast INT, hc INT, ac INT,
    hy INT, ay INT, hr INT, ar INT,
    UNIQUE (league, season, kickoff, home_id, away_id)
);

-- Odds snapshots (pre-match + closing; timestamped for movement features)
CREATE TABLE odds (
    match_id TEXT REFERENCES match, book TEXT, captured_at TIMESTAMPTZ,
    market TEXT,                            -- '1x2','ou25','ah'
    odds_h FLOAT, odds_d FLOAT, odds_a FLOAT,
    line FLOAT, odds_over FLOAT, odds_under FLOAT,
    is_closing BOOLEAN,
    PRIMARY KEY (match_id, book, market, captured_at)
);

-- Precomputed features (immutable, keyed by match — reproducibility)
CREATE TABLE feature_row (
    match_id TEXT REFERENCES match, feature_set_version TEXT,
    payload  JSONB,                         -- FEATURE_COLUMNS values
    PRIMARY KEY (match_id, feature_set_version)
);

-- Predictions (versioned; enables monitoring/drift/backfills)
CREATE TABLE prediction (
    match_id TEXT, model_version TEXT, created_at TIMESTAMPTZ,
    p_home FLOAT, p_draw FLOAT, p_away FLOAT,
    lam_home FLOAT, lam_away FLOAT,
    market_book JSONB,                      -- CS/OU/BTTS/AH/team totals
    confidence FLOAT, uncertainty FLOAT, reasons JSONB,
    PRIMARY KEY (match_id, model_version)
);
```

Design points: (a) odds are **timestamped** so line-movement features and the
pre-match/closing split are explicit; (b) `feature_row` and `prediction` are
**versioned and immutable**, which is what makes backfills, audits, and drift
comparisons possible.

---

## 4. Data sources & their predictive value

| source | in baseline | contribution |
|---|---|---|
| **Football-Data.co.uk** | ✅ | results, half-time, shots/cards/corners, bookmaker odds — the backbone; odds give the strong baseline |
| **ClubElo** | via internal Elo | continuous team-strength signal; strongest single outcome predictor |
| **FBref** | _(design)_ | possession-adjusted stats, PPDA, progressive actions — style/quality |
| **Understat / StatsBomb** | _(design)_ | shot-level **xG/xGA** — de-noises "goals" into repeatable chance quality; feeds the goal-rate model directly |
| **Transfermarkt** | _(design)_ | squad market value & manager tenure — priors for promoted/rebuilt teams Elo can't yet see |
| **Bookmaker odds** | ✅ | market consensus; both a feature and the benchmark; **movement** = sharp money |
| **Weather API** | _(design)_ | wind/rain depress goals & passing; small but real O/U signal |
| **Injury / lineup providers** | _(design)_ | availability of key players (esp. GK, top scorer) — large variance reducer near kickoff |

Each design source plugs in behind the ingestion layer as an extra enrichment
join keyed by `(match_id)` or `(team_id, date)`; nothing downstream changes
except new entries in `FEATURE_COLUMNS`.

---

## 5. Feature engineering pipeline (leakage-safe by construction)

Order: **Elo → form → context → market**. Implemented in `features/`.

* **Team strength** — dynamic goal-difference-aware **Elo** with home advantage
  and per-season regression to the mean (`elo.py`). Attack/defence strengths
  come separately from the Dixon-Coles fit. _(design: league-strength coefficient,
  squad value, manager rating.)_
* **Form** — rolling (window 6) and **EWMA** (half-life 5) of goals, shots,
  shots-on-target, corners, points; home/away splits; a momentum term
  (last-3 vs prior-3). (`form.py`)
* **Match context** — rest days, 14-day fixture congestion, season progress
  (title/relegation-pressure proxy), derby hook. (`context.py`)
* **Market signal** — de-vigged 1X2 probabilities from bookmaker odds
  (`build.market_implied_probs`).

### Leakage prevention — the non-negotiable part

The system enforces "**only information available before kickoff**" at three
levels:

1. **Rolling features are shifted by one match** within each team's own
   timeline (`form.py` uses `shift(1)` before every `rolling`/`ewm`). A match
   never enters its own rolling statistic — the classic football-model bug.
2. **Elo attaches the pre-match rating** (the value *before* the update step in
   `elo.py`), so the label of the current game can't leak into its own feature.
3. **Chronology at the split level** — the backtester only ever `fit`s on rows
   with `date < boundary`; there are **no random train/test splits anywhere**.
   In-match statistics (final shots, cards) are used *only* as lagged inputs to
   future matches, never for the current one. Closing odds are pre-kickoff but
   flagged as leakage-sensitive (`schema.py`): the baseline evaluates on
   pre-match odds so offline metrics aren't inflated by a line that, live, is
   only available seconds before kick.

---

## 6. Models — and why several instead of one

| model | role | strengths | weaknesses |
|---|---|---|---|
| **Dixon-Coles** (built) | goal-rate + joint score | correct low-score dependence (ρ), time-decay, yields *all* markets from one coherent matrix | assumes (near-)Poisson, ignores non-goal context |
| **Bivariate Poisson** _(design)_ | goal-rate | models score correlation explicitly | heavier to fit; DC already captures most of the gain |
| **Poisson regression** _(design)_ | goal-rate | interpretable, fast, GLM baseline | linear link, no interactions |
| **LightGBM (multiclass / Poisson obj.)** (built) | 1X2 / goals | non-linear interactions (form×congestion, market-vs-Elo disagreement), native NaN handling | can over-rely on odds; needs monotonic/regularisation care |
| **CatBoost / XGBoost** _(design)_ | 1X2 / goals | categorical handling (referee, team) / regularisation diversity | ensemble diversity, marginal individually |
| **Bayesian hierarchical** _(design)_ | team strength | full posterior → honest uncertainty, shrinkage for small samples | compute; retraining cadence |

Specialised models beat one universal model here because the targets are
different objects: a **goal-rate** process (Poisson family, gives the score
matrix) and an **outcome** process (discriminative, gives sharp 1X2). The
ensemble reconciles them.

### Correct-score handling (as mandated)

We never classify exact scores. We predict `λ_home`, `λ_away`, build **one**
Dixon-Coles joint matrix `P(i,j)` for 0-0…10-10, and **derive every market**
(1X2, O/U, BTTS, correct score, Asian handicap, team totals) from it
(`models/markets.py`). `pipeline/predict.py` then rescales that matrix so its
1X2 marginals match the calibrated ensemble headline — so the reported scoreline
distribution and the reported 1X2 are mutually consistent.

---

## 7. Ensemble

`ensemble/stacker.py` — a **stacked generalisation**: base models emit
`P(H,D,A)`; the meta-learner is a multinomial logistic regression on the
**log-probabilities** of the bases (log-space ⇒ a product-of-experts / geometric
blend, better behaved than arithmetic averaging). Weights are **learned on
out-of-time validation predictions**, never hand-set (a requirement). Concepts:

* **Stacking** — meta-model on base predictions (what we do; validation is a
  rolling out-of-time window, the time-series-correct analogue of out-of-fold).
* **Blending** — single hold-out variant of the above.
* **Weighted averaging** — the constrained special case (non-negative weights
  summing to 1); subsumed by the logistic meta-learner, which can also recalibrate.
* **Meta-learner** — regularised (L2) so the blend doesn't overfit one base on a
  small validation set. The per-base figures this run
  (market 0.56 / GBM 0.31 / DC 0.13) are **normalised mean absolute coefficient
  magnitudes across the three output classes — a relative *influence* summary,
  not literal mixture weights.** The logistic stacker does not decompose into a
  convex combination of base probabilities, so these must not be read as
  "56% market + 31% GBM + 13% DC". A true constrained non-negative blend
  (weights that do sum to 1) is a documented alternative in `stacker.py`.

---

## 8. Probability calibration

`calibration/calibrate.py`:

* **Temperature scaling** (multiclass Platt) — one scalar `T`; softens/sharpens
  without changing the argmax. This run `T≈0.99` (already near-calibrated).
* **Isotonic regression** (per-class one-vs-rest, renormalised) — more flexible,
  fixes non-monotone miscalibration; needs more data.
* **Reliability diagrams** — `reports/reliability.png`.
* **Expected Calibration Error (ECE)** — the headline calibration number; the
  calibrated ensemble reaches **0.0245**, better than the market's 0.0337.

Calibrators are fit on validation only, then frozen for test — same discipline
as the models.

---

## 9. Validation framework

`evaluation/backtest.py` — strictly chronological:

```
[ ........ train ........ | .... validation .... | ...... test ...... ]
                        val_start              test_start
```

* **Walk-forward validation** — Dixon-Coles refits on a rolling trailing window
  as time advances (`dc_refit_days`); the GBM refits at each period boundary.
* **Rolling retraining** — every fit sees only `date < boundary`.
* **Out-of-time testing** — the 2025/26 season is a pure hold-out never touched
  during training/tuning.
* **No random splits** — anywhere.

## 10. Evaluation metrics

`evaluation/metrics.py`: **log-loss**, **Brier**, **Ranked Probability Score**
(respects H>D>A ordinality — the standard 1X2 metric), **Expected Calibration
Error**, **Poisson deviance** (goal-rate goodness-of-fit; ~1.13/1.12 home/away
this run), ROC-AUC-ready, and top-k correct-score coverage via the score matrix.
Accuracy is reported but is **never** the optimisation target.

---

## 11. Explainability

Every prediction ships an explanation (`pipeline/predict.py`):

* **Per-match SHAP** from LightGBM's exact tree `pred_contrib` for the predicted
  class (e.g. `mkt_prob_h +0.32, elo_diff +0.06 …`).
* **Global feature importance** (gain) — market probs and Elo dominate, as
  expected.
* **Plain-language reasons** — Elo gap, form differential, market lean, rest.
* **Confidence** (top-class probability) and **uncertainty** (normalised
  entropy of the 1X2 distribution).

---

## 12. Production pipeline _(design, with existing hooks)_

```
   cron / Airflow / Prefect DAG (daily + intraday near kickoff)
   ┌───────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────┐
   │ ingest &  │→ │ validate   │→ │ build        │→ │ predict & │
   │ enrich    │  │ (fail-fast)│  │ features     │  │ publish   │
   └───────────┘  └────────────┘  └──────────────┘  └───────────┘
        │               │                │                 │
   raw store       data-quality     feature_row        prediction
   (S3/DB)          alerts          (versioned)         (versioned)

   weekly: retrain → evaluate on rolling OOT → gate on log-loss/ECE
           → register (MLflow) → shadow → promote → (rollback on drift)
```

* **Scheduled updates & feature generation** — DAG stages map 1:1 to the modules
  above.
* **Daily/weekly retraining** — walk-forward retrain; **promotion gated** on
  out-of-time log-loss + ECE not regressing vs the incumbent and vs the market.
* **Versioning & registry** — feature-set version + model version stamped on
  every `prediction` row; MLflow model registry.
* **Experiment tracking** — MLflow/W&B for runs, params, metrics, artifacts.
* **Monitoring & drift** — track live log-loss/ECE vs market; PSI on feature
  distributions and on predicted-probability distributions; alert on
  calibration decay.
* **Rollback** — registry keeps the last-good model; a drift/regression alert
  flips the serving pointer back atomically.
* **Logging** — structured logs per stage keyed by `match_id` + `model_version`.

---

## 13. Technology stack

* **Language/data:** Python 3.12, pandas, numpy, scipy.
* **Models:** scikit-learn (meta-learner, isotonic), LightGBM; _(design)_
  CatBoost/XGBoost, statsmodels/PyMC (Bayesian), SHAP.
* **Storage:** CSV/parquet now; _(design)_ Postgres/TimescaleDB or DuckDB/BigQuery.
* **Orchestration:** _(design)_ Airflow/Prefect; MLflow (tracking + registry).
* **Serving/monitoring:** _(design)_ FastAPI, Evidently/custom PSI, Grafana.
* **Plots:** matplotlib (reliability diagram).

---

## 14. Model interaction diagram

See §1. Summary of the data contract between stages:

```
matches(df) → features(df + FEATURE_COLUMNS)
            → {DC: (λ_h, λ_a, score_matrix, P_dc),
               GBM: P_gbm,
               market: P_mkt}
            → ensemble.meta(logit P_*) → P_ens
            → calibrate(P_ens) → P_final
            → reconcile(score_matrix, P_final) → coherent market book
            → MatchPrediction(+SHAP, +reasons, +confidence/uncertainty)
```

---

## 15. Output format

Per match (`MatchPrediction`, see `reports/example_predictions.json`): W/D/L
probabilities, expected goals, **top-5 scorelines**, BTTS, Over/Under ladder,
Asian-handicap ladder (with push), team totals, **confidence**, **uncertainty**,
**key reasons**, **feature importance**. Example:

```
── Tottenham vs Burnley (England-PL, 2025-08-16) ──
  1X2      : Home 64.8% | Draw 24.6% | Away 10.7%
  Exp goals: 3.00 - 1.52
  Top scores: 2-2 8.7%  1-1 8.0%  2-1 7.2%  3-1 7.2%  3-2 5.4%
  BTTS yes : 75.8%   O/U 2.5 over: 80.6%
  AH home -0.5: 64.8%   confidence: 0.65  (uncertainty 0.79)
  Why: Tottenham materially stronger on Elo (gap +108); market 69/20/11;
       top SHAP drivers mkt_prob_h +0.32, elo_diff +0.06 …
```

---

## 16. Future improvements

1. **Shot-level xG ingestion** (Understat/StatsBomb) → replace goals with
   repeatable chance quality in the Dixon-Coles rate; biggest expected gain.
2. **Bayesian hierarchical goal model** for honest posterior uncertainty and
   shrinkage on promoted/small-sample teams.
3. **Bivariate Poisson / Weibull-count** alternatives to DC as extra ensemble members.
4. **Lineup/injury feeds near kickoff** — a second, later prediction pass.
5. **Line-movement features** (timestamped odds) to capture sharp money.
6. **Shin/​power de-vig** instead of basic normalisation for a cleaner market prob.
7. **Per-league / per-market calibration** and Dirichlet (full-matrix) calibration.
8. **Value/betting layer** — compare model vs price, Kelly staking, CLV tracking
   as the true economic metric.
9. **Monotonic constraints & feature-store** for production robustness and reuse.
```
```

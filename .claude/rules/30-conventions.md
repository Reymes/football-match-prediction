# Project conventions

- **League labels** from `match_predict/data/schema.py:LEAGUE_BY_DIV`
  (e.g. `England-PL`, `France-L1`, `Germany-BL`). Pretty display names in
  `app.py:LEAGUE_LABELS`; unknown divisions fall back to the raw code.
- **Team names** use Football-Data spelling exactly (`Man City`,
  `Nott'm Forest`, `Paris SG`). No alias resolver exists.
- **Data layout:** history at `football-data/<country>/<League>_YYYY-YYYY.csv`;
  hold-out flat in `testing/`; fixtures at `football-data/fixtures.csv`;
  cached season zips in `.data-cache/`; bundle in `artifacts/`.
- **match_id** = `league|season|YYYYMMDD|home|away` (`|UPCOMING` for fixtures).
- **Seasons** roll over in July (`infer_season`, `season_code_to_name`).
- **Odds preference:** B365 → Avg → BW → WH → PS (`ODDS_PREFERENCE`).
- **Caching:** fixture predictions are keyed by `trained_through`; always
  `STORE.clear_cache()` after retraining (already wired in `/api/train`).
- **Flask app** defaults to port **5001** (5000 is macOS AirPlay).
- **Testing:** `pip install -e ".[dev]"` then `pytest -q` (87 tests, no
  network). Run after every change.

## Spec files vs reality
`task.md` and `bet.md` are aspirational backlog specs — much of what they
describe (an `upcoming/` module, Streamlit dashboard, team-alias resolver,
`decisions/` engine, `config/*.yml`) is NOT built. `ARCHITECTURE.md` and
`README.md` describe the actual system. Treat `task.md`/`bet.md` as future work.

# Python best-practice & style rules

Follow these consistently on every change so the codebase stays uniform. Match
the existing house style already used across `match_predict/`.

## Formatting
- **PEP 8**, 4-space indentation, no tabs.
- Line length ~88 chars (the repo already wraps around this). Wrap long calls
  and comprehensions rather than exceeding it.
- One blank line between methods, two between top-level defs/classes.
- Use `from __future__ import annotations` at the top of each module (the repo
  already does this) so annotations are lazy and `X | None` works everywhere.
- Prefer f-strings over `%`/`.format()`. Use `"…"` double quotes as the default.
- Imports grouped stdlib → third-party → local, each block alphabetised; local
  imports use explicit relative form inside the package (`from ..data import …`).
- Format with **black** and lint with **ruff** if available; do not hand-fight
  the formatter.

## Naming
- **Files / modules:** short, lowercase, no dashes (`dixon_coles.py`,
  `calibrate.py`, `predictor.py`). Tests are `test_<subject>.py`.
- **Packages / directories:** lowercase (`data`, `features`, `models`,
  `ensemble`, `pipeline`).
- **Functions / variables:** `snake_case` (`build_feature_frame`,
  `match_id`, `trained_through`).
- **Classes / dataclasses:** `PascalCase` (`DixonColes`, `MatchPrediction`,
  `WalkForwardBacktest`, `Predictor`).
- **Constants / module-level config:** `UPPER_SNAKE_CASE`
  (`CANONICAL_COLUMNS`, `FEATURE_COLUMNS`, `LEAGUE_BY_DIV`, `SEASON_CODES`).
- **Private helpers:** leading underscore (`_fetch`, `_parse_dates`,
  `_noop`, `_row_to_sync`).
- Names must be descriptive; avoid single letters except short-lived math
  indices (`i`, `j`) and conventional stats symbols (`lam`, `mu`, `rho`).

## Types & docstrings
- Type-annotate public function signatures and dataclass fields
  (`def load_all(*roots: str) -> pd.DataFrame:`).
- Every module and public function/class gets a concise docstring saying what
  it does and any leakage/ordering assumption it relies on.
- Prefer `@dataclass` for value objects (`ValidationReport`, `MatchPrediction`,
  `Predictor`).

## File size & structure
- **Keep files focused and small — aim for < ~400 lines per module, hard ceiling
  ~600.** If a module grows past that, split it (the package is already split by
  responsibility: `data/`, `features/`, `models/`, `ensemble/`, etc.).
- One clear responsibility per module; functions stay small and
  single-purpose. If a function exceeds ~50 lines, extract helpers.
- Modules stay independent and communicate via plain pandas frames / numpy
  arrays so any stage can be swapped without touching the others.

## Practices
- No hidden state or global mutation except the documented singletons
  (`PREDICTOR`, `STORE`, `JOBS`) guarded by locks.
- Catch narrow exceptions; where a broad catch is genuinely needed, keep it
  local and annotate it (`# noqa: BLE001`) with a reason, as the repo does.
- Never fill a required feature with an undocumented constant; fail loudly on
  structural problems, warn on tolerable quality issues.
- Pure stdlib where practical (the sync/store layers avoid extra HTTP/DB deps).
- Determinism: seed any randomness; feature/column order must be stable.
- Add or update tests with every change and run `pytest -q` before finishing.

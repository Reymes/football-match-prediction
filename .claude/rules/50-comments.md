# Comments — keep the code self-explanatory

- **Prefer self-documenting code over comments.** Clear names, small functions,
  and obvious structure should make the intent readable without commentary.
- **Do not add comments that restate what the code already says.** No
  line-by-line narration, no obvious `# increment counter` noise.
- **Only comment when genuinely needed** — a non-obvious *why*: a subtle
  leakage guard, a numerical trick, a workaround for an upstream quirk, an
  assumption a reader could not infer from the code.
- **When a comment is needed, keep it short** — one line where possible.
- Docstrings are the exception: public modules, functions, and classes still
  get a concise docstring (that is documentation, not a comment).
- If you feel the urge to explain a block with a comment, first try to make the
  code clearer (rename, extract a well-named helper) instead.

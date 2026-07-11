"""Correlation controls (bet.md §13).

Selections from the same match are frequently strongly correlated, e.g.:
  * home win  &  home -0.5 Asian handicap;
  * over 2.5  &  BTTS yes;
  * home win  &  home team over 1.5;
  * correct score 2-1  &  BTTS yes.

We must not count correlated positions as independent opportunities. The engine
groups a match's selections into correlation groups and normally keeps at most
one PRIMARY position per match (the strongest by conservative EV). Correlations
across the same team / competition / information source are also tracked at the
portfolio level.

This is deliberately rule-based (no fitted correlation matrix): the goal is
conservative de-duplication, not precise joint modelling. Accumulators are never
manufactured to inflate returns.
"""
from __future__ import annotations


# Directional "lean" of a selection: HOME-ish, AWAY-ish, OVER-ish, UNDER-ish,
# BTTS-yes/no. Selections sharing a lean within a match are correlated.
def _lean(dec) -> str:
    m, s = dec.market, dec.selection.upper()
    if m == "match_winner":
        return {"H": "HOME", "D": "DRAW", "A": "AWAY"}.get(dec.selection, "OTHER")
    if m == "over_under_2_5":
        return "OVER" if "OVER" in s else "UNDER"
    if m == "btts":
        return "BTTS_YES" if "YES" in s else "BTTS_NO"
    if m == "correct_score":
        # a scoreline leans toward whoever it favours + total goals
        try:
            h, a = (int(x) for x in dec.selection.split("-"))
            side = "HOME" if h > a else ("AWAY" if a > h else "DRAW")
            tot = "OVER" if (h + a) >= 3 else "UNDER"
            return f"CS_{side}_{tot}"
        except (ValueError, AttributeError):
            return "CS_OTHER"
    return "OTHER"


# Pairs of leans that are considered positively correlated (same directional
# view of the game), so they must share a correlation group.
_CORRELATED = {
    frozenset({"HOME", "OVER"}), frozenset({"HOME", "BTTS_YES"}),
    frozenset({"AWAY", "OVER"}), frozenset({"AWAY", "BTTS_YES"}),
    frozenset({"OVER", "BTTS_YES"}),
    frozenset({"HOME", "CS_HOME_OVER"}), frozenset({"HOME", "CS_HOME_UNDER"}),
    frozenset({"AWAY", "CS_AWAY_OVER"}), frozenset({"AWAY", "CS_AWAY_UNDER"}),
    frozenset({"OVER", "CS_HOME_OVER"}), frozenset({"OVER", "CS_AWAY_OVER"}),
    frozenset({"BTTS_YES", "CS_HOME_OVER"}), frozenset({"BTTS_YES", "CS_AWAY_OVER"}),
}


def _correlated(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) in _CORRELATED


def group_selections(match_id: str, selections: list) -> dict:
    """Assign each selection a correlation-group id within one match.

    Returns {selection_index: group_id}. Uses union-find over the correlation
    relation so transitively-correlated selections share a group.
    """
    n = len(selections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    leans = [_lean(s) for s in selections]
    for i in range(n):
        for j in range(i + 1, n):
            if _correlated(leans[i], leans[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups[i] = f"{match_id}::grp{root}"
    return groups


def select_primaries(selections: list, groups: dict,
                     max_primary_per_match: int = 1) -> list:
    """Mark at most one PRIMARY selection per correlation group, then cap the
    number of primaries per match. Returns the list (mutated in place).

    The primary is the qualifying selection with the highest conservative EV.
    Non-primary correlated selections keep their grade but are flagged with
    CORRELATED_SELECTION_ALREADY_CHOSEN and demoted to NO BET as tradeable
    positions.
    """
    from .schema import RejectionCode, DecisionStatus, Grade

    # only qualifying/strong selections compete to be primary
    qualifying_idx = [i for i, s in enumerate(selections)
                      if s.decision_status in (DecisionStatus.QUALIFIED.value,
                                                DecisionStatus.STRONG_EVIDENCE.value)]
    # attach group id to every selection
    for i, s in enumerate(selections):
        s.correlation_group = groups.get(i)

    # pick best per group
    best_by_group: dict = {}
    for i in qualifying_idx:
        g = groups.get(i)
        cur = best_by_group.get(g)
        ev = selections[i].conservative_expected_value or -1e9
        if cur is None or ev > (selections[cur].conservative_expected_value or -1e9):
            best_by_group[g] = i

    chosen = sorted(best_by_group.values(),
                    key=lambda i: -(selections[i].conservative_expected_value or 0))
    chosen = chosen[:max_primary_per_match]
    chosen_set = set(chosen)

    for i in qualifying_idx:
        if i in chosen_set:
            selections[i].is_primary = True
        else:
            selections[i].is_primary = False
            selections[i].rejection_reasons.append(
                RejectionCode.CORRELATED_SELECTION_ALREADY_CHOSEN)
            selections[i].decision_status = DecisionStatus.WATCHLIST.value
            selections[i].grade = Grade.WATCHLIST.value
    return selections

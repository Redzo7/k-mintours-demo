"""Dominated-depot removal and ordering (Prop. 8, Cor. 9)."""

from .geometry import reachable_interval
from .models import Depot


def prune_dominated_and_order_depots(
    depots: tuple[Depot, ...], L: float
) -> tuple[tuple[Depot, ...], tuple[int, ...]]:
    """Remove dominated depots and sort the rest by Ml (Prop. 8, Cor. 9).

    A depot whose reachable interval [Ml, Mr] lies inside another depot's
    interval is dominated (Prop. 8). Depots that reach nothing are dropped
    too. After pruning, both Ml and Mr are strictly increasing (Cor. 9).

    Returns (kept depots in order, ids of pruned depots).
    """
    entries: list[tuple[float, float, Depot]] = []
    pruned: list[int] = []
    for d in depots:
        iv = reachable_interval(d, L)
        if iv is None:
            pruned.append(d.id)
        else:
            entries.append((iv[0], iv[1], d))

    # Ml ascending, ties broken by larger Mr first so the wider one is kept
    entries.sort(key=lambda e: (e[0], -e[1], e[2].id))

    kept: list[Depot] = []
    best_mr = float("-inf")
    for _ml, mr, d in entries:
        if mr > best_mr:
            kept.append(d)
            best_mr = mr
        else:
            pruned.append(d.id)
    return tuple(kept), tuple(pruned)

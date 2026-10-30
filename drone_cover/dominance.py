from .geometry import reachable_interval
from .models import Depot


# Remove dominated depots and order the remaining ones by their leftmost reachable point
def prune_dominated_and_order_depots(depots: tuple[Depot, ...], L: float) -> tuple[tuple[Depot, ...], tuple[int, ...]]:
    entries: list[tuple[float, float, Depot]] = []
    pruned: list[int] = []
    for d in depots:
        iv = reachable_interval(d, L)
        if iv is None:
            pruned.append(d.id)
        else:
            entries.append((iv[0], iv[1], d))

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

from scipy.optimize import brentq

from .geometry import f, reachable_interval
from .models import Depot


# Find the equilibrium point between two ordered depots
# Returns None if the reachable areas do not overlap
def equilibrium_point(di: Depot, dj: Depot, L: float) -> float | None:
    ivi = reachable_interval(di, L)
    ivj = reachable_interval(dj, L)
    if ivi is None or ivj is None:
        return None
    (mli, mri) = ivi
    (mlj, mrj) = ivj

    if not (mli < mlj and mri < mrj):
        raise ValueError("Depots must be properly ordered (remove dominated depots first)")
    if mlj >= mri:                       # reachable areas do not intersect
        return None

    def diff(x: float) -> float:
        return f(di, x, L) - f(dj, x, L)

    lo = mlj
    hi = mri
    if diff(lo) <= 0:
        return lo
    if diff(hi) >= 0:
        return hi
    return brentq(diff, lo, hi, xtol=1e-12)


# Compute the equilibrium points between the current depot and later depot whose reach overlaps the current one
def equilibrium_points(current: Depot, others: tuple[Depot, ...], L: float) -> dict[int, float]:
    result: dict[int, float] = {}
    for dj in others:
        p = equilibrium_point(current, dj, L)
        if p is None:           # ml is increasing along `others`, so no later depot can overlap either
            break
        result[dj.id] = p
    return result


def leftmost_equilibrium(eq: dict[int, float]) -> tuple[float | None, int | None]:
    if not eq:
        return None, None
    depot_id, p = min(eq.items(), key=lambda kv: kv[1])
    return p, depot_id

"""Equilibrium points between depots (Def. 11, Prop. 12)."""

from scipy.optimize import brentq

from .geometry import f, reachable_interval
from .models import Depot


def equilibrium_point(di: Depot, dj: Depot, L: float) -> float | None:
    """Def. 11: the point P in the overlap of the reaches with f(Di,P) = f(Dj,P).

    Def. 11 and the proof of Prop. 12 disagree (f = f vs. f = g); this follows
    Def. 11, which is what Lemma 13 and the algorithm use. On the overlap
    [Ml(Dj), Mr(Di)], f(Di) − f(Dj) is >= 0 at the left end and <= 0 at the
    right end; uniqueness of the root is checked numerically in the tests.

    `di` must precede `dj` in the pruned order. Returns None if the reachable
    intervals do not overlap.
    """
    ivi = reachable_interval(di, L)
    ivj = reachable_interval(dj, L)
    if ivi is None or ivj is None:
        return None
    mli, mri = ivi
    mlj, mrj = ivj

    if not (mli < mlj and mri < mrj):
        raise ValueError(
            "Depots must be properly ordered (remove dominated depots first)"
        )
    if mlj >= mri:  # reachable areas do not intersect
        return None

    def diff(x: float) -> float:
        return f(di, x, L) - f(dj, x, L)

    lo, hi = mlj, mri
    if diff(lo) <= 0:
        return lo
    if diff(hi) >= 0:
        return hi
    return brentq(diff, lo, hi, xtol=1e-12)


def equilibrium_points(
    current: Depot, others: tuple[Depot, ...], L: float
) -> dict[int, float]:
    """Equilibrium points P_{1,j} between `current` and each later depot."""
    result: dict[int, float] = {}
    for dj in others:
        p = equilibrium_point(current, dj, L)
        if p is None:  # Ml increases along `others`: no later depot overlaps
            break
        result[dj.id] = p
    return result


def leftmost_equilibrium(eq: dict[int, float]) -> tuple[float | None, int | None]:
    """Section 4, step 3: P = the leftmost P_{1,j} and the depot j defining it."""
    if not eq:
        return None, None
    depot_id, p = min(eq.items(), key=lambda kv: kv[1])
    return p, depot_id

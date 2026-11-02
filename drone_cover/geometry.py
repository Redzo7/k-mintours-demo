"""Reachability and maximal tours (Props. 3, 4 and 6 of the paper)."""

import math

from .models import EPS, Depot, Instance, Segment, Tour


def dist(depot: Depot, x: float) -> float:
    """Euclidean distance from the depot to the point (x, 0)."""
    return math.hypot(x - depot.x, depot.h)


def is_reachable(depot: Depot, x: float, L: float) -> bool:
    """True iff (x, 0) is strictly reachable: d(D, P) < L/2.

    Uses the same boundary convention as `f`, so `f(depot, x, L) > 0`
    whenever this returns True.
    """
    return dist(depot, x) < L / 2


def f(depot: Depot, x: float, L: float) -> float:
    """Prop. 3: x-length of the right maximal tour with left endpoint x.

    f(D, P) = (L² − 2L·d) / (2(L + xP − xD − d)),  d = d(D, P).
    (The proof in the paper has a sign typo, `+ 2L·d`; the statement is right.)
    The denominator is >= 2(L − 2d) > 0 for reachable P, but both numerator
    and denominator vanish near Ml when hD ≈ 0, hence the defensive guard.
    """
    d = dist(depot, x)
    if d >= L / 2:
        return 0.0
    denom = 2 * (L + x - depot.x - d)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (L * L - 2 * L * d) / denom)


def g(depot: Depot, x: float, L: float) -> float:
    """Prop. 4: x-length of the left maximal tour with right endpoint x."""
    d = dist(depot, x)
    if d >= L / 2:
        return 0.0
    denom = 2 * (L + depot.x - x - d)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (L * L - 2 * L * d) / denom)


def right_endpoint(depot: Depot, x: float, L: float) -> float:
    """Right endpoint of the right maximal tour starting at x."""
    if not is_reachable(depot, x, L):
        raise ValueError(f"x={x} is not reachable from depot {depot.id}")
    return x + f(depot, x, L)


def left_endpoint(depot: Depot, x: float, L: float) -> float:
    """Left endpoint of the left maximal tour ending at x."""
    if not is_reachable(depot, x, L):
        raise ValueError(f"x={x} is not reachable from depot {depot.id}")
    return x - g(depot, x, L)


def right_maximal_tour(depot: Depot, x: float, L: float) -> Tour:
    return Tour(depot=depot, a=x, b=right_endpoint(depot, x, L))


def left_maximal_tour(depot: Depot, x: float, L: float) -> Tour:
    return Tour(depot=depot, a=left_endpoint(depot, x, L), b=x)


def reachable_interval(depot: Depot, L: float) -> tuple[float, float] | None:
    """Prop. 6: (Ml(D), Mr(D)) = xD ∓ sqrt(L²/4 − hD²), or None if empty."""
    h = abs(depot.h)
    if h >= L / 2:
        return None
    r = math.sqrt(L * L / 4 - h * h)
    return depot.x - r, depot.x + r


def validate_feasible(
    segments: tuple[Segment, ...], depots: tuple[Depot, ...], L: float
) -> None:
    """Raise ValueError unless every segment point is strictly reachable.

    The paper requires every point of every segment to be at distance strictly
    less than L/2 from some depot; otherwise the greedy never terminates.
    Numerically, "strictly" means at least EPS inside some [Ml, Mr].
    Runs in O(k log k + m) by merging the depots' reachable intervals.
    """
    intervals: list[tuple[float, float]] = []
    for d in depots:
        iv = reachable_interval(d, L)
        if iv is not None and iv[1] - iv[0] > 2 * EPS:
            intervals.append((iv[0] + EPS, iv[1] - EPS))
    intervals.sort()

    merged: list[list[float]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    j = 0
    for seg in segments:  # segments are sorted, so a single sweep suffices
        while j < len(merged) and merged[j][1] < seg.xl:
            j += 1
        if j == len(merged) or merged[j][0] > seg.xl:
            raise ValueError(
                f"Infeasible instance: x={seg.xl} (segment [{seg.xl}, {seg.xr}]) "
                "is not strictly reachable from any depot"
            )
        if merged[j][1] < seg.xr:
            raise ValueError(
                f"Infeasible instance: points right of x={merged[j][1]} "
                f"(segment [{seg.xl}, {seg.xr}]) are not strictly reachable "
                "from any depot"
            )


def validate_instance(instance: Instance) -> None:
    """Feasibility check of a whole instance (see `validate_feasible`)."""
    validate_feasible(instance.segments, instance.depots, instance.L)

"""Greedy algorithm of Bereg et al. for one depot (Thm. 7)."""

from dataclasses import replace

from .geometry import f, validate_instance
from .models import EPS, Depot, GreedyStep, Instance, Segment, Tour


def greedy_cover(
    depot: Depot,
    segments: tuple[Segment, ...],
    L: float,
    stop_at: float | None = None,
) -> tuple[tuple[Tour, ...], float, tuple[Segment, ...]]:
    """Greedily cover sorted, disjoint `segments` with right maximal tours.

    Thm. 7; see `greedy_cover_traced` for the details. Returns
    (tours, cut point M, remaining segments right of M).
    """
    tours, cut, remaining, _ = greedy_cover_traced(depot, segments, L, stop_at)
    return tours, cut, remaining


def greedy_cover_traced(
    depot: Depot,
    segments: tuple[Segment, ...],
    L: float,
    stop_at: float | None = None,
) -> tuple[tuple[Tour, ...], float, tuple[Segment, ...], tuple[GreedyStep, ...]]:
    """Greedily cover sorted, disjoint `segments` with right maximal tours.

    Thm. 7: from the leftmost uncovered point take the right maximal tour; if
    its right endpoint falls in a gap, shrink it to the end of the last segment
    it touches and restart at the next segment's left endpoint.

    With `stop_at` (the equilibrium point P of Section 4) the greedy stops as
    soon as the next tour would start at or right of P, i.e. once the last
    tour's right endpoint has passed P, or the next segment begins after P.

    A segment whose right end lies within EPS of a tour's right endpoint counts
    as covered by that tour, so float rounding never adds a sliver tour.

    Returns (tours, cut point M, remaining segments right of M, one
    `GreedyStep` per tour). The caller must ensure every point to cover is
    strictly reachable from `depot` (`geometry.validate_feasible`); otherwise
    a ValueError is raised.
    """
    segs = list(segments)
    n = len(segs)
    if n == 0:
        return (), 0.0, (), ()

    tours: list[Tour] = []
    steps: list[GreedyStep] = []
    i = 0
    s = segs[0].xl

    while i < n and (stop_at is None or s < stop_at):
        step = f(depot, s, L)
        if step <= 0.0:
            raise ValueError(f"x={s} is not strictly reachable from depot {depot.id}")
        b = s + step

        while i < n and segs[i].xr <= b + EPS:  # segments fully covered
            i += 1

        start = s
        if i < n and segs[i].xl <= b:  # b lies inside segment i
            end = b
            s = b  # next tour starts here (consecutive)
        else:  # b in a gap or past the last segment
            end = segs[i - 1].xr  # shorten the tour (or snap within EPS)
            if i < n:
                s = segs[i].xl
        tours.append(Tour(depot, start, end))
        steps.append(
            GreedyStep(
                start=start,
                reach=step,
                raw_end=b,
                end=end,
                shrunk=end < b - EPS,
                next_start=s if i < n else None,
                stopped_by_p=False,
            )
        )

    if i < n and steps:  # loop ended because the stop rule fired
        steps[-1] = replace(steps[-1], stopped_by_p=True)

    if i >= n:
        remaining: list[Segment] = []
    elif s > segs[i].xl:
        remaining = [Segment(s, segs[i].xr)] + segs[i + 1 :]
    else:
        remaining = segs[i:]

    cut = tours[-1].b if tours else segs[0].xl
    return tuple(tours), cut, tuple(remaining), tuple(steps)


def solve_1_mintours(instance: Instance) -> tuple[Tour, ...]:
    """Optimal solution of 1-MinTours (Thm. 7)."""
    if len(instance.depots) != 1:
        raise ValueError("1-MinTours expects exactly one depot")
    validate_instance(instance)
    tours, _, _ = greedy_cover(instance.depots[0], instance.segments, instance.L)
    return tours

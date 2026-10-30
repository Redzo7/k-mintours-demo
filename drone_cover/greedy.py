from .geometry import f
from .models import Depot, Instance, Segment, Tour


# Greedily cover `segments` (sorted, disjoint, non-empty) from one depot
# Returns (tours, cut point M, remaining segments to the right of M)
def greedy_cover(
    depot: Depot,
    segments: tuple[Segment, ...],
    L: float,
    stop_at: float | None = None,
) -> tuple[tuple[Tour, ...], float, tuple[Segment, ...]]:
    segs = list(segments)
    n = len(segs)
    if n == 0:
        return (), 0, ()

    tours: list[Tour] = []
    i = 0
    s = segs[0].xl

    while i < n and (stop_at is None or s < stop_at):
        step = f(depot, s, L)
        if step <= 1e-12:
            raise ValueError(f"point x={s} is not strictly reachable from depot {depot.id}")
        b = s + step

        while i < n and segs[i].xr <= b:         # segments fully covered
            i += 1

        start = s
        if i < n and segs[i].xl <= b:            # b lies inside segment i
            end = b
            s = b                                # next tour starts here (consecutive)
        else:                                    # b in a gap or past the last segment
            end = segs[i - 1].xr                 # shorten the tour
            if i < n:
                s = segs[i].xl
        tours.append(Tour(depot, start, end))

    if i >= n:
        remaining: list[Segment] = []
    elif s > segs[i].xl:
        remaining = [Segment(s, segs[i].xr)] + segs[i + 1:]
    else:
        remaining = segs[i:]

    cut = tours[-1].b if tours else segs[0].xl
    return tuple(tours), cut, tuple(remaining)


# Optimal solution of 1-MinTours
def solve_1_mintours(instance: Instance) -> tuple[Tour, ...]:
    if len(instance.depots) != 1:
        raise ValueError("1-MinTours expects exactly one depot")
    tours, _, _ = greedy_cover(instance.depots[0], instance.segments, instance.L)
    return tours

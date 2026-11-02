"""Shared helpers for the test suite: random instances and a naive baseline."""

import numpy as np

from drone_cover.geometry import dist, f, validate_feasible
from drone_cover.models import EPS, Depot, Instance, Segment, Solution


def random_instance(rng: np.random.Generator, L: float = 10.0) -> Instance:
    """Random instance (not necessarily feasible) on roughly [0, 20]."""
    k = int(rng.integers(1, 8))
    depots = tuple(
        Depot(
            j,
            float(rng.uniform(0, 20)),
            float(rng.uniform(0, 1) if rng.random() < 0.5 else rng.uniform(3, 4.9)),
        )
        for j in range(k)
    )
    pts = np.sort(rng.uniform(0, 20, size=2 * int(rng.integers(1, 10))))
    segments = tuple(
        Segment(float(a), float(b)) for a, b in pts.reshape(-1, 2) if b - a > 1e-3
    )
    return Instance(segments, depots, L)


def random_feasible_instances(seed: int, count: int) -> list[Instance]:
    rng = np.random.default_rng(seed)
    out: list[Instance] = []
    while len(out) < count:
        inst = random_instance(rng)
        if not inst.segments:
            continue
        try:
            validate_feasible(inst.segments, inst.depots, inst.L)
        except ValueError:
            continue
        out.append(inst)
    return out


def baseline_tour_count(inst: Instance) -> int:
    """Naive feasible solution: from the leftmost uncovered point, take the
    longest right maximal tour over all depots. Optimal is never worse."""
    segs, L = inst.segments, inst.L
    i, s, n = 0, segs[0].xl, 0
    while i < len(segs):
        step = max(f(d, s, L) for d in inst.depots)
        assert step > 0
        b = s + step
        n += 1
        while i < len(segs) and segs[i].xr <= b + EPS:
            i += 1
        if i < len(segs):
            s = max(b, segs[i].xl)
    return n


def assert_valid_solution(inst: Instance, sol: Solution) -> None:
    """Each tour fits the budget, its ends are reachable, and tours cover
    every segment (checked on a fine sample of points)."""
    L = inst.L
    # Snapping a tour end by <= EPS changes its length by <= 2·EPS
    for t in sol.tours:
        assert t.a <= t.b
        assert t.length <= L + 2 * EPS, t
        assert dist(t.depot, t.a) <= L / 2 + EPS
        assert dist(t.depot, t.b) <= L / 2 + EPS
    tours = sorted(sol.tours, key=lambda t: t.a)
    for seg in inst.segments:
        for x in np.linspace(seg.xl, seg.xr, 400):
            assert any(t.a - EPS <= x <= t.b + EPS for t in tours), (x, seg)

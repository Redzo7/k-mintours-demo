"""Seeded random instance generation for the demo and experiments."""

import numpy as np

from .geometry import reachable_interval
from .models import EPS, Depot, Instance, Segment

MAX_TRIES = 1000


def _strict_union(depots: tuple[Depot, ...], L: float) -> list[tuple[float, float]]:
    """Union of the strictly reachable intervals (Ml + margin, Mr − margin)."""
    margin = 1e3 * EPS  # stay well inside, so feasibility is not a rounding call
    ivs = []
    for d in depots:
        iv = reachable_interval(d, L)
        if iv is not None and iv[1] - iv[0] > 2 * margin:
            ivs.append((iv[0] + margin, iv[1] - margin))
    ivs.sort()
    merged: list[list[float]] = []
    for lo, hi in ivs:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def random_instance(
    seed: int,
    k: int,
    m: int,
    L: float = 10.0,
    width: float = 20.0,
    max_height_ratio: float = 0.95,
) -> Instance:
    """Random feasible instance with k depots and at most m segments.

    Depots are uniform in [0, width] x [0, max_height_ratio * L/2]. Segment
    endpoints are drawn inside the union of the depots' strictly reachable
    intervals (Prop. 6), and a segment never spans a hole in that union, so
    the instance is feasible by construction. Duplicate or too-short segments
    are dropped, so fewer than m segments may be returned (never zero).
    Deterministic for a given seed (`numpy.random.default_rng`).
    """
    if k < 1 or m < 1:
        raise ValueError(f"Need k >= 1 and m >= 1, got k={k}, m={m}")
    if not 0 <= max_height_ratio < 1:
        raise ValueError("max_height_ratio must be in [0, 1)")
    rng = np.random.default_rng(seed)
    for _ in range(MAX_TRIES):
        depots = tuple(
            Depot(
                j,
                float(rng.uniform(0, width)),
                float(rng.uniform(0, max_height_ratio * L / 2)),
            )
            for j in range(k)
        )
        union = _strict_union(depots, L)
        if not union:
            continue
        lengths = np.array([hi - lo for lo, hi in union])
        # 2m points, each in a component chosen proportionally to its length
        comp = rng.choice(len(union), size=2 * m, p=lengths / lengths.sum())
        pts = sorted(
            (int(c), float(rng.uniform(union[c][0], union[c][1]))) for c in comp
        )
        segments: list[Segment] = []
        for (ca, a), (cb, b) in zip(pts[::2], pts[1::2]):
            if ca != cb:  # would span a hole: clip to the component end
                b = union[ca][1]
            if b - a <= 1e-3 or (segments and a - segments[-1].xr <= 1e-3):
                continue
            segments.append(Segment(a, b))
        if segments:
            return Instance(tuple(segments), depots, L)
    raise ValueError(
        f"Could not generate a feasible instance after {MAX_TRIES} tries "
        f"(seed={seed}, k={k}, m={m}, L={L})"
    )

"""Data model of the k-MinTours problem (Section 2 of the paper).

All geometric comparisons in the package use the single tolerance `EPS`.
"""

import math
from dataclasses import dataclass

EPS = 1e-9


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class Segment:
    """Interval [xl, xr] on the x-axis to be covered, with xl < xr."""

    xl: float
    xr: float

    def __post_init__(self) -> None:
        _check_finite("Segment.xl", self.xl)
        _check_finite("Segment.xr", self.xr)
        if not self.xl < self.xr:
            raise ValueError(f"Segment needs xl < xr, got [{self.xl}, {self.xr}]")


@dataclass(frozen=True)
class Depot:
    """Depot at (x, h), h >= 0 being the height above the x-axis."""

    id: int  # stable index in the instance
    x: float
    h: float

    def __post_init__(self) -> None:
        _check_finite("Depot.x", self.x)
        _check_finite("Depot.h", self.h)
        if self.h < 0:
            raise ValueError(f"Depot {self.id} has negative height h={self.h}")


@dataclass(frozen=True)
class Tour:
    """Triangle tour with apex at `depot` and covering side [a, b]."""

    depot: Depot
    a: float
    b: float

    @property
    def length(self) -> float:
        """|AB| + d(D, A) + d(D, B)."""
        d = self.depot
        return (
            (self.b - self.a)
            + math.hypot(self.a - d.x, d.h)
            + math.hypot(self.b - d.x, d.h)
        )


@dataclass(frozen=True)
class Instance:
    """Segments (sorted, pairwise disjoint), depots and the length budget L.

    Structural validity is checked on construction; geometric feasibility
    (every point strictly reachable) is checked by
    `geometry.validate_feasible`, called by the solvers.
    """

    segments: tuple[Segment, ...]
    depots: tuple[Depot, ...]
    L: float

    def __post_init__(self) -> None:
        _check_finite("L", self.L)
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")
        for prev, nxt in zip(self.segments, self.segments[1:]):
            if not prev.xr < nxt.xl:
                raise ValueError(
                    "Segments must be sorted and pairwise disjoint: "
                    f"[{prev.xl}, {prev.xr}] then [{nxt.xl}, {nxt.xr}]"
                )
        ids = [d.id for d in self.depots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Depot ids must be unique, got {ids}")


@dataclass(frozen=True)
class StepTrace:
    """One recursion step of k-MinTours (Section 4), for visualisation."""

    step: int
    current_depot: int  # D1 of this step
    active_depots: tuple[int, ...]  # depots still in play
    pruned_depots: tuple[int, ...]  # removed as dominated (step 1 only)
    equilibrium_points: tuple[tuple[int, float], ...]  # (depot j, x of P_1j)
    p: float | None  # leftmost equilibrium point
    p_depot: int | None  # depot that defines P
    stop_at: float | None  # where the greedy actually stopped (None: base case)
    tours: tuple[Tour, ...]  # tours added from current depot
    m: float  # cut point M
    remaining: tuple[Segment, ...]  # sub-instance passed on


@dataclass(frozen=True)
class Solution:
    tours: tuple[Tour, ...]
    trace: tuple[StepTrace, ...]

    @property
    def num_tours(self) -> int:
        return len(self.tours)

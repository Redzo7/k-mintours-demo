import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    xl: float
    xr: float          # xl < xr

@dataclass(frozen=True)
class Depot:
    id: int            # stable index in the instance
    x: float
    h: float           # height above the x-axis

@dataclass(frozen=True)
class Tour:
    depot: Depot
    a: float           # covering side [a, b] on the x-axis
    b: float

    @property
    def length(self) -> float:
        d = self.depot
        return (self.b - self.a) + math.hypot(self.a - d.x, d.h) \
                                 + math.hypot(self.b - d.x, d.h)

@dataclass(frozen=True)
class Instance:
    segments: tuple[Segment, ...]   # sorted, disjoint
    depots: tuple[Depot, ...]
    L: float

@dataclass(frozen=True)
class StepTrace:
    step: int
    current_depot: int                    # D1 of this step
    active_depots: tuple[int, ...]        # depots still in play
    pruned_depots: tuple[int, ...]        # removed as dominated
    equilibrium_points: dict[int, float]  # depot j -> x of P_1j
    p: float | None                       # leftmost equilibrium point
    p_depot: int | None                   # depot that defines P
    tours: tuple[Tour, ...]               # tours added from current depot
    m: float                              # cut point M
    remaining: tuple[Segment, ...]        # sub-instance passed on

@dataclass(frozen=True)
class Solution:
    tours: tuple[Tour, ...]
    trace: tuple[StepTrace, ...]    # StepTrace as in the other thread

    @property
    def m(self) -> int:
        return len(self.tours)
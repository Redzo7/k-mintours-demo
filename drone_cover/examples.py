"""Hand-crafted instances that each exercise one case of the algorithm.

All use L = 10, so every depot reaches at most L/2 = 5 from its position.
Used by the demo (`python -m drone_cover.demo --example NAME`) and the tests.
"""

from collections.abc import Callable

from .models import Depot, Instance, Segment

L_DEFAULT = 10.0


def _inst(
    segments: list[tuple[float, float]], depots: list[tuple[float, float]]
) -> Instance:
    return Instance(
        segments=tuple(Segment(a, b) for a, b in segments),
        depots=tuple(Depot(i, x, h) for i, (x, h) in enumerate(depots)),
        L=L_DEFAULT,
    )


def basic() -> Instance:
    """Two overlapping depots: one equilibrium point P, one handover."""
    return _inst([(1, 6), (7.5, 13)], [(4, 3), (11, 3)])


def dominated() -> Instance:
    """D1's reachable interval lies inside D0's, so D1 is pruned (Prop. 8)."""
    return _inst([(1, 5), (6, 9), (10, 14)], [(4, 3), (5, 4.5), (11, 2)])


def gap() -> Instance:
    """Many small segments: tours ending in a gap are shortened (Thm. 7)."""
    return _inst(
        [(0.5, 1.5), (2, 2.3), (4, 6), (8.5, 9.5), (11, 12), (15, 18)],
        [(5, 1), (14, 1)],
    )


def isolated() -> Instance:
    """D0 overlaps no other depot: no equilibrium point, D0 covers up to Mr."""
    return _inst([(0, 6), (11, 17)], [(3, 3), (14, 3)])


def axis() -> Instance:
    """A depot on the axis (h = 0): f jumps from 0 to L/2 at Ml."""
    return _inst([(-1, 4), (5, 13)], [(3, 0), (10, 2)])


def showcase() -> Instance:
    """Five depots: a dominated one, one on the axis, gaps and handovers."""
    return _inst(
        [(0, 2.5), (3, 6), (7, 9.5), (10, 13), (14.5, 15), (16, 20), (21, 24)],
        [(3, 3), (4.5, 4.4), (10, 1), (21, 2), (17, 0)],
    )


EXAMPLES: dict[str, Callable[[], Instance]] = {
    "showcase": showcase,
    "basic": basic,
    "dominated": dominated,
    "gap": gap,
    "isolated": isolated,
    "axis": axis,
}

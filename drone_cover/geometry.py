import math

from .models import Depot, Tour


# Euclidean distance from the depot to the point (x, 0)
def dist(depot: Depot, x: float) -> float:
    return math.hypot(x - depot.x, depot.h)


def is_reachable(depot: Depot, x: float, L: float) -> bool:
    return dist(depot, x) <= L / 2 + 1e-12


# Maximum rightward covering span when the left endpoint is x
def f(depot: Depot, x: float, L: float) -> float:
    d = dist(depot, x)
    if d >= L / 2:
        return 0.0

    num = L * L - 2 * L * d
    denom = 2 * (L + x - depot.x - d)

    if denom <= 1e-12:
        return 0.0
    return max(0.0, num / denom)


# Maximum leftward covering span when the right endpoint is x
def g(depot: Depot, x: float, L: float) -> float:
    d = dist(depot, x)
    if d >= L / 2:
        return 0.0

    num = L * L - 2 * L * d
    denom = 2 * (L + depot.x - x - d)

    if denom <= 1e-12:
        return 0.0
    return max(0.0, num / denom)


# Right endpoint of the right-maximal tour starting at x
def right_endpoint(depot: Depot, x: float, L: float) -> float:
    if not is_reachable(depot, x, L):
        raise ValueError(f"x={x} is not reachable from depot {depot.id}")
    return x + f(depot, x, L)


# Left endpoint of the left-maximal tour ending at x
def left_endpoint(depot: Depot, x: float, L: float) -> float:
    if not is_reachable(depot, x, L):
        raise ValueError(f"x={x} is not reachable from depot {depot.id}")
    return x - g(depot, x, L)


def right_maximal_tour(depot: Depot, x: float, L: float) -> Tour:
    return Tour(depot=depot, a=x, b=right_endpoint(depot, x, L))


def left_maximal_tour(depot: Depot, x: float, L: float) -> Tour:
    return Tour(depot=depot, a=left_endpoint(depot, x, L), b=x)


# Return the leftmost and rightmost reachable x-coordinates
def reachable_interval(depot: Depot, L: float) -> tuple[float, float] | None:
    if depot.h >= L / 2:
        return None
    r = math.sqrt(L * L / 4 - depot.h ** 2)
    return depot.x - r, depot.x + r

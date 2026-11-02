import math

import numpy as np
import pytest

from drone_cover.geometry import (
    dist,
    f,
    g,
    is_reachable,
    reachable_interval,
    validate_feasible,
)
from drone_cover.models import Depot, Segment, Tour

L = 10.0


def test_f_and_g_match_perimeter_equation():
    """Props. 3-4: the maximal tours have length exactly L."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        d = Depot(0, float(rng.uniform(-5, 5)), float(rng.uniform(0, 4.9)))
        ml, mr = reachable_interval(d, L)
        x = float(rng.uniform(ml, mr))
        assert math.isclose(Tour(d, x, x + f(d, x, L)).length, L, rel_tol=1e-9)
        assert math.isclose(Tour(d, x - g(d, x, L), x).length, L, rel_tol=1e-9)


def test_hand_computed_f_on_axis():
    # Depot on the axis: right of it a tour [x, 5] has length 2·5 = L
    d = Depot(0, 0.0, 0.0)
    assert math.isclose(f(d, -4.0, L), 5.0)
    assert math.isclose(f(d, 1.0, L), 4.0)
    assert math.isclose(g(d, 1.0, L), 5.0)


def test_reachable_interval_matches_distance():
    d = Depot(0, 2.0, 3.0)
    ml, mr = reachable_interval(d, L)
    assert math.isclose(ml, 2.0 - 4.0) and math.isclose(mr, 2.0 + 4.0)
    assert math.isclose(dist(d, ml), L / 2)
    assert reachable_interval(Depot(1, 0.0, 5.0), L) is None


def test_unreachable_points():
    d = Depot(0, 0.0, 3.0)
    assert not is_reachable(d, 4.0, L)  # d = L/2 exactly: not strict
    assert f(d, 4.0, L) == 0.0 and g(d, 4.0, L) == 0.0
    assert is_reachable(d, 3.99, L) and f(d, 3.99, L) > 0


def test_negative_height_rejected():
    with pytest.raises(ValueError, match="negative height"):
        Depot(0, 0.0, -10.0)


def test_validate_feasible_accepts_overlapping_reaches():
    depots = (Depot(0, 0.0, 3.0), Depot(1, 6.0, 3.0))  # reaches (-4,4), (2,10)
    validate_feasible((Segment(-3.0, 9.0),), depots, L)


@pytest.mark.parametrize(
    "segment",
    [
        Segment(-3.0, 4.0),  # ends exactly at Mr: not strictly reachable
        Segment(4.5, 5.0),  # entirely outside
        Segment(-5.0, 0.0),  # starts left of Ml
    ],
)
def test_validate_feasible_rejects(segment):
    with pytest.raises(ValueError, match="Infeasible"):
        validate_feasible((segment,), (Depot(0, 0.0, 3.0),), L)


def test_validate_feasible_rejects_gap_between_depots():
    depots = (Depot(0, 0.0, 3.0), Depot(1, 8.0, 3.0))  # reaches touch at x=4
    with pytest.raises(ValueError, match="Infeasible"):
        validate_feasible((Segment(-3.0, 11.0),), depots, L)

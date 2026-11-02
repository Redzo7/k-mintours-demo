import numpy as np
import pytest

from drone_cover.dominance import prune_dominated_and_order_depots
from drone_cover.equilibrium import (
    equilibrium_point,
    equilibrium_points,
    leftmost_equilibrium,
)
from drone_cover.geometry import f, reachable_interval
from drone_cover.models import Depot

L = 10.0


def _overlapping_ordered_pairs(seed: int, count: int):
    rng = np.random.default_rng(seed)
    found = 0
    while found < count:
        ds = tuple(
            Depot(j, float(rng.uniform(0, 10)), float(rng.uniform(0, 4.9)))
            for j in range(2)
        )
        kept, _ = prune_dominated_and_order_depots(ds, L)
        if len(kept) < 2:
            continue
        a, b = kept
        if reachable_interval(b, L)[0] >= reachable_interval(a, L)[1]:
            continue
        found += 1
        yield a, b


def test_equilibrium_satisfies_definition():
    """Def. 11: f(Di, P) = f(Dj, P), and P lies in the overlap of the reaches."""
    for a, b in _overlapping_ordered_pairs(3, 500):
        p = equilibrium_point(a, b, L)
        mlb = reachable_interval(b, L)[0]
        mra = reachable_interval(a, L)[1]
        assert mlb <= p <= mra
        # f(Dj) is extremely steep near Ml(Dj) when hDj ≈ 0, so check that
        # P brackets the sign change rather than |f(Di) − f(Dj)| at P itself.
        assert f(a, p - 1e-9, L) - f(b, p - 1e-9, L) >= -1e-9
        assert f(a, p + 1e-9, L) - f(b, p + 1e-9, L) <= 1e-9


def test_equilibrium_is_unique():
    """Prop. 12, checked numerically: f(Di) − f(Dj) changes sign at most once."""
    for a, b in _overlapping_ordered_pairs(4, 500):
        lo = reachable_interval(b, L)[0]
        hi = reachable_interval(a, L)[1]
        xs = np.linspace(lo, hi, 1000)[1:-1]
        v = np.array([f(a, x, L) - f(b, x, L) for x in xs])
        signs = np.sign(v[np.abs(v) > 1e-12])
        assert np.count_nonzero(np.diff(signs)) <= 1


def test_no_overlap_gives_none():
    assert equilibrium_point(Depot(0, 0.0, 3.0), Depot(1, 10.0, 3.0), L) is None


def test_wrong_order_raises():
    with pytest.raises(ValueError):
        equilibrium_point(Depot(0, 5.0, 3.0), Depot(1, 0.0, 3.0), L)


def test_equilibrium_points_stop_at_first_non_overlapping_depot():
    d1, d2, d3 = Depot(0, 0.0, 3.0), Depot(1, 5.0, 3.0), Depot(2, 20.0, 3.0)
    eq = equilibrium_points(d1, (d2, d3), L)
    assert set(eq) == {1}
    assert leftmost_equilibrium(eq) == (eq[1], 1)
    assert leftmost_equilibrium({}) == (None, None)

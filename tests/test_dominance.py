from itertools import pairwise

import numpy as np

from drone_cover.dominance import prune_dominated_and_order_depots
from drone_cover.geometry import f, reachable_interval
from drone_cover.models import Depot

L = 10.0


def test_contained_interval_is_pruned_and_order_is_by_ml():
    wide = Depot(0, 0.0, 0.5)  # reach ≈ (-4.97, 4.97)
    narrow = Depot(1, 0.5, 4.0)  # reach (-2.5, 3.5), inside `wide`
    right = Depot(2, 6.0, 1.0)
    kept, pruned = prune_dominated_and_order_depots((right, narrow, wide), L)
    assert [d.id for d in kept] == [0, 2]
    assert pruned == (1,)


def test_depot_reaching_nothing_is_pruned():
    kept, pruned = prune_dominated_and_order_depots((Depot(0, 0.0, 6.0),), L)
    assert kept == () and pruned == (0,)


def test_kept_depots_have_strictly_increasing_reach():
    """Cor. 9."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        depots = tuple(
            Depot(j, float(rng.uniform(0, 20)), float(rng.uniform(0, 4.9)))
            for j in range(6)
        )
        kept, pruned = prune_dominated_and_order_depots(depots, L)
        assert len(kept) + len(pruned) == len(depots)
        ivs = [reachable_interval(d, L) for d in kept]
        for (ml1, mr1), (ml2, mr2) in pairwise(ivs):
            assert ml1 < ml2 and mr1 < mr2


def test_pruned_depot_never_has_a_longer_tour():
    """Prop. 8: the dominating depot is at least as good at every point."""
    rng = np.random.default_rng(2)
    checked = 0
    while checked < 2000:
        a = Depot(0, float(rng.uniform(-5, 5)), float(rng.uniform(0, 4.9)))
        b = Depot(1, float(rng.uniform(-5, 5)), float(rng.uniform(0, 4.9)))
        kept, pruned = prune_dominated_and_order_depots((a, b), L)
        if len(kept) == 2:
            continue
        dominated = a if pruned == (0,) else b
        ml, mr = reachable_interval(dominated, L)
        for x in np.linspace(ml, mr, 20):
            assert f(dominated, x, L) <= f(kept[0], x, L) + 1e-9
        checked += 1

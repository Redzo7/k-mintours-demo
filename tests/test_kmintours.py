import pytest
from helpers import (
    assert_valid_solution,
    baseline_tour_count,
    random_feasible_instances,
)

from drone_cover.kmintours import solve_k_mintours
from drone_cover.models import Depot, Instance, Segment

L = 10.0


def test_hand_computed_two_depots():
    # Reaches (-5, 5) and (1, 11); every tour covers at most 5, so 14/5 → 3.
    # With h = 0, f(D2, ·) jumps from 0 at Ml(D2) = 1 to 5 just right of it,
    # so P = 1⁺ and D1 also takes the tour [1, 5].
    d1, d2 = Depot(0, 0.0, 0.0), Depot(1, 6.0, 0.0)
    sol = solve_k_mintours(Instance((Segment(-4.0, 10.0),), (d1, d2), L))
    assert sol.num_tours == 3
    assert [(t.depot.id, t.a, t.b) for t in sol.tours] == [
        (0, -4.0, 1.0),
        (0, 1.0, pytest.approx(5.0)),
        (1, pytest.approx(5.0), 10.0),
    ]
    first = sol.trace[0]
    assert first.p == pytest.approx(1.0) and first.p_depot == 1
    assert first.m == pytest.approx(5.0)


def test_single_depot_matches_greedy():
    sol = solve_k_mintours(Instance((Segment(-4.5, 4.5),), (Depot(0, 0, 0),), L))
    assert sol.num_tours == 2
    assert sol.trace[0].stop_at is None


def test_non_overlapping_depots_stop_at_mr():
    d1, d2 = Depot(0, 0.0, 3.0), Depot(1, 20.0, 3.0)
    segs = (Segment(-3.0, 3.0), Segment(17.0, 23.0))
    sol = solve_k_mintours(Instance(segs, (d1, d2), L))
    assert sol.trace[0].p is None
    assert sol.trace[0].stop_at == pytest.approx(4.0)
    assert {t.depot.id for t in sol.tours[:1]} == {0}
    assert sol.tours[-1].depot.id == 1


def test_trace_records_pruned_depots_and_is_hashable():
    depots = (Depot(0, 0.0, 0.5), Depot(1, 0.5, 4.0))
    sol = solve_k_mintours(Instance((Segment(-1.0, 1.0),), depots, L))
    assert sol.trace[0].pruned_depots == (1,)
    hash(sol)


def test_empty_instance():
    sol = solve_k_mintours(Instance((), (Depot(0, 0, 0),), L))
    assert sol.num_tours == 0 and sol.trace == ()


@pytest.mark.parametrize("seed", range(4))
def test_random_instances_are_valid_and_no_worse_than_baseline(seed):
    for inst in random_feasible_instances(seed, 150):
        sol = solve_k_mintours(inst)
        assert_valid_solution(inst, sol)
        assert sol.num_tours <= baseline_tour_count(inst)


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "segments",
    [
        (Segment(2.0, 3.0), Segment(-1.0, 0.0)),  # unsorted
        (Segment(0.0, 2.0), Segment(1.0, 3.0)),  # overlapping
        (Segment(0.0, 1.0), Segment(1.0, 2.0)),  # touching
    ],
)
def test_bad_segment_order_rejected(segments):
    with pytest.raises(ValueError, match="sorted and pairwise disjoint"):
        Instance(segments, (Depot(0, 0, 1),), L)


def test_degenerate_segment_rejected():
    with pytest.raises(ValueError, match="xl < xr"):
        Segment(1.0, 1.0)


def test_duplicate_depot_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        Instance((Segment(0, 1),), (Depot(0, 0, 1), Depot(0, 1, 1)), L)


@pytest.mark.parametrize("bad_L", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_budget_rejected(bad_L):
    with pytest.raises(ValueError):
        Instance((Segment(0, 1),), (Depot(0, 0, 1),), bad_L)


def test_infeasible_gap_rejected_up_front():
    depots = (Depot(0, 0.0, 1.0), Depot(1, 20.0, 1.0))
    with pytest.raises(ValueError, match="Infeasible"):
        solve_k_mintours(Instance((Segment(0.0, 10.0),), depots, L))


def test_no_usable_depot_rejected():
    with pytest.raises(ValueError, match="Infeasible"):
        solve_k_mintours(Instance((Segment(0.0, 1.0),), (Depot(0, 0, 6.0),), L))

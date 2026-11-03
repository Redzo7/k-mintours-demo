import pytest
from helpers import assert_valid_solution, baseline_tour_count

from drone_cover.examples import EXAMPLES
from drone_cover.geometry import validate_instance
from drone_cover.kmintours import solve_k_mintours


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_is_feasible_and_solves(name):
    inst = EXAMPLES[name]()
    validate_instance(inst)
    sol = solve_k_mintours(inst)
    assert_valid_solution(inst, sol)
    assert sol.num_tours <= baseline_tour_count(inst)


def _trace(name):
    return solve_k_mintours(EXAMPLES[name]()).trace


def test_basic_has_one_handover():
    trace = _trace("basic")
    assert len(trace) == 2 and trace[0].p is not None


def test_dominated_prunes_a_depot():
    assert _trace("dominated")[0].pruned_depots == (1,)


def test_gap_shrinks_tours():
    assert any(g.shrunk for t in _trace("gap") for g in t.greedy_steps)


def test_isolated_stops_at_mr():
    first = _trace("isolated")[0]
    assert first.p is None and first.stop_at is not None


def test_axis_has_depot_on_axis():
    assert any(d.h == 0 for d in EXAMPLES["axis"]().depots)


def test_showcase_combines_cases():
    trace = _trace("showcase")
    assert trace[0].pruned_depots
    assert len(trace) >= 3
    assert any(g.shrunk for t in trace for g in t.greedy_steps)

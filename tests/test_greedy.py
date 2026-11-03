import math

import numpy as np
import pytest

from drone_cover.geometry import f, reachable_interval
from drone_cover.greedy import greedy_cover, greedy_cover_traced, solve_1_mintours
from drone_cover.models import EPS, Depot, Instance, Segment

L = 10.0
AXIS = Depot(0, 0.0, 0.0)  # on the axis: every maximal tour ends at x = 5 or covers 5


def test_hand_computed_two_tours():
    tours = solve_1_mintours(Instance((Segment(-4.5, 4.5),), (AXIS,), L))
    assert [(t.a, t.b) for t in tours] == [(-4.5, 0.5), (0.5, 4.5)]


def test_exact_fit_segment_needs_one_tour():
    rng = np.random.default_rng(5)
    d = Depot(0, 0.0, 1.0)
    ml, mr = reachable_interval(d, L)
    for _ in range(500):
        s = float(rng.uniform(ml + 0.1, mr - 0.5))
        b = s + f(d, s, L)
        tours, _, _ = greedy_cover(d, (Segment(s, b),), L)
        assert len(tours) == 1


def test_tour_ending_in_gap_is_shrunk():
    segs = (Segment(-4.0, -2.0), Segment(2.0, 3.0))
    tours, cut, remaining = greedy_cover(AXIS, segs, L)
    # first tour would reach x=1 (a gap): shrunk to -2, next starts at 2
    assert [(t.a, t.b) for t in tours] == [(-4.0, -2.0), (2.0, 3.0)]
    assert cut == 3.0 and remaining == ()


def test_stop_at_cuts_after_tour_passing_p():
    tours, cut, remaining = greedy_cover(AXIS, (Segment(-4.5, 4.5),), L, stop_at=0.0)
    assert [(t.a, t.b) for t in tours] == [(-4.5, 0.5)]
    assert cut == 0.5
    assert remaining == (Segment(0.5, 4.5),)


def test_stop_at_before_first_segment_builds_nothing():
    tours, cut, remaining = greedy_cover(AXIS, (Segment(1.0, 2.0),), L, stop_at=0.5)
    assert tours == () and cut == 1.0 and remaining == (Segment(1.0, 2.0),)


def test_empty_segments():
    assert greedy_cover(AXIS, (), L) == ((), 0.0, ())


def test_one_mintours_rejects_infeasible_before_building_tours():
    d = Depot(0, 0.0, 1.0)
    _, mr = reachable_interval(d, L)
    with pytest.raises(ValueError, match="Infeasible"):
        solve_1_mintours(Instance((Segment(0.0, mr),), (d,), L))


def test_one_mintours_near_boundary_terminates():
    d = Depot(0, 1e6, 1.0)
    _, mr = reachable_interval(d, L)
    tours = solve_1_mintours(Instance((Segment(1e6, mr - 1e-6),), (d,), L))
    assert math.isclose(tours[-1].b, mr - 1e-6, abs_tol=1e-6)


def test_one_mintours_wants_one_depot():
    with pytest.raises(ValueError):
        solve_1_mintours(Instance((Segment(0.0, 1.0),), (AXIS, Depot(1, 1, 0)), L))


def test_trace_matches_tours():
    segs = (Segment(-4.0, -2.0), Segment(2.0, 3.0))
    tours, cut, remaining, steps = greedy_cover_traced(AXIS, segs, L)
    assert (tours, cut, remaining) == greedy_cover(AXIS, segs, L)
    assert len(steps) == len(tours)
    first, second = steps
    assert first.shrunk and first.raw_end == pytest.approx(1.0)
    assert first.end == -2.0 and first.next_start == 2.0
    assert second.next_start is None and not second.stopped_by_p


def test_trace_marks_stop_rule():
    *_, steps = greedy_cover_traced(AXIS, (Segment(-4.5, 4.5),), L, stop_at=0.0)
    assert len(steps) == 1
    assert steps[0].stopped_by_p and not steps[0].shrunk
    assert steps[0].next_start == pytest.approx(0.5)


def test_trace_consistent_on_random_instances():
    rng = np.random.default_rng(11)
    d = Depot(0, 0.0, 2.0)
    ml, mr = reachable_interval(d, L)
    for _ in range(200):
        pts = np.sort(rng.uniform(ml + 0.01, mr - 0.01, size=8))
        segs = tuple(Segment(float(a), float(b)) for a, b in pts.reshape(-1, 2))
        tours, _, _, steps = greedy_cover_traced(d, segs, L)
        for t, g in zip(tours, steps, strict=True):
            assert (g.start, g.end) == (t.a, t.b)
            assert g.reach == f(d, g.start, L)
            assert g.raw_end == g.start + g.reach
            assert g.shrunk == (g.end < g.raw_end - EPS)

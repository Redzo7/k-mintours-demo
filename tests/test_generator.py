import pytest
from helpers import assert_valid_solution

from drone_cover.generator import random_instance
from drone_cover.geometry import validate_instance
from drone_cover.kmintours import solve_k_mintours


def test_deterministic_per_seed():
    assert random_instance(3, 4, 6) == random_instance(3, 4, 6)
    assert random_instance(3, 4, 6) != random_instance(4, 4, 6)


@pytest.mark.parametrize("seed", range(60))
def test_feasible_and_solvable(seed):
    k, m = 1 + seed % 6, 1 + seed % 9
    inst = random_instance(seed, k, m)
    assert len(inst.depots) == k
    assert 1 <= len(inst.segments) <= m
    for a, b in zip(inst.segments, inst.segments[1:]):
        assert a.xr < b.xl
    validate_instance(inst)
    assert_valid_solution(inst, solve_k_mintours(inst))


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        random_instance(0, 0, 3)
    with pytest.raises(ValueError):
        random_instance(0, 2, 3, max_height_ratio=1.0)

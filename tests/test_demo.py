import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from drone_cover.demo import build_frames, draw_frame, main, make_view
from drone_cover.examples import EXAMPLES
from drone_cover.generator import random_instance
from drone_cover.kmintours import solve_k_mintours
from drone_cover.models import Instance

INSTANCES = {name: make() for name, make in EXAMPLES.items()} | {
    f"random{s}": random_instance(s, 5, 8) for s in range(3)
}


def expected_frame_count(sol) -> int:
    # instance + pruning + per step (equilibrium + tours) + cuts between + result
    trace = sol.trace
    return 2 + sum(1 + len(t.tours) for t in trace) + (len(trace) - 1) + 1


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_frames_and_drawing(name):
    inst = INSTANCES[name]
    sol = solve_k_mintours(inst)
    frames = build_frames(inst, sol)
    assert len(frames) == expected_frame_count(sol)
    assert frames[0].kind == "instance" and frames[1].kind == "prune"
    assert frames[-1].kind == "result"
    assert frames[-1].done_tours == sol.tours
    tour_frames = [fr for fr in frames if fr.kind == "tour"]
    assert tuple(fr.tour for fr in tour_frames) == sol.tours
    for fr in tour_frames:
        assert fr.raw_tour is None or fr.raw_tour.b > fr.tour.b

    view = make_view(inst, sol)
    fig, (ax, ax_text) = plt.subplots(2)
    try:
        for fr in frames:
            draw_frame(ax, ax_text, fr, inst, view)
            assert fr.caption
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_empty_instance_frames():
    inst = Instance((), (), 10.0)
    frames = build_frames(inst, solve_k_mintours(inst))
    assert [fr.kind for fr in frames] == ["instance", "result"]


def test_main_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in EXAMPLES)


def test_main_reports_bad_random_arguments(capsys):
    assert main(["--random", "--k", "0"]) == 1
    assert "error" in capsys.readouterr().err

"""Step-by-step visualisation of k-MinTours (Section 4 of the paper).

    python -m drone_cover.demo                    # showcase example
    python -m drone_cover.demo --example gap
    python -m drone_cover.demo --list
    python -m drone_cover.demo --random --seed 1 --k 4 --m 6

Navigation: Next/Prev buttons or the arrow keys, Home/End to jump.

The module only consumes what the algorithm returns (`Solution.trace`); it never
re-runs parts of the algorithm. `build_frames` produces plain data (testable
without a display) and `draw_frame` renders one frame with matplotlib.
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from .examples import EXAMPLES
from .generator import random_instance
from .geometry import dist, f, reachable_interval
from .kmintours import solve_k_mintours
from .models import EPS, Depot, Instance, Segment, Solution, StepTrace, Tour

# Categorical palette in fixed order (one hue per depot, in pruned x-order).
PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
INK = "#222222"
MUTED = "#8a8a85"
GRID = "#d9d9d4"
SURFACE = "#ffffff"
CAPTION_WIDTH = 112


@dataclass(frozen=True)
class Frame:
    """Everything one picture of the demo shows (no matplotlib objects)."""

    kind: str  # instance | prune | equilibrium | tour | cut | result
    title: str
    caption: str
    current: int | None = None  # depot id of D1 in this step
    active: tuple[int, ...] = ()  # depots still in play (pruned order)
    pruned: tuple[int, ...] = ()  # dominated depots (shown greyed out)
    show_pruned: bool = False  # mark pruned depots explicitly
    done_tours: tuple[Tour, ...] = ()  # tours fixed before this frame
    tour: Tour | None = None  # tour highlighted in this frame
    raw_tour: Tour | None = None  # right maximal tour before shortening
    eq: tuple[tuple[int, float], ...] = ()  # equilibrium points P_{1,j}
    p: float | None = None  # leftmost equilibrium point
    stop_at: float | None = None  # where the greedy of this step stops
    m: float | None = None  # cut point M
    remaining: tuple[Segment, ...] | None = None  # sub-instance still to cover


# --------------------------------------------------------------------------
# Frames (pure data)
# --------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _depot_by_id(instance: Instance) -> dict[int, Depot]:
    return {d.id: d for d in instance.depots}


def _interval_text(d: Depot, L: float) -> str:
    iv = reachable_interval(d, L)
    if iv is None:
        return f"D{d.id}: reaches nothing (h = {_fmt(d.h)} >= L/2)"
    return (
        f"D{d.id} ({_fmt(d.x)}, {_fmt(d.h)}): [Ml, Mr] = [{_fmt(iv[0])}, {_fmt(iv[1])}]"
    )


def _dominator(d: Depot, kept: tuple[int, ...], by_id: dict[int, Depot], L: float):
    """Some kept depot whose reachable interval contains d's (Prop. 8)."""
    iv = reachable_interval(d, L)
    if iv is None:
        return None
    for j in kept:
        jv = reachable_interval(by_id[j], L)
        if jv is not None and jv[0] <= iv[0] + EPS and iv[1] <= jv[1] + EPS:
            return j
    return None


def _instance_frame(instance: Instance) -> Frame:
    L = instance.L
    lines = [
        (
            f"{len(instance.depots)} depots, {len(instance.segments)} segments, "
            f"tour budget L = {_fmt(L)}."
        ),
        (
            "A point P is reachable from depot D iff d(D, P) <= L/2, so D reaches the "
            "interval [Ml, Mr] = xD -/+ sqrt(L²/4 - hD²) (Prop. 6), drawn as a bar "
            "below the axis."
        ),
        "   ".join(_interval_text(d, L) for d in instance.depots),
    ]
    return Frame(
        kind="instance",
        title="Instance",
        caption="\n".join(lines),
        active=tuple(d.id for d in instance.depots),
    )


def _prune_frame(instance: Instance, first: StepTrace) -> Frame:
    by_id = _depot_by_id(instance)
    L = instance.L
    kept, pruned = first.active_depots, first.pruned_depots
    lines = [
        (
            "Prop. 8: a depot whose reachable interval lies inside another depot's is "
            "dominated - every tour it can fly, the other can fly too. It is removed."
        )
    ]
    if pruned:
        parts = []
        for pid in pruned:
            j = _dominator(by_id[pid], kept, by_id, L)
            if j is None:
                parts.append(f"D{pid} (reaches nothing)")
            else:
                parts.append(f"D{pid} (inside D{j})")
        lines.append("Removed: " + ", ".join(parts) + ".")
    else:
        lines.append("No depot is dominated.")
    lines.append(
        "Cor. 9: after pruning, Ml and Mr both increase with x. Order: "
        + " < ".join(f"D{i}" for i in kept)
        + f". The algorithm treats D{kept[0]} as D1."
    )
    return Frame(
        kind="prune",
        title="Step 1 - remove dominated depots, sort by x",
        caption="\n".join(lines),
        active=kept,
        pruned=pruned,
        show_pruned=True,
    )


def _equilibrium_frame(
    instance: Instance,
    t: StepTrace,
    done: tuple[Tour, ...],
    pruned: tuple[int, ...],
    todo: tuple[Segment, ...],
) -> Frame:
    by_id = _depot_by_id(instance)
    L = instance.L
    c = t.current_depot
    dc = by_id[c]
    others = t.active_depots[1:]
    lines = [
        f"Current depot D1 = D{c}. Depots still in play: "
        + ", ".join(f"D{i}" for i in t.active_depots)
        + "."
    ]
    if not others:
        title = f"Recursion {t.step}: D{c} is the last depot"
        lines.append(
            f"Only D{c} is left (step 2): the greedy of Bereg et al. (Thm. 7) "
            "covers all remaining segments optimally."
        )
    elif t.p is None:
        iv = reachable_interval(dc, L)
        title = f"Recursion {t.step}: D{c} overlaps no other depot"
        lines.append(
            f"No remaining depot's interval overlaps D{c}'s, so there is no "
            "equilibrium point. Nobody else can reach the points up to "
            f"Mr(D{c}) = {_fmt(iv[1]) if iv else '?'}: D{c} covers them alone "
            "(deviation, see docs/implementation-notes.md)."
        )
    else:
        title = f"Recursion {t.step}: equilibrium points of D{c}"
        lines.append(
            "Def. 11: P_1j is the point where the right maximal tours of D1 and "
            "Dj starting there cover the same length, f(D1, P) = f(Dj, P). "
            "Left of P, D1's tours are longer; right of it, Dj's are."
        )
        eq_txt = []
        for j, x in t.equilibrium_points:
            eq_txt.append(
                f"P_{c},{j} = {_fmt(x)}  [f(D{c}) = {_fmt(f(dc, x, L))}, "
                f"f(D{j}) = {_fmt(f(by_id[j], x, L))}]"
            )
        lines.append("   ".join(eq_txt))
        lines.append(
            f"Step 3: P = leftmost = {_fmt(t.p)} (with D{t.p_depot}). "
            f"Step 4: D{c} runs the greedy until its tours pass P."
        )
    return Frame(
        kind="equilibrium",
        title=title,
        caption="\n".join(lines),
        current=c,
        active=t.active_depots,
        pruned=pruned,
        done_tours=done,
        eq=t.equilibrium_points,
        p=t.p,
        stop_at=t.stop_at,
        remaining=todo,
    )


def _tour_frame(
    instance: Instance,
    t: StepTrace,
    idx: int,
    done: tuple[Tour, ...],
    pruned: tuple[int, ...],
    todo: tuple[Segment, ...],
) -> Frame:
    L = instance.L
    g = t.greedy_steps[idx]
    tour = t.tours[idx]
    d = tour.depot
    c = d.id
    lines = [
        (
            f"Leftmost uncovered point s = {_fmt(g.start)}. "
            f"Prop. 3: f(D{c}, s) = (L² - 2L·d) / (2(L + s - xD - d)) with "
            f"d = d(D{c}, s) = {_fmt(dist(d, g.start))}  =>  f = {_fmt(g.reach)}, "
            f"so the right maximal tour ends at s + f = {_fmt(g.raw_end)}."
        )
    ]
    if g.shrunk:
        lines.append(
            f"{_fmt(g.raw_end)} lies in a gap (or past the last segment): the tour "
            f"is shortened to the end of the last segment it touches, b = "
            f"{_fmt(g.end)} (Thm. 7). Dashed: the maximal tour."
        )
    lines.append(
        f"Tour [{_fmt(tour.a)}, {_fmt(tour.b)}]: length |AB| + d(D, A) + d(D, B) = "
        f"{_fmt(tour.length)} <= L = {_fmt(L)}."
    )
    if d.h <= EPS:
        lines.append(
            f"D{c} is on the axis (h = 0): its tour is a flat triangle, "
            "flying out and back along the axis."
        )
    if g.stopped_by_p:
        limit = "P" if t.p is not None else f"Mr(D{c})"
        lines.append(
            f"The next tour would start at {_fmt(g.next_start or 0.0)} >= "
            f"{limit} = {_fmt(t.stop_at or 0.0)}: D{c} stops here (step 4)."
        )
    elif g.next_start is None:
        lines.append("All segments are covered.")
    elif g.shrunk:
        lines.append(f"Next tour starts at the next segment, x = {_fmt(g.next_start)}.")
    else:
        lines.append(f"Next tour starts right here, at x = {_fmt(g.next_start)}.")
    raw = Tour(d, g.start, g.raw_end) if g.shrunk else None
    return Frame(
        kind="tour",
        title=f"Recursion {t.step}: D{c}, tour {idx + 1} of {len(t.tours)}",
        caption="\n".join(lines),
        current=c,
        active=t.active_depots,
        pruned=pruned,
        done_tours=done,
        tour=tour,
        raw_tour=raw,
        eq=t.equilibrium_points,
        p=t.p,
        stop_at=t.stop_at,
        remaining=todo,
    )


def _cut_frame(t: StepTrace, done: tuple[Tour, ...], pruned: tuple[int, ...]) -> Frame:
    c = t.current_depot
    nxt = t.active_depots[1] if len(t.active_depots) > 1 else None
    lines = [
        (
            f"D{c} used {len(t.tours)} tour(s). Cut point M = right end of its last "
            f"tour = {_fmt(t.m)}."
        ),
    ]
    if t.remaining:
        lines.append(
            f"Step 5: recurse on the {len(t.remaining)} segment(s) right of M "
            f"(highlighted), with D{c} removed. D{nxt} becomes the new D1; "
            f"D{c}'s tours are prepended to the result."
        )
    else:
        lines.append("Nothing is left to cover.")
    return Frame(
        kind="cut",
        title=f"Recursion {t.step}: cut at M, recurse",
        caption="\n".join(lines),
        current=c,
        active=t.active_depots,
        pruned=pruned,
        done_tours=done,
        p=t.p,
        stop_at=t.stop_at,
        m=t.m,
        remaining=t.remaining,
    )


def _result_frame(instance: Instance, sol: Solution, pruned: tuple[int, ...]) -> Frame:
    L = instance.L
    counts: dict[int, int] = {}
    for tour in sol.tours:
        counts[tour.depot.id] = counts.get(tour.depot.id, 0) + 1
    longest = max((t.length for t in sol.tours), default=0.0)
    lines = [
        f"{sol.num_tours} tours in total (optimal, Section 4). Per depot: "
        + (", ".join(f"D{i}: {n}" for i, n in counts.items()) or "none")
        + ".",
        (
            f"Longest tour: {_fmt(longest)} <= L = {_fmt(L)}. "
            f"Expected running time O(m + n + k²)."
        ),
    ]
    return Frame(
        kind="result",
        title=f"Result: {sol.num_tours} tours",
        caption="\n".join(lines),
        active=tuple(d.id for d in instance.depots if d.id not in pruned),
        pruned=pruned,
        done_tours=sol.tours,
    )


def build_frames(instance: Instance, solution: Solution) -> list[Frame]:
    """The demo's frame sequence: instance, pruning, then per recursion step
    the equilibrium points, one frame per tour and the cut, then the result.
    The last recursion step has no cut frame."""
    frames = [_instance_frame(instance)]
    trace = solution.trace
    if not trace:
        frames.append(_result_frame(instance, solution, ()))
        return frames
    pruned = trace[0].pruned_depots
    frames.append(_prune_frame(instance, trace[0]))

    done: tuple[Tour, ...] = ()
    todo: tuple[Segment, ...] = instance.segments
    for n, t in enumerate(trace):
        frames.append(_equilibrium_frame(instance, t, done, pruned, todo))
        for i in range(len(t.tours)):
            frames.append(_tour_frame(instance, t, i, done, pruned, todo))
            done = done + (t.tours[i],)
        if n < len(trace) - 1:
            frames.append(_cut_frame(t, done, pruned))
        todo = t.remaining
    frames.append(_result_frame(instance, solution, pruned))
    return frames


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """Fixed geometry shared by all frames of one instance."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    colors: dict[int, str]
    rows: dict[int, int]  # row of each depot's reach bar below the axis
    bar_gap: float


def make_view(instance: Instance, solution: Solution) -> View:
    L = instance.L
    xs = [s.xl for s in instance.segments] + [s.xr for s in instance.segments]
    for d in instance.depots:
        xs.append(d.x)
        iv = reachable_interval(d, L)
        if iv is not None:
            xs.extend(iv)
    lo, hi = (min(xs), max(xs)) if xs else (0.0, 1.0)
    pad = 0.04 * (hi - lo or 1.0)
    order = solution.trace[0].active_depots if solution.trace else ()
    pruned = solution.trace[0].pruned_depots if solution.trace else ()
    ids = (
        list(order)
        + [i for i in pruned]
        + [d.id for d in instance.depots if d.id not in order and d.id not in pruned]
    )
    colors = {i: PALETTE[n % len(PALETTE)] for n, i in enumerate(order)}
    for i in ids:
        colors.setdefault(i, MUTED)
    top = max((d.h for d in instance.depots), default=1.0)
    bar_gap = max(top, L / 4) * 0.07
    rows = {i: n for n, i in enumerate(ids)}
    bottom = -(len(ids) + 1.5) * bar_gap
    return View(
        (lo - pad, hi + pad), (bottom, top * 1.18 + bar_gap), colors, rows, bar_gap
    )


def _triangle(ax: Axes, tour: Tour, color: str, alpha: float, lw: float, ls="-"):
    d = tour.depot
    ax.add_patch(
        Polygon(
            [(tour.a, 0.0), (d.x, d.h), (tour.b, 0.0)],
            closed=True,
            facecolor=color if ls == "-" else "none",
            alpha=alpha if ls == "-" else 1.0,
            edgecolor=color,
            linewidth=lw,
            linestyle=ls,
            zorder=2,
        )
    )


def draw_frame(
    ax: Axes, ax_text: Axes, frame: Frame, instance: Instance, view: View
) -> None:
    """Render one frame onto a plot axes and a caption axes."""
    ax.clear()
    ax_text.clear()
    L = instance.L
    ax.set_xlim(*view.xlim)
    ax.set_ylim(*view.ylim)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)

    active = set(frame.active)
    pruned = set(frame.pruned)

    # reach intervals [Ml, Mr] as bars below the axis (Prop. 6)
    for d in instance.depots:
        iv = reachable_interval(d, L)
        if iv is None:
            continue
        y = -(view.rows[d.id] + 1) * view.bar_gap
        out = (
            d.id in pruned
            and not frame.show_pruned
            or (frame.kind not in ("instance", "prune") and d.id not in active)
        )
        col = MUTED if d.id in pruned else view.colors[d.id]
        ax.plot(
            iv,
            (y, y),
            color=col,
            lw=3,
            alpha=0.25 if out else 0.8,
            solid_capstyle="round",
            zorder=1,
        )
        if not out:
            ax.text(
                iv[0], y, f"D{d.id} ", ha="right", va="center", fontsize=7, color=INK
            )

    # segments: covered/finished in ink, still-to-cover highlighted
    todo = frame.remaining
    for s in instance.segments:
        ax.plot((s.xl, s.xr), (0, 0), color=INK, lw=5, solid_capstyle="butt", zorder=3)
    if todo is not None:
        for s in todo:
            ax.plot(
                (s.xl, s.xr),
                (0, 0),
                color="#e34948",
                lw=2.2,
                solid_capstyle="butt",
                zorder=4,
            )

    # tours
    for t in frame.done_tours:
        _triangle(ax, t, view.colors[t.depot.id], 0.16, 1.0)
    if frame.raw_tour is not None:
        _triangle(
            ax, frame.raw_tour, view.colors[frame.raw_tour.depot.id], 1, 1.4, ls="--"
        )
    if frame.tour is not None:
        _triangle(ax, frame.tour, view.colors[frame.tour.depot.id], 0.42, 2.0)
        for x in (frame.tour.a, frame.tour.b):
            ax.plot([x], [0], marker="o", ms=5, color=INK, zorder=6)

    # depots
    for d in instance.depots:
        is_pruned = d.id in pruned
        gone = frame.kind not in ("instance", "prune", "result") and d.id not in active
        col = view.colors[d.id]
        if is_pruned:
            ax.plot(
                [d.x],
                [d.h],
                marker="X",
                ms=10,
                color=MUTED,
                zorder=7,
                markeredgecolor=SURFACE,
            )
        else:
            ax.plot(
                [d.x],
                [d.h],
                marker="o",
                ms=11 if d.id == frame.current else 8,
                color=SURFACE if gone else col,
                markeredgecolor=col,
                markeredgewidth=2,
                zorder=7,
            )
        if d.id == frame.current:
            ax.plot(
                [d.x], [d.h], marker="o", ms=18, mfc="none", mec=INK, mew=1.2, zorder=7
            )
        label = f"D{d.id}" + (" (dominated)" if is_pruned and frame.show_pruned else "")
        ax.annotate(
            label,
            (d.x, d.h),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=MUTED if is_pruned else INK,
            zorder=8,
        )

    # equilibrium points, P, stop limit and M
    ytop = view.ylim[1]
    for j, x in frame.eq:
        ax.plot(
            [x],
            [0],
            marker="v",
            ms=8,
            color=view.colors.get(j, MUTED),
            markeredgecolor=INK,
            markeredgewidth=0.6,
            zorder=6,
        )
    if frame.p is not None:
        ax.axvline(frame.p, color=INK, ls="--", lw=1.1, zorder=5)
        ax.text(frame.p, ytop, " P", va="top", fontsize=10, color=INK)
    elif frame.stop_at is not None and frame.kind != "result":
        ax.axvline(frame.stop_at, color=INK, ls=":", lw=1.1, zorder=5)
        ax.text(
            frame.stop_at,
            ytop,
            f" Mr(D{frame.current})",
            va="top",
            fontsize=10,
            color=INK,
        )
    if frame.m is not None:
        ax.axvline(frame.m, color=INK, lw=1.6, zorder=5)
        ax.text(frame.m, ytop * 0.9, " M", va="top", fontsize=10, color=INK)
        ax.axvspan(view.xlim[0], frame.m, color=GRID, alpha=0.35, zorder=0)

    legend = [
        Line2D([], [], color=INK, lw=5, label="segment"),
        Line2D([], [], color="#e34948", lw=2.2, label="still to cover"),
        Line2D([], [], color=INK, ls="--", label="P (leftmost equilibrium)"),
        Line2D([], [], color=INK, lw=1.6, label="cut point M"),
        Line2D([], [], color=INK, ls="none", marker="v", label="equilibrium P_1j"),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        fontsize=8,
        frameon=False,
        ncol=5,
        bbox_to_anchor=(0, 1.0),
    )

    ax_text.axis("off")
    wrapped = "\n".join(
        textwrap.fill(line, CAPTION_WIDTH) for line in frame.caption.splitlines()
    )
    ax_text.text(
        0,
        1,
        wrapped,
        va="top",
        ha="left",
        fontsize=10,
        color=INK,
        family="monospace",
        transform=ax_text.transAxes,
    )


# --------------------------------------------------------------------------
# Interactive window
# --------------------------------------------------------------------------


def run(instance: Instance, name: str = "") -> None:
    """Open a matplotlib window to step through the algorithm on `instance`."""
    from matplotlib.widgets import Button

    solution = solve_k_mintours(instance)
    frames = build_frames(instance, solution)
    view = make_view(instance, solution)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0.04, 0.33, 0.94, 0.55))
    ax_text = fig.add_axes((0.04, 0.07, 0.94, 0.22))
    ax_prev = fig.add_axes((0.78, 0.015, 0.09, 0.045))
    ax_next = fig.add_axes((0.88, 0.015, 0.09, 0.045))
    b_prev, b_next = Button(ax_prev, "< Prev"), Button(ax_next, "Next >")
    state = {"i": 0}

    def show() -> None:
        i = state["i"]
        fr = frames[i]
        draw_frame(ax, ax_text, fr, instance, view)
        prefix = f"{name}: " if name else ""
        fig.suptitle(
            f"{prefix}{fr.title}    [{i + 1}/{len(frames)}]",
            fontsize=13,
            color=INK,
            x=0.04,
            ha="left",
        )
        fig.canvas.draw_idle()

    def go(i: int) -> None:
        state["i"] = max(0, min(len(frames) - 1, i))
        show()

    b_prev.on_clicked(lambda _e: go(state["i"] - 1))
    b_next.on_clicked(lambda _e: go(state["i"] + 1))

    def on_key(event) -> None:
        keys = {"right": 1, "down": 1, " ": 1, "left": -1, "up": -1}
        if event.key in keys:
            go(state["i"] + keys[event.key])
        elif event.key == "home":
            go(0)
        elif event.key == "end":
            go(len(frames) - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(0.04, 0.03, "<- / -> to step, Home / End to jump", fontsize=8, color=MUTED)
    show()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m drone_cover.demo",
        description="Step-by-step visualisation of the k-MinTours algorithm.",
    )
    parser.add_argument("--example", default="showcase", choices=sorted(EXAMPLES))
    parser.add_argument("--list", action="store_true", help="list the examples")
    parser.add_argument(
        "--random",
        action="store_true",
        help="use a seeded random instance instead of an example",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=4, help="number of depots")
    parser.add_argument("--m", type=int, default=6, help="number of segments (max)")
    parser.add_argument("--L", type=float, default=10.0, help="tour budget")
    args = parser.parse_args(argv)

    if args.list:
        for key, make in EXAMPLES.items():
            doc = (make.__doc__ or "").strip().splitlines()[0]
            print(f"{key:<10} {doc}")
        return 0
    try:
        if args.random:
            inst = random_instance(args.seed, args.k, args.m, args.L)
            name = f"random seed={args.seed}"
        else:
            inst = EXAMPLES[args.example]()
            name = args.example
        run(inst, name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

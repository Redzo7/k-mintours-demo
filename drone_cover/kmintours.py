"""Recursive k-depot algorithm (Section 4)."""

from .dominance import prune_dominated_and_order_depots
from .equilibrium import equilibrium_points, leftmost_equilibrium
from .geometry import reachable_interval, validate_instance
from .greedy import greedy_cover
from .models import Instance, Solution, StepTrace, Tour


def solve_k_mintours(instance: Instance) -> Solution:
    """Optimal solution of k-MinTours (Section 4), with a per-step trace.

    1. Remove dominated depots, sort the rest by x (Prop. 8, Cor. 9).
    2. One depot left: greedy of Bereg et al. (Thm. 7).
    3. Otherwise P = leftmost equilibrium point P_{1,j} (Def. 11).
    4. Greedy from D1 until a tour's right endpoint M passes P.
    5. Recurse right of M with D2..Dk.

    The recursion is written as a loop. Raises ValueError for infeasible
    instances before building any tour.
    """
    validate_instance(instance)
    L = instance.L
    depots, pruned = prune_dominated_and_order_depots(instance.depots, L)
    segments = instance.segments

    all_tours: list[Tour] = []
    trace: list[StepTrace] = []
    step = 0

    if not segments:
        return Solution(tours=(), trace=())

    while segments:
        step += 1
        current, others = depots[0], depots[1:]
        active_ids = tuple(d.id for d in depots)

        eq: dict[int, float] = {}
        p: float | None = None
        p_depot: int | None = None
        stop_at: float | None = None
        if others:
            eq = equilibrium_points(current, others, L)
            p, p_depot = leftmost_equilibrium(eq)
            if p is not None:
                stop_at = p
            else:
                # Deviation: Section 4 assumes D1 overlaps some D_j. If it
                # overlaps none, D1 alone can cover everything up to Mr(D1).
                iv = reachable_interval(current, L)
                assert iv is not None  # pruning drops depots that reach nothing
                stop_at = iv[1]
        # else: base case (step 2), stop_at stays None and the greedy finishes

        new_tours, cut, remaining = greedy_cover(current, segments, L, stop_at)

        trace.append(
            StepTrace(
                step=step,
                current_depot=current.id,
                active_depots=active_ids,
                pruned_depots=tuple(pruned) if step == 1 else (),
                equilibrium_points=tuple(eq.items()),
                p=p,
                p_depot=p_depot,
                stop_at=stop_at,
                tours=new_tours,
                m=cut,
                remaining=remaining,
            )
        )
        all_tours.extend(new_tours)
        segments = remaining
        depots = others

    return Solution(tours=tuple(all_tours), trace=tuple(trace))

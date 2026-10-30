from .dominance import prune_dominated_and_order_depots
from .equilibrium import equilibrium_points, leftmost_equilibrium
from .geometry import reachable_interval
from .greedy import greedy_cover
from .models import Instance, Solution, StepTrace, Tour


# Optimal solution of k-MinTours
def solve_k_mintours(instance: Instance) -> Solution:
    L = instance.L
    depots, pruned = prune_dominated_and_order_depots(instance.depots, L)
    segments = list(instance.segments)

    all_tours: list[Tour] = []
    trace: list[StepTrace] = []
    step = 0

    if not segments:
        return Solution(tours=(), trace=())
    if not depots:
        raise ValueError("No usable depots")

    while segments:
        step += 1
        current, others = depots[0], depots[1:]
        active_ids = tuple(d.id for d in depots)

        if not others:                                   # base case
            eq: dict[int, float] = {}
            p = None
            p_depot = None
            new_tours, cut, remaining = greedy_cover(current, segments, L)
        else:
            eq = equilibrium_points(current, others, L)
            p, p_depot = leftmost_equilibrium(eq)
            if p is not None:
                stop_at = p
            else:
                # D1 overlaps no other depot: it handles everything it can reach
                stop_at = reachable_interval(current, L)[1]
            new_tours, cut, remaining = greedy_cover(current, segments, L, stop_at)

        trace.append(
            StepTrace(
                step=step,
                current_depot=current.id,
                active_depots=active_ids,
                pruned_depots=tuple(pruned) if step == 1 else (),
                equilibrium_points=dict(eq),
                p=p,
                p_depot=p_depot,
                tours=tuple(new_tours),
                m=cut,
                remaining=tuple(remaining),
            )
        )
        all_tours.extend(new_tours)
        segments = remaining
        depots = others

    return Solution(tours=tuple(all_tours), trace=tuple(trace))

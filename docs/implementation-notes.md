# Implementation notes for the course report

Places where the implementation fills a gap in, or departs from, the paper's text.

## D1 overlaps no other depot
Section 4 assumes that D1 has an equilibrium point with some later depot. If D1's
reachable interval overlaps no other depot's, the greedy from D1 runs until it has
covered everything up to Mr(D1) (`kmintours.py`, `stop_at = Mr(D1)`). No other
depot can reach those points, so this is forced. `StepTrace.stop_at` records it.

## Stop rule "until the right endpoint M passes P"
The implementation stops when the next tour would *start* at or right of P.
Without gaps this is the same thing. When a tour is shrunk at a gap, M is the end
of the last segment it touched (possibly < P), and the next segment starts right
of P, so D2 takes over from there.

## Strict reachability, numerically
The paper requires d(D, P) < L/2 for every segment point. Otherwise the greedy needs
infinitely many tours as it approaches Mr. `geometry.validate_feasible` checks this
up front, treating "strictly" as "at least EPS = 1e-9 inside some [Ml, Mr]".
Before this check, infeasible inputs either failed late with a misleading error or,
after a translation to x ≈ 1e6, silently returned a "solution".

## Depots on the axis (h = 0)
For h = 0, f(D, ·) jumps from 0 at Ml(D) (unreachable by strictness) to L/2 just
right of it. The equilibrium point can then sit at Ml(Dj)⁺, and the optimum is
unchanged. For small h > 0, f(Dj, ·) is extremely steep near Ml(Dj), so tests check
the sign change of f(Di) − f(Dj) around P rather than |f(Di,P) − f(Dj,P)|.

## Typos already handled (see CLAUDE.md)
The sign in the proof of Prop. 3/4, f = f vs. f = g in Def. 11 / Prop. 12, and
uniqueness of the equilibrium, which is verified numerically in `tests/test_equilibrium.py`.

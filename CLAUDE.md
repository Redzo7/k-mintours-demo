# CLAUDE.md

Guidance for Claude when working in this repository.

## Project

University course project: a Python demonstrator (and later a statistical
analysis) of the k-MinTours algorithm from Beaudou, Foucaud, Lorieau, Misra —
*An efficient algorithm for a drone multidepot segment covering problem*.
The goal is clarity and correctness for teaching, not raw performance.

## Environment

- Python 3.13, conda environment `drone-cover` defined in `environment.yml`
  (channel: conda-forge).
- Always run commands inside the environment: `conda activate drone-cover`.
- Add new dependencies to `environment.yml` (not ad-hoc `pip install`), then
  `conda env update -f environment.yml --prune`. Ask before adding packages
  beyond the existing scientific stack.

## Commands

```bash
pytest                      # run tests
pytest tests/test_x.py -k name   # single test
ruff check src tests        # lint
ruff format src tests       # format
python -m drone_cover.demo  # visual demonstration
jupyter lab                 # notebooks
```

## Layout

```
drone_cover/
  geometry.py     # reachability, Ml/Mr (Prop. 6), f and g (Props. 3–4)
  dominance.py    # dominated-depot removal, sorting by x (Prop. 8, Cor. 9)
  equilibrium.py  # equilibrium points between depots (Def. 11, Prop. 12)
  greedy.py       # 1-depot greedy algorithm of Bereg et al. (Thm. 7)
  kmintours.py    # recursive k-depot algorithm (Section 4)
  generator.py    # random instance generation for experiments
  demo.py         # step-by-step matplotlib visualisation
notebooks/        # interactive demos, statistical analysis
tests/            # pytest suite
```

Keep the algorithm modules free of plotting code; visualisation lives in
`demo.py` and notebooks and consumes the data the algorithm returns.

## Problem model (from the paper)

- Segments: disjoint intervals `[xl, xr]` on the x-axis, sorted, `xl < xr`.
- Depot: point `(xD, hD)`. Tour: triangle with apex at the depot and covering
  side `[A, B]` on the x-axis; length = `|AB| + d(D,A) + d(D,B) <= L`.
- Point P is reachable from D iff `d(D, P) <= L/2`.
  `Ml(D), Mr(D) = xD ∓ sqrt(L²/4 − hD²)`.
- Instance is feasible only if every point of every non-degenerate segment is
  at distance strictly less than L/2 from some depot. Validate this and raise
  a clear error otherwise (otherwise the greedy loops forever).
- `f(D, P)`: x-length covered by the right maximal tour with left endpoint P.
  `g(D, P)`: same, leftwards, with right endpoint P.

## Algorithm (Section 4)

1. Remove dominated depots (`S_D' ⊂ S_D`), sort remaining by x.
2. One depot left → greedy of Bereg et al.: from the leftmost uncovered point,
   take right maximal tours; if a tour's right endpoint falls in a gap, shrink
   it to the end of the last segment it touches and restart at the next
   segment's left endpoint.
3. Otherwise compute equilibrium points `P_{1,j}` between D1 and every other
   depot; P = leftmost one.
4. Run the greedy from D1 until a tour's right endpoint M passes P.
5. Recurse on the part of the instance right of M with depots D2..Dk;
   prepend D1's tours.

Expected complexity: O(m + n + k²).

## Known issues in the paper — follow these, not the text

- **Prop. 3 proof**: last line has `L² + 2L·√…`; the correct numerator is
  `L² − 2L·√((xP−xD)² + hD²)`, as in the statement. Same sign for g (Prop. 4).
- **Def. 11 vs Prop. 12**: Def. 11 says `f(Di,P) = f(Dj,P)`, the proof of
  Prop. 12 uses `f(Di,P) = g(Dj,P)`. Implement `f(Di,P) = f(Dj,P)` (both
  right-going tours from the same left endpoint), which is what Lemma 13 and
  the algorithm use. Verify uniqueness numerically in tests rather than
  assuming it.
- Typos ("tours" for "depots" in Section 4, `(Di, L, R)` for `(Di, Q, R)` in
  Lemma 13) — treat by intent.
- If implementation and paper disagree in some other way, flag it to the user
  rather than silently choosing; it belongs in the course report.

## Conventions

- Type hints everywhere; small frozen dataclasses for `Segment`, `Depot`,
  `Tour`, `Instance`, `Solution`.
- Floating point: use a single module-level tolerance (e.g. `EPS = 1e-9`) for
  geometric comparisons; never compare floats with `==`.
- Docstrings reference the paper item they implement (e.g. "Prop. 3").
- Every algorithm module gets tests. Useful checks:
  - each returned tour has length ≤ L + EPS and its depot reaches both ends;
  - the union of tours covers all segments;
  - closed-form f/g match the perimeter equation;
  - on small instances, the result is no worse than a brute-force / MILP
    or a naive baseline (e.g. greedy from the nearest depot).
- Seed all randomness (`numpy.random.default_rng(seed)`) so experiments are
  reproducible.
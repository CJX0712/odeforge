# ODEForge architecture

Author: 晨星 (Chen Xing) <CJX0712@users.noreply.github.com>

## Layering

The dependency graph is strictly one-directional and acyclic:

```
cli.py
  |
  v
pipeline/                 ODEForgePipeline
  |
  +--> data/              synthetic system family
  +--> integrators/       euler rk4 dp54 vverlet yoshida4 yoshida6 symp_adaptive
  +--> routing/           hamiltonian detector + SympFuse vote
  +--> autodiff/          tape, vector, adjoint
  +--> training/          differentiable parameter identification
  +--> hpo/               grid search, coordinate descent, budget alignment
  +--> eval/              metrics, invariants, baselines, budget, gates
        |
        v
      core/               types, errors, config, interfaces, seed
```

Nothing below `pipeline` imports `pipeline`. `core` imports nothing from ODEForge at
all, so the seed registry and the error hierarchy cannot acquire a cycle.

`eval/budget.py` exists because `hpo` and `integrators` alone cannot express "spend
exactly N evaluations" for an adaptive solver; it depends on `integrators` and
`eval.baselines` and is depended on by `eval.benchmarks`.

## Data flow of a single solve

1. `core.seed.set_all(cfg.seed)` -- the only randomness entry point.
2. `routing.hamiltonian.HamiltonianDetector.analyse(spec)` -- three indicators, each
   normalised to `[0, 1]`.
3. `routing.sympfuse.SympFuse.route(spec)` -- three-way vote, separable check, and the
   SciPy delegation decision.
4. `integrators.*` -- solve, returning a `Trajectory` with `y` shaped
   `(n_points, n_states)`.
5. `eval.metrics` / `eval.invariants` -- metrics and invariant checks.

## The axis contract

`Trajectory.y` is `(n_points, n_states)`: **rows are time points**. This is the transpose
of `scipy.integrate.solve_ivp`'s `sol.y`, and the transposition happens in exactly one
place, `eval.baselines.solve_ivp_trajectory`.

Mixing the two produces a plausible-looking but meaningless energy series: on a
`(n_states, n_points)` array, `A[..., 0]` selects a single time point, so
`0.5*(A[...,0]**2 + A[...,1]**2)` has `n_states` entries rather than `n_points`, and any
`mean`/`sum` over it silently describes one instant. `tests/test_axis_convention.py`
locks this down in five tests, including one that asserts the resulting broadcast error.

## Numerical decisions and their evidence

| decision | evidence |
|---|---|
| Ship canonical Yoshida tables, not derived ones | a least-squares order-8 search found no solution; an order-6 search converged to residual `7.9e-02` and measured order **2.42** |
| Yoshida4 has **three** stages | the two-stage composition measured order **-0.002** |
| Force reuse across stages (`k` evals/step, not `2k`) | bit-identical trajectories; measured `n_fev = 1 + k*n_steps` |
| Reference solutions come from the symplectic family | self-disagreement: `yoshida6` **1.15e-10** vs `dp54` 6.88e-05 vs `DOP853` 5.38e-03 |
| Cubic, not linear, interpolation in `trajectory_error` | linear interpolation of DOP853's 617-point Lorenz grid carries **4.9e-02** error; the two solvers actually disagree by 1.5e-09 |
| Linear-Hamiltonian-structure test, not "Jacobian is symmetric" | the harmonic oscillator's Jacobian is **antisymmetric** (`A + A^T = 0`); the shortcut is simply false |
| Equal-`n_fev` alignment via bisection on `rtol` plus `max_step` | tolerance-only search leaves DOP853 at 3494 evaluations against a 20000 target |

## Determinism

`core.seed.set_all` is the single entry point. Probe points for the Hamiltonian detector
come from `numpy.random.default_rng` seeded through `core.seed.rng(offset=...)`, and
both the seed and the probe count are recorded in every report. Two demo runs agree
bit-for-bit on every field except `wall_sec`; `tools/check_determinism.py` asserts it and
runs in CI.

## Extending

- **A new integrator**: add a module under `integrators/`, return a `Trajectory`, and
  register the name in `integrators.ALL_METHODS` and `hpo.search.budget_steps_for` (with a
  *measured* per-step cost, not an assumed one).
- **A new system**: add a `SystemSpec` to `data/systems.py` with `rhs`, `y0`, `t_span`,
  and `grad_v` if it is separable. The detector and gates pick it up automatically.
- **A new invariant**: add a `ConservedQuantity` to the spec and an expectation to
  `eval.invariants.INVARIANT_EXPECTATIONS` whose tolerance reflects the *routed* solver,
  not the analytic exactness of the quantity.

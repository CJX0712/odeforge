# Changelog

All notable changes to ODEForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-04

Initial release. Author: 晨星 (Chen Xing) <CJX0712@users.noreply.github.com>.

### Added

**Integrators (pure numpy, no downloads required)**

- `euler`, `rk4` -- fixed-step baselines.
- `vverlet` (2nd order), `yoshida4` (3-stage composition, order 4),
  `yoshida6` (7-stage composition, order 6) -- symplectic family for separable
  Hamiltonians.
- `dp54` -- hand-written Dormand-Prince 5(4) with the full 7x6 Butcher tableau, FSAL,
  and a PI step controller. Verified order conditions: `b5.c^k = 1/(k+1)` for k<=4 and
  `b4.c^k = 1/(k+1)` for k<=3; `A[6] == b5[:6]`.
- `symplectic_adaptive` -- variable-step symplectic integration by uniform substep
  refinement, which preserves the symplectic structure (a composition of maps that share
  one substep width is still an exact symplectic map).
- Force reuse across composition stages: a `k`-stage composition costs `k` force
  evaluations per step, not `2k`. Verified bit-identical trajectories.

**Routing**

- Multi-indicator Hamiltonian detector: involutivity `f(x) == -f(-x)`, divergence-free
  consistency, and the rigorous linear-Hamiltonian-structure test (existence of a
  symmetric `S` with `A^T S + S A = 0`). All three must agree before a system is called
  Hamiltonian.
- `SympFuse` three-way vote (Hamiltonian structure / conserved-quantity monitoring /
  Jacobian structure), symmetric, deterministic, degrading to `dp54` on ties and
  conflicts.
- SciPy delegation on non-Hamiltonian systems, where the flagship's own 5th-order
  solver is structurally outmatched by DOP853's 8th order.

**Differentiable integration**

- `autodiff.tape` -- reverse-mode taping with vjp closures over 15 primitive
  categories. All gradient categories agree with central differences to better than
  `1e-10` (gate: `1e-6`).
- `autodiff.adjoint` -- exact discrete adjoint of one RK4 step, O(1) adjoint memory.
  Verified against finite differences of the discretised objective at `6e-08`.
- `training.task` -- damped-oscillator parameter identification from noisy
  observations, optimised through the tape. Loss `2.67e-01 -> 1.06e-03`.

**Evaluation**

- `net_drift` and `max_deviation` reported separately, plus an `interpolation_error`
  self-test so a reader can tell whether a trajectory-error number is meaningful.
- `eval.budget` -- equal-`n_fev` budget alignment for both sides of every comparison.
- Ten synthetic systems with difficulty knobs, declared invariants and separability
  flags.

### Known limitations

- **G2 (equal-budget trajectory accuracy) does not pass.** At genuinely aligned budgets
  the flagship's fixed-step 6th-order symplectic composition loses to adaptive 8th-order
  DOP853: 227x on the double well, 1.15x on the pendulum. The trade is deliberate and
  documented rather than hidden -- see `BLOCKED.md` and `docs/model_card.md`.
- Robertson A/B/C ships with `t_span = [0, 1e2]`, not the textbook `1e11`. An explicit
  method needs of order `1e7` steps for the full span; the limitation is measured and
  recorded in `data/systems.py`.

### Fixed during development (all regression-tested)

1. Generic velocity-Verlet acceleration had the wrong sign, giving `a = +q` and a blowup
   to `1e74`.
2. `budget_steps_for` charged `k+1` evaluations per step instead of `k`, denying the
   flagship 12.5% of its budget.
3. `backward()` never assigned the seed adjoint, so every gradient was exactly zero.
4. The training loss stored plain floats, severing the tape from the trajectory.
5. `_accum` raised on a shape mismatch, aborting the reverse sweep silently upstream.
6. The adjoint recursion used `lam - M^T lam` instead of `M^T lam` and accumulated the
   parameter term after the pull-back.
7. The reference solution for gates was roundoff-limited, making G2 compare noise.
8. Linear interpolation of adaptively-stored grids injected `4.9e-02` of error on
   Lorenz, five orders of magnitude above the solver disagreement being measured.
9. G2/G3 compared solvers at unmatched `n_fev` (3.3x apart), invalidating both.

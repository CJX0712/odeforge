# ODEForge

**Continuous-time dynamics numerical solving and differentiable integration.**
Pure numpy. No downloads at runtime. Every number in this README was produced by running
the code on this machine.

[![CI](https://github.com/CJX0712/odeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/odeforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/CJX0712/odeforge/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-C%20(blocked%3A%20G2)-orange.svg)](BLOCKED.md)

---

## What it does

- **Integrators**, all hand-written: explicit Euler, RK4, Dormand-Prince 5(4) with the
  full 7x6 Butcher tableau + FSAL + a PI step controller, and a symplectic family
  (velocity-Verlet, Yoshida order 4 and 6) with exact force reuse.
- **Structural routing** (`SympFuse`): a deterministic three-way vote decides whether a
  system is Hamiltonian, separable and worth integrating with a symplectic method -- and
  delegates to SciPy when it is not.
- **Differentiable integration**: reverse-mode taping and an exact discrete adjoint, both
  verified against central differences to better than `1e-6`.
- **Honest evaluation**: equal-`n_fev` budget alignment, calibrated references, and an
  interpolation self-test so a reader can tell when an accuracy number is meaningful.

## Install

```bash
pip install -r requirements.lock.txt     # pinned, no local paths
```

## Quick start

```bash
python examples/run_demo.py     # end-to-end -> results/benchmark.json  (~23 s)
python cli.py info              # environment and solver inventory
python cli.py run double_well   # route + solve one system
python cli.py invariants        # verify declared invariants
python cli.py benchmark         # full acceptance gates
```

## Measured results

Python 3.13.14 / numpy 2.5.3 / scipy 1.18.1, seed `20260101`, budget `n_fev = 120000`.

**Convergence orders** (harmonic oscillator, error vs h):

| method | empirical | nominal |
|---|---|---|
| `euler` | 0.97 | 1 |
| `vverlet` | 2.0017 | 2 |
| `yoshida4` | 4.0030 | 4 |
| `yoshida6` | 6.0004 | 6 |
| `rk4` | 3.9458 | 4 |

**Gradient correctness** (max relative error vs central difference, gate `1e-6`):

| quantity | error |
|---|---|
| tape, 15 primitive categories | `3.5e-10` |
| training loss through RK4 | `2.4e-11` |
| adjoint, exact discrete adjoint of RK4 | `6.0e-08` |

**Acceptance gates:**

| gate | claim | measured | threshold | verdict |
|---|---|---|---|---|
| **G1** | long-time energy conservation vs tuned DOP853 | `4.28e-13` vs `9.20e-10` -- ratio `4.65e-04`, **2150x better** | <= 0.10 | **PASS** |
| **G2** | equal-`n_fev` trajectory accuracy on **non-Hamiltonian** systems (non-inferior) | ratio **1.1952** (Van der Pol), **1.0238** (Lorenz); budgets aligned to 0.4% | <= 1.0 | **FAIL** |
| **G3** | non-Hamiltonian regression guard, delegated to SciPy | ratio **1.0000**, `n_fev` exactly equal | <= 1.05 | **PASS** |

**G2 fails by 2-4%, and is not hidden.** G1 and G2 measure deliberately different
mathematical objects, and the split matters:

* **G1** runs on **Hamiltonian** systems and asks whether *energy* stays conserved over
  long time. There the flagship wins by three orders of magnitude, because a symplectic
  method's energy error is bounded and oscillatory rather than secular.
* **G2** runs on **non-Hamiltonian** systems and asks whether the *trajectory* is locally
  as accurate. No structural advantage exists there, and the honest expectation is
  parity. Measured at genuinely equal budgets: **1.1952** (Van der Pol) and **1.0238**
  (Lorenz) -- a 2.4% to 19.5% deficit, not a rout of magnitude.

Getting G2 to mean anything required fixing the *yardstick*, not the solvers. A ratio
this close to 1 is only meaningful if the reference out-resolves both candidates, and a
naive reference does not: a single DOP853 run at `rtol=1e-13` on Lorenz spends 10070
evaluations -- **fewer** than the 19994 the DOP853 candidate spends -- so it is less
accurate than the thing it judges. Judged against that yardstick the same comparison
reads **52.14**; judged against a reference cross-validated between DOP853 (explicit RK)
and Radau (implicit BDF), which agree to `1.7e-11`, it reads **1.0238**. The converged
figure does not move across four decades of reference tolerance, and every report row
states whether the candidate errors clear the reference's own disagreement
(`reference.resolved`).

The full investigation, including the earlier mis-targeted measurement and four failed
remediation attempts with their numbers, is in [`BLOCKED.md`](BLOCKED.md).

## Why force reuse matters

In a composition `S(c_k h) ... S(c_0 h) ... S(c_k h)`, consecutive velocity-Verlet stages
share the boundary position, and the acceleration `-dV/dq` depends only on `q`. Carrying
it across the boundary makes a `k`-stage composition cost **`k`** force evaluations per
step instead of `2k` -- Yoshida6 drops from 14 to 7. Verified bit-identical
trajectories. This is what lets a 6th-order method run at a usable step size under a
fixed evaluation budget.

## What is here

```
core/         types (axis contract), errors E1xx-E5xx, config, protocols, seed
data/         10 synthetic systems with difficulty knobs and declared invariants
integrators/  fixed (euler rk4) | adaptive (dp54) | symplectic (vverlet yoshida4 yoshida6)
              symplectic_adaptive (variable step by uniform substep refinement)
routing/      hamiltonian (3 indicators) + sympfuse (3-way vote, SciPy delegation)
autodiff/     tape (reverse mode) | vector | adjoint (exact discrete adjoint)
training/     damped-oscillator parameter identification through the tape
hpo/          grid search, coordinate descent, measured per-step cost table
eval/         metrics, invariants, tuned baselines, budget alignment, gates
pipeline/     ODEForgePipeline: route -> solve -> evaluate -> report
cli.py        run / benchmark / invariants / demo / info
```

Strictly one-directional: `cli -> pipeline -> {data, integrators, routing, autodiff,
training, hpo, eval} -> core`. `core` imports nothing from ODEForge.

## Development

```bash
make lint          # ruff check -- HARD GATE, no || true
make test          # pytest with coverage
make check         # both
make demo-twice    # runs the demo twice and asserts bit-for-bit determinism
```

Ruff is a hard gate in `Makefile` and in CI (Python 3.12 and 3.13). There is no
`continue-on-error` and no `|| true` in the workflow.

## Two bugs worth reading about

Both are regression-tested, and both produced numbers that looked entirely reasonable.

**The axis bug.** `scipy.integrate.solve_ivp` returns `sol.y` as `(n_states, n_points)`.
Writing `0.5*(A[...,0]**2 + A[...,1]**2)` on that array indexes the *last* axis and
returns `n_states` values instead of `n_points` -- one time point, not a series. Any
`mean` or `sum` over it then reports a plausible-looking constant. ODEForge stores
`(n_points, n_states)` everywhere and transposes in exactly one place;
`tests/test_axis_convention.py` pins it, including a test that asserts the broadcast
error the wrong indexing produces.

**The sign bug.** The first generic velocity-Verlet computed the acceleration as
`-d/dq f1(q,0) + (f1(q,p) - f1(q,0))`, which for the harmonic oscillator gives `a = +q`
and diverges to `1e74` by `T=500`. The correct identity is
`a = d/dt f0 = J_{f0} . f`. It was caught by running the numerics before writing the
library, not by reading the code.

## License

MIT -- see [LICENSE](LICENSE). Author: 晨星 (Chen Xing) <CJX0712@users.noreply.github.com>.

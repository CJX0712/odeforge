# ODEForge model card

Author: 晨星 (Chen Xing) <CJX0712@users.noreply.github.com>

## What this is

ODEForge is a **numerical component library**, not a trained model. It solves initial
value problems and differentiates through the solve. There are no learned weights, no
training data, and no inference-time distribution shift. The "model" in the usual sense
does not apply; what follows describes the numerical claims instead.

## Measured performance

All numbers produced by `examples/run_demo.py` on Python 3.13.14 / numpy 2.5.3 /
scipy 1.18.1, seed 20260101, budget `n_fev = 120000`.

### Convergence orders (harmonic oscillator, error vs h)

| method | empirical | nominal |
|---|---|---|
| `euler` | 0.97 | 1 |
| `vverlet` | 2.0017 | 2 |
| `yoshida4` | 4.0030 | 4 |
| `yoshida6` | 6.0004 | 6 |
| `rk4` | 3.9458 | 4 |

### Gradient correctness

| quantity | max relative error vs central difference | gate |
|---|---|---|
| tape, 15 primitive categories | 3.5e-10 | 1e-6 |
| training loss (through RK4) | 2.4e-11 | 1e-6 |
| adjoint (exact discrete adjoint of RK4) | 6.0e-08 | 1e-6 |

### Acceptance gates

| gate | claim | measured | threshold | verdict |
|---|---|---|---|---|
| **G1** | Hamiltonian long-time energy conservation | double well **4.28e-13** vs tuned DOP853 **9.20e-10**, ratio **4.65e-04** (2150x better); pendulum **0.0** vs **6.29e-10** | <= 0.10 | **PASS** |
| **G2** | equal-`n_fev` trajectory accuracy on non-Hamiltonian systems, non-inferior | **1.1952** (Van der Pol, `nfev` 20047 vs 19982); **1.0238** (Lorenz, 19981 vs 19994) | <= 1.0 | **FAIL** |
| **G3** | non-Hamiltonian regression guard (delegated to SciPy) | Van der Pol **1.0000**; Lorenz **1.0000**, `n_fev` exactly equal (19982/19982, 19994/19994) | <= 1.0526 | **PASS** |

## Why G2 fails, stated plainly

G1 and G2 measure **different mathematical objects**, and conflating them was the first
defect found in review:

| gate | family | question | flagship |
|---|---|---|---|
| G1 | Hamiltonian | does *energy* stay conserved over long time? | **2150x better** |
| G2 | non-Hamiltonian | is the *trajectory* locally as accurate? | **2-4% worse** |

G2 therefore runs on Van der Pol and Lorenz. At genuinely equal budgets (aligned to
within 0.4% on both sides) the flagship's own 5th-order `dp54` is **1.1952** and
**1.0238** times the tuned 8th-order DOP853's error. That is a 2.4% to 19.5% deficit --
an honest FAIL, reported at that precision rather than relaxed away.

The measurement itself needed work to be worth anything, because a ratio this close to 1
is only meaningful if the reference out-resolves both candidates. Building the reference
by tightening `rtol` does not: the adaptive error estimate has an `eps*|y|` floor, so
`rtol=1e-13` on Lorenz spends 10070 evaluations -- *fewer* than the 19994 the DOP853
candidate spends. Against that reference the same comparison reads **52.14**; against a
reference cross-validated between DOP853 (explicit RK) and Radau (implicit BDF), which
agree to `1.7e-11`, it reads **1.0238**, unchanged across four decades of reference
tolerance. Every row reports `reference.resolved`, and the shipped configuration resolves
on both systems by three to four orders of magnitude.

Reaching parity would need either an order-8 symplectic composition (attempted; no valid
coefficient table found) or SciPy delegation, which is what the library actually does on
these systems. See `BLOCKED.md`.

## Comparison discipline

- Both sides of every accuracy comparison are solved through `eval.budget`, which pins
  each to the target `n_fev` and reports the achieved count. An earlier revision compared
  a flagship that spent 4321 evaluations against a baseline that spent 1310; that
  comparison was meaningless and is the reason the module exists.
- `trajectory_error` resamples with a cubic spline and exposes
  `interpolation_error(ref)`, so a reader can check that the yardstick is finer than the
  quantity being measured.
- Gate references are calibrated, not assumed: the tightest tolerance the solver can
  actually deliver is selected, and solutions that hit the roundoff floor are rejected.

## Ablation equivalence (important)

The five ablations are **not** five independent experiments. `no_cons` (conserved-quantity
monitoring disabled) and `dp54_only` (routing removed entirely) produce identical
numbers, because the conserved-quantity monitor only ever *adds a vote* to the router --
removing it changes the vote tally, but on the double well the remaining votes already
send the system to `dp54`, so both variants execute the same code path. Reading their
equal values as "two independent confirmations" would be a mistake. The genuinely
distinct arms are `full`, `no_ham` (Yoshida6 -> Yoshida4) and `dp54_only`.

## Intended use and limits

**Use it for**: long-time integration of separable Hamiltonian systems where energy
conservation is the correctness criterion; differentiable integration for parameter
identification; a routing decision between symplectic and adaptive methods.

**Do not use it for**: stiff problems where an implicit method is required (Robertson at
`T=1e11` needs `1e7` steps from any explicit method); trajectory accuracy on long chaotic
horizons, where the flagship is outmatched by DOP853; problems where the system is
Hamiltonian but not separable (Kepler, three-body, Lotka-Volterra) without first
transforming to canonical coordinates.

## Known limitations

1. G2 fails; the trade-off is documented above.
2. Robertson ships with `t_span = [0, 1e2]` rather than the textbook `1e11`.
3. The Hamiltonian detector needs a `SystemSpec.grad_v` for the separable family; without
   it, Kepler and the three-body problem are correctly detected as Hamiltonian but
   correctly routed to `dp54`.
4. `dp54` is a 5th-order method. On non-Hamiltonian systems ODEForge delegates to SciPy
   rather than compete; the delegation is explicit and reported, not silent.

## Reproducing

```bash
pip install -r requirements.lock.txt
python examples/run_demo.py
python -m pytest -W ignore::UserWarning --cov=core --cov=data --cov=integrators \
    --cov=routing --cov=autodiff --cov=training --cov=hpo --cov=eval --cov=pipeline
```

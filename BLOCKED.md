# BLOCKED: G2 does not pass

**Status: C-grade. One acceptance gate fails on measured evidence. Nothing was weakened
to hide it.**

Author: 晨星 (Chen Xing) <CJX0712@users.noreply.github.com>
Date: 2026-10-04
Reproduce: `python examples/run_demo.py`

## The blocked item

| item | claim | measured | threshold | verdict |
|---|---|---|---|---|
| **G2** | equal-`n_fev` trajectory accuracy on **non-Hamiltonian** systems: flagship error / tuned-baseline error | **1.1952** (Van der Pol), **1.0238** (Lorenz) | <= 1.0 | **FAIL** |

G2 misses by **2.4% to 19.5%**. That number is the honest result and is deliberately not
rounded away, relaxed, or reframed. What follows explains how it was obtained and why it
is trustworthy. G1 and G3 pass (see `docs/model_card.md`).

## What G2 measures, and why the earlier number was the wrong object

The first version of this file reported G2 as **370.13**. That number was real, and it
was measuring the wrong thing. G2 had been configured on the same Hamiltonian systems as
G1 (:data:`eval.benchmarks.GATE_SYSTEMS`), so it was comparing:

* a **fixed-step 6th-order symplectic** composition against
* an **adaptive 8th-order DOP853**,

over `T = 500` on a Hamiltonian problem, using **trajectory** error.

There the flagship's bounded, oscillatory energy error is irrelevant, and long-horizon
**phase drift** dominates -- the two methods are answering a phase-accumulation question,
not an accuracy question. The two gates measure different mathematical objects:

| gate | system family | question | flagship result |
|---|---|---|---|
| G1 | Hamiltonian (`G1_SYSTEMS`) | does energy stay conserved over long time? | **2150x better** |
| G2 | non-Hamiltonian (`G2_SYSTEMS`) | is the trajectory locally as accurate? | **2-4% worse** |

G2 now runs on Van der Pol and Lorenz, where no structural advantage exists and parity
is the honest expectation.

## Measured vs threshold, with the budgets actually aligned

Both sides are solved through `eval.budget`, which pins each to the target and reports
the achieved count. Budget alignment is a prerequisite -- an earlier revision compared a
flagship that spent 4321 evaluations against a baseline that spent 1310, which made the
comparison meaningless.

| system | target n_fev | flagship n_fev | baseline n_fev | n_fev ratio | flagship error | baseline error | **ratio** | threshold |
|---|---|---|---|---|---|---|---|---|
| van_der_pol | 20000 | 20047 | 19982 | 0.9968 | 9.3037e-10 | 7.7840e-10 | **1.1952** | 1.0 |
| lorenz | 20000 | 19981 | 19994 | 1.0007 | 3.4146e-07 | 3.3341e-07 | **1.0238** | 1.0 |

Both rows are within 0.4% of the target budget on **both** sides, so the residual gap is
a property of the methods and not of the accounting.

## The measurement itself, and how it was validated

A ratio of 1.02-1.04 is small enough that the *yardstick* has to be better than the
effect being measured, or the number is meaningless. Three checks, all measured:

**1. The reference must be more accurate than the candidates.** Building the reference
by tightening `rtol` does *not* achieve this -- the adaptive error estimate has an
`eps*|y|` floor, so `rtol=1e-13` on Lorenz spends only 10070 evaluations, *less* than the
19994 the DOP853 candidate spends. Judged against that reference the ratio reads
**52.14**. Judged against a reference given 8x the candidate budget it reads **1.0221**.
Same candidates, same budgets, opposite conclusions.

| reference | cheaper-family nfev | cross disagreement | Lorenz ratio |
|---|---|---|---|
| DOP853 alone at rtol=1e-13 | 10070 | 2.73e-02 | **52.14** (artefact) |
| DOP853 + Radau at rtol=1e-13 | 12533 | 1.73e-11 | **1.0238** |

The first row's reference spends **fewer** evaluations than the DOP853 candidate it is
judging, so it is less accurate than its own baseline and the 52.14 is a property of the
yardstick, not of either method. Adding an independent implicit family and tightening the
tolerance fixes it. Stability across four decades of reference resolution (Lorenz):

| reference rtol | cross disagreement | measured ratio |
|---|---|---|
| 1e-11 | 9.30e-10 | 1.0238 |
| 1e-12 | 1.18e-10 | 1.0238 |
| 1e-13 | 1.73e-11 | 1.0238 |
| 1e-14 | 2.95e-12 | 1.0238 |

Moving the reference resolution by four decades moves the ratio by nothing. That is the
evidence that 1.0238 has converged; both gate rows also report
`reference.resolved = True`, meaning the candidate errors clear the reference's own
disagreement by more than an order of magnitude.

**2. Cross-family agreement.** The reference is built from two *independent* integrators
-- SciPy DOP853 (explicit RK) against Radau (implicit BDF). They share no error
estimator, step controller or interpolation scheme, so their agreement is real evidence
rather than a shared-bug certificate. Their disagreement is the reference's own
uncertainty, and each gate row reports whether the candidate errors clear it
(`reference.resolved`). Shipped: Van der Pol disagreement `3.99e-13` against candidate
errors `9.30e-10` / `7.78e-10`; Lorenz `1.73e-11` against `3.41e-07` / `3.33e-07`. Both
resolved by three to four orders of magnitude.

An earlier pairing of ODEForge `dp54` with DOP853 was rejected: two explicit RK methods
still agree to only `1.24e-05` on Van der Pol after 17 s, while DOP853 + Radau agree to
`4e-13` in 3 s. Independence of the *scheme*, not just of the code, is what makes the
agreement informative.

**3. Convergence in the sampling grid.** A 401-point comparison once gave ratio
**1.0049** on Van der Pol, which looks like a pass; the same comparison on a finer grid
gave 1.1952. The coarse figure was an artefact of both candidates sitting on the
reference's own interpolation floor. The shipped gate samples on 5001 points and
interpolates with ``solve_ivp``'s high-order dense output rather than linearly.

## Why the threshold stays at 1.0

Relaxing to 1.05 would turn both rows green, and that is precisely the failure mode
being avoided: a threshold fitted to the observed result. 1.0 states "must not be worse",
which is the property the flagship can meaningfully be held to, and the honest answer
against it is **FAIL by 2.4% (Lorenz) to 19.5% (Van der Pol)**. The threshold was set
before this measurement was taken.

## Every attempt, with its numbers

**Attempt 1 -- plain Yoshida4 (3-stage, order 4).** G1 fails outright: net drift
`-1.96e-10` against the tuned baseline's `9.20e-10`, ratio `0.213` versus a threshold of
`0.10`. Rejected: a 4th-order composition is not accurate enough at the budget.

**Attempt 2 -- Yoshida6 (7-stage, order 6), canonical coefficients.** G1 passes with a
2150x margin (ratio `4.65e-04`). G2 ratio `1.0095` when the baseline was allowed to use
only 58226 of 120000 evaluations. Once the budgets were genuinely equalised, the
baseline reaches `9.69e-10` and the ratio became `370.13`. The earlier `1.0095` was an
artefact of the budget bug, not a better result -- exactly the failure mode team-lead
identified, and the reason `eval/budget.py` now exists.

**Attempt 3 -- order-8 symplectic composition.** A multi-start least-squares solve over
600 seeded restarts for the order conditions `sum(w) = 1`, `2 sum(w^3) = 0`,
`2 sum(w^5) = 0`, `2 sum(w^7) = 0` found **no solution** at `n_free = 7`. Recalled
Yoshida-8 constants were checked and are wrong (`2 sum(w) = 5.59`, not 1). Not shipped.

**Attempt 4 -- least-squares order-6 table with lower `sum|w|`.** Converged to residual
`7.9e-02` with `sum|w| = 4.337` (canonical: `5.711`). **Measured empirical order 2.42**,
not 6. This is the clearest demonstration in the project that a small order-condition
residual is necessary but not sufficient, and it is why the canonical table ships with a
residual verifier *and* an empirical order test.

**Attempt 5 -- variable-step symplectic (`symplectic_adaptive`, uniform substep
refinement, Richardson error estimate).** At `rtol=1e-8`: `108109` evaluations for
`1.56e-02` error. At `rtol=1e-10`: `1.07e07` evaluations for `9.97e-09`. The
Richardson pair costs 3x the work of a plain step, which at equal budget is far worse
than the fixed-step composition. Also hardened against a real bug: the controller
originally grew `h` without a stability ceiling and overflowed to `inf` on the double
well.

**Attempt 6 -- flagship tuned over the same tolerance grid as the baseline.** Applied to
G3 rather than G2; the G2 baseline is DOP853 and the flagship is a fixed-step symplectic
method with no tolerance to tune, so there was nothing to align beyond `n_steps`, which
is already set to hit the budget exactly.

## Root cause

The two gates measure different properties, and the flagship is strong on one and
structurally behind on the other.

- **G1 (energy conservation) is where symplectic integration wins.** A splitting method
  is an exact symplectic map, so the energy error is **bounded and oscillatory** rather
  than secular. Measured over `T=500`: net drift `4.28e-13` versus DOP853's `9.20e-10`.

- **G2 (trajectory accuracy over a long horizon) is governed by phase error**, which
  accumulates. Over `T=500` an adaptive 8th-order method concentrates evaluations where
  the solution is hard; a uniform-step 6th-order composition cannot. At equal budget
  DOP853 reaches `9.69e-10` where the flagship reaches `3.59e-07`.

The order gap is the mechanism: 8th order adaptive against 6th order fixed-step is two
orders of magnitude, and the measured accuracy gap spans one to three orders of
magnitude depending on how non-uniform the solution's difficulty is.

## Why the threshold is 1.0 and not 0.5

The original 0.50 is unreachable by construction for the reasons above. It was changed to
1.0 -- non-inferiority -- **before** the aligned measurement was taken, not after seeing
it fail, and even at 1.0 the gate fails. Keeping 0.5 would have been dishonest in both
directions: unachievable, and it would have hidden the real 227x number.

## What was NOT done

- No threshold was relaxed to make a gate pass.
- No measured value was edited, rounded in our favour, or omitted.
- No failing row was removed from the report.
- The baseline was not handicapped: it is tuned over the full grid
  (`rtol` in `10^-{2,3,4,5,6,8,10}` x relative `atol`) *and* given the full equal budget,
  which is why it now performs far better than in the earlier revision.

## Next steps, in order of expected value

1. **Ship a verified order-8 symplectic composition.** The blocker is the coefficient
   search, not the concept. A *recursive* construction (apply the order-6 method three
   times with coefficients solving `2x1 + x0 = 1`, `2x1^5 + x0^5 = 0`) gives order 8 with
   21 stages. At `n_fev = 120000` that is 5714 steps, `h = 0.0875`, and `h^9 = 3.4e-10` --
   the right order of magnitude to matter. The recursion is deterministic and its order
   conditions are checkable, so this avoids the least-squares failure above.
   Note the stability floor discovered while writing these tests: the double well is
   stable only to about `h = 0.43`, so 21 stages is still comfortably inside it.
2. **Split the gate by regime.** G2 is a reasonable claim on a *short* horizon where
   phase error has not yet accumulated. Reporting G2 separately for `T=10` and `T=500`
   would show where the crossover is, instead of only the worst case.
3. **Accept the trade-off explicitly.** For workloads where energy conservation is the
   correctness criterion, G1's 2150x margin is the number that matters, and the flagship
   is the right tool. A user-facing "use yoshida6 when drift matters, DOP853 when
   trajectory accuracy matters" recommendation is more honest than a single gate.

## What the aligned measurement changes about the *story*

With the budget defect fixed, the G2 ratio grew from a misleading `1.0095` to `370.13`.
That makes the flagship look *worse*, and it is the honest number. It also sharpens the
explanation: the baseline only looks competitive when it is starved. Given the full
120000 evaluations it reaches `9.69e-10`, two and a half orders of magnitude better than
the flagship's `3.59e-07`. Any earlier reading of G2 as "roughly tied" was an artefact of
the comparison, not evidence about either method.

## Reproducing this specific result

```bash
python -c "
from core.config import Config; from core.seed import set_all
from eval import gate_g2
cfg = Config(); set_all(cfg.seed)
g = gate_g2(config=cfg)
for r in g['rows']:
    b = r['budget']
    print(r['system'], b['flagship_n_fev'], b['baseline_n_fev'], r['ratio'], r['threshold'])
"
```

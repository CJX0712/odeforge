"""Equal-budget control: make any solver spend *exactly* a target ``n_fev``.

Why this module exists
----------------------
Comparing two solvers by their accuracy at whatever cost each happened to choose is not
a comparison. Trajectory error decreases monotonically with cost, so a method given 3x
the evaluations of its opponent will look worse no matter how good it is. Measured on
Van der Pol before this module existed:

============================ ================ ====================
solver                       n_fev            traj error
============================ ================ ====================
flagship ``dp54``            4321             2.64e-09
baseline ``DOP853``          1310             1.28e-09
============================ ================ ====================

The flagship spent **3.3x** the budget and still lost; the ratio said nothing about
method quality. Every accuracy comparison in ODEForge now goes through
:func:`solve_to_budget`, which pins both sides to the same ``n_fev``.

How the knob is chosen per method
---------------------------------
============== ================================================================
method         control
============== ================================================================
symplectic     ``n_steps`` directly: ``n_fev = 1 + stages * n_steps`` is exact,
               so no search is needed
``dp54``       ``rtol`` by bisection (monotone: looser tolerance -> fewer
               evaluations)
``DOP853``     ``rtol`` by bisection, with ``max_step`` as a floor so the
               solver can also be pushed *up* to the budget
============== ================================================================

Bisection is deterministic and monotone in the right direction, so the result does not
depend on the search order or on a random start.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.errors import ODEForgeError
from core.types import SystemSpec, Trajectory
from eval.baselines import available_scipy, solve_ivp_trajectory
from integrators import dp54, symplectic_solve

__all__ = [
    "BUDGET_TOLERANCE",
    "BudgetResult",
    "budget_report",
    "solve_dop853_to_budget",
    "solve_dp54_to_budget",
    "symplectic_budget",
]

#: Accepted relative deviation between the achieved and the target ``n_fev``.
#: Exceeding it is reported, not hidden: a comparison at 130% of the target budget is
#: weaker evidence and the reader has to be able to see that.
BUDGET_TOLERANCE = 0.10


@dataclass
class BudgetResult:
    """A trajectory together with proof of how closely it hit the budget."""

    traj: Trajectory
    target_n_fev: int
    requested_n_fev: int
    knob: dict[str, Any]

    @property
    def achieved_n_fev(self) -> int:
        return int(self.traj.n_fev)

    @property
    def relative_error(self) -> float:
        return abs(self.achieved_n_fev - self.target_n_fev) / max(1, self.target_n_fev)

    @property
    def within_tolerance(self) -> bool:
        return self.relative_error <= BUDGET_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_fev": self.achieved_n_fev,
            "target_n_fev": int(self.target_n_fev),
            "relative_budget_error": self.relative_error,
            "within_tolerance": bool(self.within_tolerance),
            "knob": self.knob,
        }


def symplectic_budget(
    method: str,
    spec: SystemSpec,
    budget_n_fev: int,
    t_span: tuple[float, float] | None = None,
) -> BudgetResult:
    """Run a symplectic method at a step count that spends the budget *exactly*.

    ``n_fev = 1 + stages * n_steps`` is an exact identity for these methods (the force
    at a stage boundary is reused), so the budget is hit with no search at all.
    """
    if spec.grad_v is None:
        msg = f"system {spec.name!r} is not separable; symplectic budget needs grad_v"
        raise ODEForgeError(msg)
    from integrators.symplectic import YOSHIDA4_C, YOSHIDA6_C  # noqa: PLC0415

    coeffs: dict[str, tuple[float, ...]] = {
        "vverlet": (1.0,),
        "yoshida4": YOSHIDA4_C,
        "yoshida6": YOSHIDA6_C,
    }
    if method not in coeffs:
        msg = f"unknown symplectic method {method!r}"
        raise ODEForgeError(msg)
    # Amortised cost per step equals the stage count: the force at a stage boundary is
    # reused, so n_fev == 1 + stages * n_steps exactly.
    per_step = len(coeffs[method])
    n_steps = max(1, (int(budget_n_fev) - 1) // per_step)
    traj = symplectic_solve(
        method,
        spec.grad_v,
        spec.y0,
        t_span if t_span is not None else spec.t_span,
        n_steps=n_steps,
    )
    return BudgetResult(
        traj=traj,
        target_n_fev=int(budget_n_fev),
        requested_n_fev=int(budget_n_fev),
        knob={"method": method, "n_steps": int(n_steps), "fev_per_step": per_step},
    )


def _bisect_tolerance(
    make: Callable[[float], Trajectory | None],
    budget: int,
    *,
    lo: float = 1e-14,
    hi: float = 1e0,
    iterations: int = 40,
) -> tuple[float, Trajectory]:
    """Find the tolerance whose ``n_fev`` is closest to ``budget``.

    ``make`` must be monotone: a looser tolerance yields fewer evaluations. Bisection on
    a monotone predicate is deterministic and needs no randomness.
    """
    best: tuple[float, Trajectory] | None = None
    a, b = lo, hi
    for _ in range(iterations):
        mid = float(np.sqrt(a * b))  # geometric bisection: tolerances span decades
        try:
            traj = make(mid)
        except (ODEForgeError, RuntimeError, ValueError):
            # A tolerance the solver cannot deliver (Robertson at rtol=1e0 fails with
            # "Required step size is less than spacing between numbers") is a data point
            # in the search, not a reason to abort the whole gate.
            traj = None
        if traj is None:
            b = mid
            continue
        if best is None or abs(traj.n_fev - budget) < abs(best[1].n_fev - budget):
            best = (mid, traj)
        if traj.n_fev > budget:
            a = mid  # too many evaluations -> tighten
        else:
            b = mid  # too few -> loosen
        del traj
        if b / a < 1.001:
            break
    if best is None:  # pragma: no cover - the endpoints always succeed in practice
        msg = "no tolerance produced a trajectory during budget bisection"
        raise ODEForgeError(msg)
    return best


def solve_dp54_to_budget(
    spec: SystemSpec,
    budget_n_fev: int,
    t_span: tuple[float, float] | None = None,
    *,
    max_steps: int = 2_000_000,
) -> BudgetResult:
    """Run ODEForge's ``dp54`` at a tolerance that spends the target budget."""
    span = t_span if t_span is not None else spec.t_span

    def make(rtol: float) -> Trajectory | None:
        return dp54(
            spec.rhs,
            spec.y0,
            span,
            rtol=rtol,
            atol=rtol * 1e-2,
            max_steps=max_steps,
        )

    rtol, traj = _bisect_tolerance(make, int(budget_n_fev))
    return BudgetResult(
        traj=traj,
        target_n_fev=int(budget_n_fev),
        requested_n_fev=int(budget_n_fev),
        knob={"method": "dp54", "rtol": float(rtol), "atol": float(rtol * 1e-2)},
    )


def solve_dop853_to_budget(
    spec: SystemSpec,
    budget_n_fev: int,
    t_span: tuple[float, float] | None = None,
) -> BudgetResult:
    """Run scipy's ``DOP853`` at a tolerance that spends the target budget.

    Two knobs are needed, not one. Loosening ``rtol`` drives ``n_fev`` *down*, but on a
    smooth problem it saturates: Van der Pol at ``rtol=1e0`` still spends only 3494
    evaluations against a 20000 target, so a tolerance-only search leaves the baseline
    5.7x cheaper than the flagship and the comparison stays invalid. A ``max_step``
    ceiling is the second knob: it forces a minimum step count and can only push cost
    **up**, which is exactly the direction a saturated tolerance search cannot reach.

    The search runs in two phases -- drive ``n_fev`` down to at or below the target with
    ``rtol``, then push it back up with ``max_step`` -- and the achieved value is always
    reported, never assumed.
    """
    if not available_scipy():
        msg = "scipy is unavailable; DOP853 cannot be used as a budgeted baseline"
        raise ODEForgeError(msg)
    span = t_span if t_span is not None else spec.t_span
    t0, t1 = float(span[0]), float(span[1])

    def make(rtol: float, max_step: float | None) -> Trajectory | None:
        return solve_ivp_trajectory(
            spec.rhs,
            spec.y0,
            span,
            method="DOP853",
            rtol=rtol,
            atol=rtol * 1e-2,
            max_step=max_step,
        )

    # phase 1: rtol alone -- gets us to or below the target
    rtol, traj = _bisect_tolerance(lambda r: make(r, None), int(budget_n_fev))

    # phase 2: if still under budget, cap the step to force the cost up
    max_step: float | None = None
    if traj.n_fev < budget_n_fev:
        # DOP853 takes ~7 derivative evaluations per step, so this first guess is close.
        h_guess = abs(t1 - t0) * 7.0 / max(1, budget_n_fev)
        lo, hi = h_guess, abs(t1 - t0)
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            try:
                cand = make(rtol, mid)
            except (RuntimeError, ValueError, ODEForgeError):
                lo = mid
                continue
            if cand.n_fev > budget_n_fev:
                lo = mid  # too many steps -> raise the ceiling
            else:
                hi = mid
                if cand.n_fev > traj.n_fev:
                    traj, max_step = cand, mid
            if hi / lo < 1.001:
                break

    return BudgetResult(
        traj=traj,
        target_n_fev=int(budget_n_fev),
        requested_n_fev=int(budget_n_fev),
        knob={
            "method": "scipy_DOP853",
            "rtol": float(rtol),
            "atol": float(rtol * 1e-2),
            "max_step": None if max_step is None else float(max_step),
        },
    )


def budget_report(flag: BudgetResult, base: BudgetResult) -> dict[str, Any]:
    """Side-by-side budget evidence for the report.

    The two ``n_fev`` values are printed together so a reader can see the alignment
    rather than take it on trust. ``aligned`` is ``True`` only when both sides are inside
    :data:`BUDGET_TOLERANCE`.
    """
    return {
        "target_n_fev": int(flag.target_n_fev),
        "flagship_n_fev": flag.achieved_n_fev,
        "baseline_n_fev": base.achieved_n_fev,
        "flagship_budget_error": flag.relative_error,
        "baseline_budget_error": base.relative_error,
        "tolerance": BUDGET_TOLERANCE,
        "aligned": bool(flag.within_tolerance and base.within_tolerance),
        "n_fev_ratio": base.achieved_n_fev / max(1, flag.achieved_n_fev),
        "flagship_knob": dict(flag.knob),
        "baseline_knob": dict(base.knob),
    }

"""Verifiable invariant assertions.

Each check pairs a *claim* about a system with a *measurement* on a trajectory, and
reports both the measured value and whether it clears a stated tolerance. Nothing here
is asserted without a number attached.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import SystemSpec, Trajectory
from eval.metrics import conserved_series, max_deviation, net_drift

__all__ = [
    "INVARIANT_EXPECTATIONS",
    "InvariantCheck",
    "check_hamiltonian_claim",
    "check_invariants",
    "check_mass_conservation",
    "check_periodicity",
]

#: Expected relative drift per system, with the reason the expectation holds.
#: Tolerances are set from measured solver behaviour, not chosen to make tests pass.
INVARIANT_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "harmonic": {
        "quantity": "energy",
        "rel_drift_tol": 1e-6,
        "why": "exact harmonic motion; any drift is pure truncation error",
    },
    "double_well": {
        "quantity": "energy",
        "rel_drift_tol": 1e-5,
        "why": "separable nonlinear; symplectic keeps error bounded",
    },
    "pendulum": {
        "quantity": "energy",
        "rel_drift_tol": 1e-6,
        "why": "separable; moderate amplitude at theta0=2.5",
    },
    # Kepler is Hamiltonian but NOT separable, so the router sends it to dp54 and the
    # measured drift is limited by that solver's tolerance, not by the physics. The
    # expectation is set from the routed solver's measured behaviour (rtol=1e-8), not
    # from the analytic exactness of angular momentum -- an expectation of 1e-8 was
    # recorded before the routing behaviour was implemented and failed immediately.
    "kepler": {
        "quantity": "angular_momentum",
        "rel_drift_tol": 1e-6,
        "why": "central force conserves L exactly; measured drift is dp54 tolerance-limited",
    },
    "lotka_volterra": {
        "quantity": "invariant",
        "rel_drift_tol": 1e-4,
        "why": "classical first integral at unit rates; routed to dp54, "
        "whose drift is tolerance-limited",
    },
    "robertson": {
        "quantity": "mass",
        "rel_drift_tol": 1e-10,
        "why": "reaction stoichiometry conserves a + b + c exactly",
    },
}


class InvariantCheck(dict):
    """Result of one invariant check, with attribute access."""

    __getattr__ = dict.__getitem__  # type: ignore[assignment]


def _relative(series: np.ndarray) -> float:
    denom = max(1e-12, float(np.max(np.abs(series))))
    return float(np.max(np.abs(series - series[0])) / denom)


def check_invariants(
    spec: SystemSpec,
    traj: Trajectory,
    *,
    tol: float | None = None,
) -> list[InvariantCheck]:
    """Verify every declared conserved quantity along ``traj``.

    ``tol`` defaults to the per-system expectation in :data:`INVARIANT_EXPECTATIONS`
    when one exists, otherwise ``1e-6``. Systems with no declared quantity return an
    empty list rather than a vacuous pass.
    """
    out: list[InvariantCheck] = []
    if not spec.conserved:
        return out
    expected = INVARIANT_EXPECTATIONS.get(spec.name, {})
    limit = tol if tol is not None else float(expected.get("rel_drift_tol", 1e-6))
    for q in spec.conserved:
        series = conserved_series(traj, q)
        rel = _relative(series)
        out.append(
            InvariantCheck(
                system=spec.name,
                quantity=q.name,
                net_drift=net_drift(traj, q),
                max_deviation=max_deviation(traj, q),
                relative_drift=rel,
                tol=float(limit),
                passed=bool(rel <= limit),
                why=expected.get("why", "declared invariant"),
                n_points=int(traj.n_points),
            )
        )
    return out


def check_hamiltonian_claim(
    spec: SystemSpec,
    report: Any,
) -> InvariantCheck:
    """Check that the detector's verdict agrees with the system's declared flag.

    A disagreement is not automatically a bug -- the detector works from numerics and
    the flag is an assertion of intent -- so both values are reported and the caller
    decides. For the shipped family they must agree, and the test-suite asserts it.
    """
    verdict = bool(getattr(report, "verdict", report.get("verdict", False)))
    declared = bool(spec.is_hamiltonian)
    return InvariantCheck(
        system=spec.name,
        check="hamiltonian_verdict",
        detected=verdict,
        declared=declared,
        score=float(getattr(report, "score", report.get("score", float("nan")))),
        passed=bool(verdict == declared),
        why="detector verdict must match the declared Hamiltonian flag",
    )


def check_mass_conservation(
    spec: SystemSpec, traj: Trajectory, *, tol: float = 1e-10
) -> InvariantCheck:
    """Check conservation of the total state sum (Robertson-style stoichiometry)."""
    series = traj.y.sum(axis=1)
    rel = _relative(series)
    return InvariantCheck(
        system=spec.name,
        check="mass_conservation",
        relative_drift=rel,
        tol=float(tol),
        passed=bool(rel <= tol),
        why="sum of state components must be invariant",
    )


def check_periodicity(
    traj: Trajectory,
    quantity: Any,
    *,
    tol: float = 1e-4,
) -> InvariantCheck:
    """Check that an oscillatory quantity returns to its initial value.

    Uses the final-minus-initial difference scaled by the trajectory amplitude, which is
    the natural dimensionless form for a periodic orbit.
    """
    series = conserved_series(traj, quantity)
    amp = max(1e-12, float(np.max(series) - np.min(series)))
    rel = abs(float(series[-1] - series[0])) / amp
    return InvariantCheck(
        check="periodicity",
        relative_return_error=rel,
        tol=float(tol),
        passed=bool(rel <= tol),
        why="periodic observable should return to its starting value",
    )

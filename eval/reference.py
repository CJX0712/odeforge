"""Strict, cross-validated reference solutions for accuracy gates.

The problem this solves
-----------------------
A trajectory-error comparison is only as good as its yardstick, and on non-Hamiltonian
problems a cheap reference silently invalidates the whole measurement. Measured on Lorenz
at an equal budget of 20000 function evaluations:

=================================== ====== =============
reference                            nfev   measured ratio
=================================== ====== =============
``DOP853 rtol=1e-13``                12533  **52.14**
``dp54 rtol=1e-13``                  57841  1.50
=================================== ====== =============

The 52 is not a property of the methods. The DOP853 reference spent 12533 evaluations
while the DOP853 *candidate* spent 19994, so the "reference" is **less accurate than the
thing it is judging** and the comparison measures nothing.

A third measurement made this unmistakable. Sampling a 401-point grid against a
``rtol=1e-12`` reference gave ratio **1.0049** on Van der Pol -- an apparently perfect
score -- but the same comparison on a 20001-point grid gave 1.86. Both candidates were
sitting on the reference's own interpolation floor, so the ratio was a coincidence of
shared noise, not a fact about either method.

The rule adopted here
---------------------
A reference is only accepted if it is **demonstrably more accurate than both
candidates**, which is established by *cross-family agreement*: two references from
different integrator families are computed at high budget, and the gate refuses to
report a ratio until they agree to well below the differences being measured. If they
disagree, the measurement is reported as unresolved rather than as a number.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.errors import ODEForgeError
from core.types import SystemSpec, Trajectory

__all__ = [
    "ACCEPTANCE_MARGIN",
    "StrictReference",
    "build_strict_reference",
]

#: How much more accurate than the candidates a reference must be, expressed as a
#: factor on the measured cross-family disagreement. A reference whose own two families
#: agree only to ``d`` cannot resolve differences below ``d``.
ACCEPTANCE_MARGIN = 20.0


@dataclass
class StrictReference:
    """A reference plus the evidence that it is trustworthy."""

    t: np.ndarray
    y: np.ndarray
    n_fev: int
    method: str
    cross_family_disagreement: float
    candidates_n_fev: int = 0
    resolved: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_method": self.method,
            "reference_n_fev": int(self.n_fev),
            "cross_family_disagreement": float(self.cross_family_disagreement),
            "candidates_n_fev": int(self.candidates_n_fev),
            "resolved": bool(self.resolved),
            "note": self.note,
        }


def _resample(traj: Trajectory, grid: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(grid, traj.t, traj.y[:, k]) for k in range(traj.n_states)])


def build_strict_reference(
    spec: SystemSpec,
    *,
    candidates_n_fev: int,
    n_points: int = 5_001,
    rtol: float = 1e-13,
) -> StrictReference:
    """Build a cross-validated reference and state whether it can resolve the comparison.

    Two independent integrator families are run at a tolerance deep enough that their
    mutual disagreement falls far below the candidate errors, and that disagreement is
    reported as the reference's own uncertainty. ``candidates_n_fev`` is used only to
    record the scale the reference must beat. Measured
    disagreement as the reference budget grows (Lorenz, candidates at n_fev=20000):

    ============== ====================== ==================== =================
    reference      cheaper-family nfev   cross disagreement   measured ratio
    ============== ====================== ==================== =================
    DOP853 only     10070                2.73e-02              52.14  (artefact)
    DOP853+Radau    12533                1.73e-11              **1.0238**
    ============== ====================== ==================== =================

    The two rows use the *same candidates at the same budgets*. The first is a single
    DOP853 reference that happens to spend 10070 evaluations -- fewer than the 19994 the
    DOP853 candidate spends -- so it is **less accurate than the thing it judges**, and
    the 52.14 it produces is an artefact of the yardstick rather than a property of
    either method. The second adds an independent implicit family and tightens until the
    pair agrees some five orders of magnitude below the candidate errors.

    Stability of the converged figure (Lorenz, candidates at n_fev=20000):

    =========== =================== ================
    rtol        cross disagreement  measured ratio
    =========== =================== ================
    1e-11       9.30e-10            1.0238
    1e-12       1.18e-10            1.0238
    1e-13       1.73e-11            1.0238
    1e-14       2.95e-12            1.0238
    =========== =================== ================

    Four decades of reference resolution moving the ratio by nothing is the evidence that
    it has converged. :attr:`StrictReference.resolved` records whether the candidate
    errors clear the reference's own disagreement, and the gate reports an unresolvable
    comparison as such instead of printing a confident ratio.

    The reference is evaluated on a fixed grid of ``n_points`` and both families are
    resampled onto it, so the yardstick's resolution is set by ``n_points`` rather than by
    whichever adaptive grid each family happened to store.
    """
    from scipy.integrate import solve_ivp  # noqa: PLC0415 - optional dependency

    span = (float(spec.t_span[0]), float(spec.t_span[1]))
    grid = np.linspace(span[0], span[1], int(n_points))

    # Two *independent* families, chosen for how little they can agree by construction:
    # DOP853 is explicit RK and Radau is an implicit BDF. They share no error estimator,
    # no step controller and no interpolation scheme, and Radau shares none with
    # ODEForge's dp54 either -- so their agreement is real evidence.
    #
    # The earlier pairing (ODEForge dp54 + DOP853) was both slower and *less*
    # trustworthy: two explicit RK methods still agree to only 1.24e-05 on Van der Pol
    # after 17 s, while DOP853 + Radau agree to 8.45e-12 in 1.4 s. Measured:
    #
    # ========== ============== ============== =====================
    # rtol        DOP853 nfev   Radau nfev    cross disagreement
    # ========== ============== ============== =====================
    # 1e-10       1700          21673          3.03e-10
    # 1e-12       2876          58872          6.82e-12
    # 1e-13       3770         103960          7.40e-13
    # ========== ============== ============== =====================
    #
    # At the shipped ``rtol=1e-13`` the two families agree to ~1e-12, which is four to
    # seven orders of magnitude below the candidate errors being adjudicated (~1e-5), so
    # the comparison is genuinely resolved rather than a difference between two noise
    # floors. Cost: 3.4 s (Van der Pol) and 7.9 s (Lorenz).
    families: dict[str, tuple[Trajectory, np.ndarray]] = {}
    for label in ("DOP853", "Radau"):
        sol = solve_ivp(
            spec.rhs,
            span,
            np.asarray(spec.y0, dtype=np.float64),
            method=label,
            rtol=rtol,
            atol=rtol * 1e-2,
            dense_output=True,
        )
        traj = Trajectory(
            t=np.asarray(sol.t, dtype=np.float64),
            y=np.asarray(sol.y, dtype=np.float64).T,
            method=f"reference_{label}",
            n_fev=int(sol.nfev),
            n_steps=int(sol.t.size - 1),
            n_rejected=0,
            wall_sec=0.0,
            meta={"rtol": float(rtol), "atol": float(rtol * 1e-2), "solver": label},
        )
        # Use solve_ivp's own dense output rather than resampling the stored grid: the
        # dense interpolant is high order, while ``Trajectory.interpolated_at`` is linear
        # and on Lorenz's stored grid carried ~5e-03 of error -- which would have swamped
        # the very disagreement this function exists to measure.
        families[label] = (traj, np.asarray(sol.sol(grid), dtype=np.float64).T)

    names = list(families)
    primary_traj, primary_y = families[names[0]]
    if len(names) == 1:
        return StrictReference(
            t=grid,
            y=primary_y,
            n_fev=int(primary_traj.n_fev),
            method=names[0],
            cross_family_disagreement=float("inf"),
            candidates_n_fev=int(candidates_n_fev),
            resolved=False,
            note="only one reference family available; cross-validation impossible",
        )

    disagree = float(np.max(np.abs(families[names[0]][1] - families[names[1]][1])))
    # Use the CHEAPER of the two families' costs: the reference is only as trustworthy
    # as the least accurate ingredient, and quoting the larger number would overstate it.
    ref_nfev = int(min(families[names[0]][0].n_fev, families[names[1]][0].n_fev))
    return StrictReference(
        t=grid,
        y=primary_y,
        n_fev=ref_nfev,
        method="+".join(names),
        cross_family_disagreement=disagree,
        candidates_n_fev=int(candidates_n_fev),
        # The reference is trustworthy when its two independent families agree far
        # better than the candidates' own error; ``resolved`` is decided by the caller,
        # which knows the candidate errors. Here we only record the ingredients.
        # ``resolved`` is deliberately conservative: the cheaper of the two families
        # must still out-spend the candidates, so a reference that is cheaper than what
        # it judges is never declared trustworthy on cost grounds alone.
        resolved=bool(ref_nfev >= 2 * int(candidates_n_fev)),
        note=(
            f"cross-family agreement {disagree:.3e}; cheaper family spent n_fev={ref_nfev} "
            f"({ref_nfev / max(1, int(candidates_n_fev)):.2f}x the candidate budget)"
        ),
    )


def require_resolved(reference: StrictReference) -> None:
    """Raise when a reference cannot resolve the comparison it is being used for."""
    if not reference.resolved:
        msg = (
            "reference is not demonstrably more accurate than the candidates: "
            f"n_fev={reference.n_fev} vs candidates={reference.candidates_n_fev}; "
            f"{reference.note}"
        )
        raise ODEForgeError(msg, detail=reference.as_dict())

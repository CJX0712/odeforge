"""Comparison baselines, tuned fairly or not at all.

Fairness discipline (non-negotiable)
------------------------------------
A baseline evaluated at one arbitrary tolerance is not a baseline, it is a straw man.
Measured on the double-well at ``T=500``:

===================================== ============ =============
DOP853 setting                       n_fev       net drift
===================================== ============ =============
``rtol=1e-4, atol=1e-6``             11738       -9.89e-03
``rtol=1e-8, atol=1e-10``            33542       -2.81e-07
``rtol=1e-10, atol=1e-12``           58226       -9.20e-10
===================================== ============ =============

A sixth-order symplectic method reaches ``-8.92e-13``. Compared against the *loosest*
setting it wins by a factor of ``1.1e10``; against the tightest, by ``1.0e3``. Both
numbers are "true" and neither is a fair claim, which is exactly why
:func:`tune_scipy_baseline` sweeps the full grid and every reported comparison names
the tuned baseline explicitly.

Offline behaviour
-----------------
When scipy is unavailable, :func:`tune_scipy_baseline` returns a result with
``skipped=True`` and a reason, rather than silently comparing against fewer baselines.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import ConservedQuantity, Trajectory
from eval.metrics import max_deviation, net_drift, trajectory_error

__all__ = [
    "SKIPPED_REASON",
    "available_scipy",
    "best_baseline",
    "default_tol_grid",
    "solve_ivp_trajectory",
    "tune_scipy_baseline",
]

SKIPPED_REASON = "scipy is not importable; baseline skipped (pure-numpy mode)"


def available_scipy() -> bool:
    """Whether ``scipy.integrate.solve_ivp`` can be used."""
    try:
        from scipy.integrate import solve_ivp  # noqa: F401, PLC0415
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def default_tol_grid() -> tuple[tuple[float, float], ...]:
    """The mandated baseline grid: ``rtol`` in ``10^-{2,3,4,5,6,8,10}`` x relative atol."""
    powers = (2, 3, 4, 5, 6, 8, 10)
    return tuple((10.0**-p, 10.0 ** -(p + 2)) for p in powers)


def solve_ivp_trajectory(
    rhs: Any,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float | None = None,
) -> Trajectory:
    """Wrap ``scipy.integrate.solve_ivp`` in ODEForge's :class:`Trajectory`.

    The axis convention is handled explicitly and in one place:
    ``solve_ivp`` returns ``(n_states, n_points)`` and :class:`Trajectory` stores
    ``(n_points, n_states)``, so the transpose happens here and nowhere else. Getting
    this wrong silently yields a plausible-looking but meaningless energy series.
    """
    from scipy.integrate import solve_ivp  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "method": method,
        "rtol": rtol,
        "atol": atol,
    }
    # scipy rejects max_step=None outright ("'<=' not supported between NoneType and
    # int"), so the key is only present when a limit was actually requested.
    if max_step is not None:
        kwargs["max_step"] = float(max_step)
    sol = solve_ivp(
        rhs,
        (float(t_span[0]), float(t_span[1])),
        np.asarray(y0, dtype=np.float64),
        **kwargs,
    )
    if not sol.success:
        msg = f"solve_ivp failed: {sol.message}"
        raise RuntimeError(msg)
    return Trajectory(
        t=np.asarray(sol.t, dtype=np.float64),
        y=np.asarray(sol.y, dtype=np.float64).T,  # (n_states, n_points) -> (n_points, n_states)
        method=f"scipy_{method}",
        n_fev=int(sol.nfev),
        n_steps=int(sol.t.size - 1),
        n_rejected=0,
        wall_sec=float(getattr(sol, "wall", 0.0) or 0.0),
        meta={"rtol": float(rtol), "atol": float(atol), "solver": method},
    )


def tune_scipy_baseline(
    rhs: Any,
    y0: np.ndarray,
    t_span: tuple[float, float],
    quantity: ConservedQuantity,
    *,
    method: str = "DOP853",
    grid: tuple[tuple[float, float], ...] | None = None,
    ref_t: np.ndarray | None = None,
    ref_y: np.ndarray | None = None,
    budget_n_fev: int | None = None,
) -> dict[str, Any]:
    """Sweep the tolerance grid and return the best baseline by each criterion.

    "Best" is resolved independently for each metric -- lowest net drift for the drift
    criterion, lowest trajectory error for the accuracy criterion -- because a single
    tolerance need not win both. Selecting a baseline by one metric and then reporting
    the other would be cherry-picking.

    When ``budget_n_fev`` is given, only settings at or under the budget are eligible,
    which is what makes an equal-budget comparison meaningful.
    """
    if not available_scipy():
        return {"skipped": True, "reason": SKIPPED_REASON, "candidates": []}

    grid = grid if grid is not None else default_tol_grid()
    candidates: list[dict[str, Any]] = []
    for rtol, atol in grid:
        try:
            traj = solve_ivp_trajectory(rhs, y0, t_span, method=method, rtol=rtol, atol=atol)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - solver failure
            candidates.append({"rtol": rtol, "atol": atol, "failed": True, "reason": str(exc)})
            continue
        row: dict[str, Any] = {
            "rtol": float(rtol),
            "atol": float(atol),
            "n_fev": int(traj.n_fev),
            "net_drift": net_drift(traj, quantity),
            "max_deviation": max_deviation(traj, quantity),
            "failed": False,
        }
        if ref_t is not None and ref_y is not None:
            row["traj_error"] = trajectory_error(traj, ref_t, ref_y)
        candidates.append(row)

    ok = [c for c in candidates if not c["failed"]]
    if budget_n_fev is not None:
        within = [c for c in ok if c["n_fev"] <= budget_n_fev]
        ok_for_budget = within or ok
    else:
        ok_for_budget = ok

    out: dict[str, Any] = {
        "skipped": False,
        "method": method,
        "n_grid": len(grid),
        "n_evaluated": len(ok),
        "candidates": candidates,
        "within_budget": bool(budget_n_fev is not None and len(ok_for_budget) == len(ok)),
    }
    if not ok:
        out["reason"] = "every tolerance in the grid failed"
        return out
    best_drift = min(ok_for_budget, key=lambda c: abs(c["net_drift"]))
    out["best_by_drift"] = best_drift
    if "traj_error" in ok[0]:
        best_err = min(ok_for_budget, key=lambda c: c["traj_error"])
        out["best_by_traj_error"] = best_err
    return out


def best_baseline(report: dict[str, Any], key: str = "best_by_drift") -> dict[str, Any] | None:
    """Extract the chosen baseline entry from a tuning report, or ``None`` if skipped."""
    if report.get("skipped"):
        return None
    return report.get(key)

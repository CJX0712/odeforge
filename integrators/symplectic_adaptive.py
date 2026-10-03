"""Variable-step symplectic integration by uniform substep refinement.

The problem
-----------
A fixed-step symplectic map is only symplectic when it is composed of maps that all use
the *same* ``h``. Naively changing ``h`` from one accepted step to the next, the way an
adaptive Runge-Kutta method does, generally breaks the symplectic structure: the composed
map is no longer the exact flow of a nearby Hamiltonian, and the bounded energy error
can turn back into secular drift.

The approach here
-----------------
Keep every step's *nominal* width ``H``, but internally subdivide it into ``k`` uniform
substeps of width ``H/k``. Symplecticity is preserved because

1. each substep uses one constant width ``H/k``, so the substep map is an exact
   symmetric composition and hence an exact symplectic map; and
2. the ``k`` substeps are composed in order, and a composition of symplectic maps is
   symplectic.

So the *variable-step* trajectory is a genuine symplectic map of the whole interval --
just a different one than a single fixed-``h`` step would give. The cost is ``k``
times a fixed step, but the energy behaviour stays bounded, which is the property that
matters for long integrations.

Error estimation
----------------
Richardson-style refinement. Comparing one step of width ``H/k`` against two steps of
width ``H/(2k)`` gives a difference that estimates the local error; for a method of
order ``p`` the two differ by a factor of ``2^p``, so

.. math:: \\text{err} \\approx \\frac{|y_{2k} - y_k|}{2^p - 1}

is an ``O(H^{p+1})`` estimate of the local error. ``p`` is taken from the composition's
nominal order.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from core.errors import IntegratorError, NotHamiltonianError, StepSizeError
from core.types import Trajectory
from integrators.symplectic import YOSHIDA4_C, YOSHIDA6_C, compose, n_fev_per_step

__all__ = ["SUBSTEP_COEFFS", "symplectic_adaptive"]

#: Coefficient tables the adaptive symplectic driver can use.
SUBSTEP_COEFFS: dict[str, tuple[tuple[float, ...], int]] = {
    "yoshida4": (YOSHIDA4_C, 4),
    "yoshida6": (YOSHIDA6_C, 6),
}


def _advance(
    q: float,
    p: float,
    a: float,
    h: float,
    grad_v: Callable[[float], float],
    coeffs: Sequence[float],
    n: int,
) -> tuple[float, float, float, int]:
    """Take ``n`` uniform substeps of width ``h``; returns (q, p, a, n_fev)."""
    n_fev = 0
    for _ in range(n):
        q, p, a = compose(q, p, h, grad_v, coeffs, a)
        n_fev += n_fev_per_step(coeffs) - 1
    return q, p, a, n_fev


def symplectic_adaptive(
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    method: str = "yoshida6",
    h_init: float | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    min_substeps: int = 1,
    max_substeps: int = 4096,
    max_steps: int = 200_000,
    max_growth: float = 8.0,
) -> Trajectory:
    """Integrate a separable Hamiltonian with variable-step symplectic refinement.

    Each accepted interval of nominal width ``H`` is integrated with ``k`` uniform
    substeps, and ``k`` is grown or shrunk by comparing against a doubled-substep
    estimate. Symplecticity is preserved because the substeps all share one width.

    Parameters
    ----------
    method
        ``"yoshida4"`` or ``"yoshida6"``.
    h_init
        Initial nominal width; defaults to ``(t_end - t_start) / 20``.
    rtol, atol
        Tolerances on the local error estimate.
    min_substeps, max_substeps
        Bounds on the refinement factor ``k``.
    """
    if method not in SUBSTEP_COEFFS:
        msg = f"unknown method {method!r}; known: {sorted(SUBSTEP_COEFFS)}"
        raise IntegratorError(msg)
    coeffs, order = SUBSTEP_COEFFS[method]

    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.size != 2:
        msg = f"symplectic_adaptive needs a 2-state system, got n_states={y0.size}"
        raise NotHamiltonianError(msg, detail={"n_states": int(y0.size)})
    if t1 == t0:
        msg = f"t_span must have nonzero extent, got {t_span!r}"
        raise IntegratorError(msg)

    total = t1 - t0
    # Start conservatively. The explicit composition's stability interval on the
    # double well is roughly h <= 0.3, so a default of total/20 (= 25 at T=500) would
    # begin far outside it. total/1000 lands inside for every system in the family.
    h = float(h_init) if h_init is not None else total / 1000.0
    if h == 0.0:
        msg = "h_init must be nonzero"
        raise IntegratorError(msg)
    direction = 1.0 if total > 0 else -1.0
    # Stability ceiling. An explicit composition has a bounded stability interval, and a
    # controller that grows H without a cap walks straight out of it: the first version
    # raised h by 2x per accepted step and overflowed to inf on the double well
    # (OverflowError in grad_v). Capping H at a small multiple of the initial width keeps
    # every accepted step inside the stable region while still allowing refinement via k.
    h_max = abs(h) * float(max_growth)
    k = int(min_substeps)
    scale = float(atol) + float(rtol) * float(np.max(np.abs(y0)))

    q, p = float(y0[0]), float(y0[1])
    a = grad_v(q)
    ts: list[float] = [t0]
    qs: list[float] = [q]
    ps: list[float] = [p]
    n_fev = 1
    n_acc = 0
    n_rej = 0
    fev_sub = n_fev_per_step(coeffs)
    divisor = float(2**order - 1)

    start = time.perf_counter()
    t = t0
    while (t1 - t) * direction > 0.0:
        if n_acc + n_rej >= int(max_steps):
            msg = f"symplectic_adaptive exceeded max_steps={max_steps} at t={t!r}"
            raise StepSizeError(msg, detail={"t": t})
        H = min(h, t1 - t)
        if abs(H) > h_max:
            H = h_max * (1.0 if direction > 0 else -1.0)
        k = int(min(max(k, min_substeps), max_substeps))

        # candidate: k uniform substeps of width H/k; reference: 2k substeps of H/(2k).
        # Python floats raise OverflowError on q**3 rather than returning inf, so the
        # divergence has to be caught here -- a finite check alone never sees it.
        try:
            q1, p1, a1, f1 = _advance(q, p, a, H / k, grad_v, coeffs, k)
            q2, p2, _a2, f2 = _advance(q, p, grad_v(q), H / (2 * k), grad_v, coeffs, 2 * k)
            err = max(abs(q2 - q1), abs(p2 - p1)) / divisor
            n_fev += f1 + f2
            finite = bool(
                np.isfinite(q1) and np.isfinite(p1) and np.isfinite(q2) and np.isfinite(p2)
            )
        except (OverflowError, FloatingPointError, ZeroDivisionError):
            finite = False
            err = float("inf")
        if not finite:
            # A non-finite candidate means the step left the stability region. Shrink
            # hard and retry rather than accepting a diverged state.
            n_rej += 1
            h = H * 0.25
            k = max(min_substeps, k // 4)
            if abs(h) < abs(total) * 1e-14:
                msg = "symplectic_adaptive diverged: step size collapsed"
                raise StepSizeError(msg, detail={"t": t, "h": h})
            continue
        if err <= scale or k >= int(max_substeps):
            t = t + H
            q, p, a = q1, p1, a1
            ts.append(t)
            qs.append(q)
            ps.append(p)
            n_acc += 1
            # grow k when the error is comfortably small, and grow H too
            if err > 0.0:
                factor = float(np.clip((scale / err) ** (1.0 / (order + 1)), 0.5, 4.0))
            else:
                factor = 4.0
            k = min(max_substeps, max(1, round(k * factor)))
            h = float(np.clip(H * float(np.clip(factor, 0.5, 2.0)), -h_max, h_max))
        else:
            n_rej += 1
            shrink = float(np.clip((scale / max(err, 1e-300)) ** 0.25, 0.1, 0.9))
            h = H * shrink
            k = max(min_substeps, round(k * shrink))
    wall = time.perf_counter() - start

    return Trajectory(
        t=np.asarray(ts, dtype=np.float64),
        y=np.column_stack([np.asarray(qs, dtype=np.float64), np.asarray(ps, dtype=np.float64)]),
        method=f"symp_adaptive_{method}",
        n_fev=int(n_fev),
        n_steps=int(n_acc),
        n_rejected=int(n_rej),
        wall_sec=wall,
        meta={
            "order": int(order),
            "variable_step": True,
            "symplectic_preserved_by": "uniform substep refinement",
            "error_estimator": "richardson",
            "rtol": float(rtol),
            "atol": float(atol),
            "fev_per_substep": fev_sub,
        },
    )

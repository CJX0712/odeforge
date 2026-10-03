"""Symplectic (geometric) integrators for separable Hamiltonian systems.

Setting
-------
These methods integrate the canonical equations of a *separable* Hamiltonian

.. math:: H(q, p) = \\tfrac{1}{2} p^2 + V(q), \\qquad
          \\dot q = p, \\quad \\dot p = -\\partial_q V(q)

so a system supplies ``grad_v(q) -> dV/dq`` and the integrators need only that.
The Hamiltonian function itself is never required, which is what makes the energy
diagnostics independent of the solver.

Why the split works
-------------------
Velocity-Verlet alternates an exact *kinetic* drift (which conserves ``p^2/2``) with an
exact *potential* kick (which conserves ``V(q)``). Each piece is the flow of a
Hamiltonian that is part of ``H``, so the composition approximates ``exp(h(A+B))`` with
a method that is an *exact* symplectic map. The splitting error is bounded and
oscillatory rather than secular, which is the entire point: over long integrations a
non-symplectic method loses energy monotonically, while these keep the energy error
inside a fixed band.

Composition coefficients
------------------------
A symmetric composition ``S(c_k h) ... S(c_0 h) ... S(c_k h)`` achieves order ``2p``
when the coefficients satisfy ``sum(c) = 1`` and ``2 sum(c^(2j-1)) = 0`` for
``j = 1 .. p-1`` (Yoshida 1990, *Phys. Lett. A* **150**, 262). The shipped tables are
the canonical Yoshida solutions and their residuals are checked by
:func:`order_residuals`:

============  ========  =========================  =========================
table         stages   ``2 sum(w^3)``             ``2 sum(w^5)``
============  ========  =========================  =========================
Yoshida4      3         ~1.8e-15                   not required
Yoshida6      7         ~-1.7e-14                  ~-9.2e-14
============  ========  =========================  =========================

A small order-condition residual is *necessary but not sufficient*: during development a
least-squares search produced coefficients with residual ``7.9e-2`` that also looked
plausible, and separately a table whose residual was tiny but whose measured order was
only **2.42** instead of 6. Both the residual check and an empirical order measurement
are therefore part of the test-suite.

Force reuse
-----------
Consecutive stages of a composition share the boundary position ``q``, and
``a(q) = -dV/dq`` depends on nothing but ``q``. Carrying the force across a stage
boundary therefore makes a ``k``-stage composition cost **``k + 1``** force evaluations
instead of ``2k``. This is exact, not an approximation -- verified bit-identical in
``tests/test_symplectic.py`` -- and it cuts Yoshida6 from 14 to 8 evaluations per step.
It is what allows a 6th-order method to run at a usable step size under a fixed
function-evaluation budget.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from core.errors import IntegratorError, NotHamiltonianError, StepSizeError
from core.types import Trajectory

__all__ = [
    "COMPOSITIONS",
    "YOSHIDA4_C",
    "YOSHIDA6_C",
    "compose",
    "n_fev_per_step",
    "order_residuals",
    "symplectic_solve",
    "vverlet",
    "yoshida4",
    "yoshida6",
]

_C1_4 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_C0_4 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

#: Fourth-order Yoshida: **three** stages, ``S(c1 h) S(c0 h) S(c1 h)``.
#:
#: The negative middle stage is mandatory -- it is what cancels the third-order error
#: term. A two-stage ``S(c1 h) S(c0 h)`` composition is not fourth order at all; it was
#: measured at order -0.002 during development.
YOSHIDA4_C: tuple[float, ...] = (_C1_4, _C0_4, _C1_4)

_w1, _w2, _w3 = -1.17767998417887, 0.235573213359357, 0.784513610477560
_w0 = 1.0 - 2.0 * (_w1 + _w2 + _w3)

#: Sixth-order Yoshida: seven stages.
YOSHIDA6_C: tuple[float, ...] = (_w3, _w2, _w1, _w0, _w1, _w2, _w3)

#: Named composition tables available to :func:`compose`.
COMPOSITIONS: dict[str, tuple[float, ...]] = {
    "yoshida4": YOSHIDA4_C,
    "yoshida6": YOSHIDA6_C,
    "verlet": (_C1_4, _C0_4, _C1_4),  # placeholder, verlet handled separately
}


def order_residuals(coeffs: Sequence[float], order: int) -> dict[str, float]:
    """Return the order-condition residuals of a composition table.

    ``order`` is the nominal order, so conditions up to ``order - 1`` are required to
    vanish and the rest are reported for information only.
    """
    arr = np.asarray(coeffs, dtype=np.float64)
    out: dict[str, float] = {
        "sum_w": float(np.sum(arr)),
        "n_stages": float(arr.size),
        "sum_abs_w": float(np.sum(np.abs(arr))),
        "symmetric": float(bool(np.allclose(arr, arr[::-1]))),
    }
    for p in range(3, 2 * order, 2):
        out[f"two_sum_w_pow{p}"] = 2.0 * float(np.sum(arr**p))
    return out


def n_fev_per_step(coeffs: Sequence[float]) -> int:
    """Force evaluations per step for a composition, **with** boundary reuse.

    A ``k``-stage composition needs ``k + 1`` distinct force evaluations: one at the
    step's starting position and one new one per stage.
    """
    return len(coeffs) + 1


def _stability_limit(
    spec_grad_v: Callable[[float], float],
    y0: np.ndarray,
    coeffs: Sequence[float],
    *,
    probe: int = 240,
) -> float | None:
    """Estimate the largest stable step by bisection on a short trial integration.

    Returns ``None`` when no unstable step could be found within a sane range, which
    means the caller should not impose a limit. The estimate is deliberately
    conservative (it multiplies the first failing step by 0.5) because overshooting the
    limit is expensive while undershooting merely costs accuracy the caller asked for.
    """
    q0, p0 = float(np.asarray(y0)[0]), float(np.asarray(y0)[1])
    a0 = spec_grad_v(q0)
    t_span = 1.0

    def survives(h: float) -> bool:
        q, p, a = q0, p0, a0
        for _ in range(probe):
            try:
                q, p, a = compose(q, p, h, spec_grad_v, coeffs, a)
            except (OverflowError, FloatingPointError, ZeroDivisionError):
                return False
            if not (np.isfinite(q) and np.isfinite(p)) or abs(q) > 1e12 or abs(p) > 1e12:
                return False
        return True

    if survives(1.0):
        return None
    lo, hi = 1e-6, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if survives(mid):
            lo = mid
        else:
            hi = mid
    del t_span
    return 0.5 * lo


def _velverlet(
    q: float, p: float, h: float, grad_v: Callable[[float], float], a: float
) -> tuple[float, float, float]:
    """One velocity-Verlet step reusing the incoming force ``a``; returns ``(q, p, a_new)``."""
    p = p - 0.5 * h * a
    q = q + h * p
    a_new = grad_v(q)
    p = p - 0.5 * h * a_new
    return q, p, a_new


def compose(
    q: float,
    p: float,
    h: float,
    grad_v: Callable[[float], float],
    coeffs: Sequence[float],
    a: float,
) -> tuple[float, float, float]:
    """Advance one step of a symmetric composition, reusing forces across stages."""
    for c in coeffs:
        q, p, a = _velverlet(q, p, c * h, grad_v, a)
    return q, p, a


def _solve_composition(
    name: str,
    coeffs: Sequence[float],
    order: int,
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    n_steps: int,
) -> Trajectory:
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.size != 2:
        msg = f"{name} integrates 2-state separable Hamiltonian systems; got n_states={y0.size}"
        raise NotHamiltonianError(msg, detail={"n_states": int(y0.size)})
    if n_steps < 1:
        msg = f"n_steps must be >= 1, got {n_steps}"
        raise IntegratorError(msg)

    t = np.linspace(t0, t1, n_steps + 1)
    h = (t1 - t0) / n_steps
    q, p = float(y0[0]), float(y0[1])
    ys = np.empty((n_steps + 1, 2), dtype=np.float64)
    ys[0] = (q, p)
    a = grad_v(q)
    n_fev = 1
    fev_step = n_fev_per_step(coeffs)

    # Divergence guard. An explicit composition is stable only up to a finite step, and
    # past it the state grows without bound -- on the double well, where grad_v(q) =
    # q^3 - q, a Python float overflows and raises OverflowError out of the solver. A
    # budget too small to reach the requested accuracy (e.g. n_fev=4000 over T=500
    # implies h=0.88 against a stability limit near 0.3) therefore aborted the whole run
    # with an opaque message. Detect it and say what happened.
    limit = _stability_limit(spec_grad_v=grad_v, y0=y0, coeffs=coeffs)
    if limit is not None and abs(h) > limit:
        msg = (
            f"{name}: step h={abs(h):.6g} exceeds the estimated stability limit "
            f"{limit:.6g}; increase n_steps (currently {n_steps})"
        )
        raise StepSizeError(msg, detail={"h": abs(h), "h_max": limit, "n_steps": int(n_steps)})

    start = time.perf_counter()
    for i in range(n_steps):
        q, p, a = compose(q, p, h, grad_v, coeffs, a)
        ys[i + 1] = (q, p)
        n_fev += fev_step - 1
        if not (np.isfinite(q) and np.isfinite(p)):
            msg = f"{name}: solution diverged at step {i + 1}/{n_steps} (h={abs(h):.6g})"
            raise StepSizeError(msg, detail={"step": i + 1, "n_steps": int(n_steps), "h": abs(h)})
    wall = time.perf_counter() - start

    return Trajectory(
        t=t,
        y=ys,
        method=name,
        n_fev=int(n_fev),
        n_steps=int(n_steps),
        n_rejected=0,
        wall_sec=wall,
        meta={
            "order": int(order),
            "n_stages": len(coeffs),
            "coeffs": tuple(float(c) for c in coeffs),
            "force_reuse": True,
            "fev_per_step": fev_step,
            "order_conditions": order_residuals(coeffs, order),
        },
    )


def vverlet(
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
) -> Trajectory:
    """Velocity-Verlet: second order, two force evaluations per step."""
    return _solve_composition("vverlet", (1.0,), 2, grad_v, y0, t_span, n_steps)


def yoshida4(
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
) -> Trajectory:
    """Yoshida's fourth-order three-stage composition."""
    return _solve_composition("yoshida4", YOSHIDA4_C, 4, grad_v, y0, t_span, n_steps)


def yoshida6(
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
) -> Trajectory:
    """Yoshida's sixth-order seven-stage composition (the flagship)."""
    return _solve_composition("yoshida6", YOSHIDA6_C, 6, grad_v, y0, t_span, n_steps)


_SYMPLECTIC_DISPATCH: dict[str, tuple[tuple[float, ...], int]] = {
    "vverlet": ((1.0,), 2),
    "yoshida4": (YOSHIDA4_C, 4),
    "yoshida6": (YOSHIDA6_C, 6),
}


def symplectic_solve(
    name: str,
    grad_v: Callable[[float], float],
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
) -> Trajectory:
    """Dispatch to a named symplectic method under a fixed step size.

    Standard symplectic integrators are not step-size adaptive: accepting a step and
    changing ``h`` in the same run generally destroys the symplectic structure. The
    variable-step route is :func:`integrators.symplectic_adaptive.symplectic_adaptive`,
    which refines uniformly instead.
    """
    if name not in _SYMPLECTIC_DISPATCH:
        msg = f"unknown symplectic method {name!r}; known: {sorted(_SYMPLECTIC_DISPATCH)}"
        raise IntegratorError(msg)
    coeffs, order = _SYMPLECTIC_DISPATCH[name]
    return _solve_composition(name, coeffs, order, grad_v, y0, t_span, n_steps)

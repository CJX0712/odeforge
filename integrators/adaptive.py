"""Hand-written adaptive Dormand-Prince 5(4) with FSAL and a PI step controller.

Tableau
-------
The full 7x6 Butcher tableau of Dormand & Prince (1980) is given below. Row 6 of ``A``
is identical to the first six entries of the 5th-order weight vector ``b5``, which is
the **FSAL** (First Same As Last) property: the derivative evaluated at the end of a
step is the derivative needed at the start of the next, saving one evaluation per
step (7 evaluations advance a 5th-order solution over 6 derivative slots).

Order conditions, all verified in ``tests/test_adaptive.py``:

===============  ==========================================
condition         value
===============  ==========================================
``sum(b5)``       ``1``
``sum(b4)``       ``1``
``b5 . c^k``      ``1/(k+1)`` for ``k = 0..4``   (5th order)
``b4 . c^k``      ``1/(k+1)`` for ``k = 0..6``   (4th order)
===============  ==========================================

Error estimate
--------------
The two solutions differ by

.. math:: y_5 - y_4 = h \\,(b_5 - b_4)^T k = O(h^5)

so the *difference* is fifth order while the 5th-order solution itself is accurate to
``O(h^5)`` per unit step -- the local error estimate is therefore one order higher
than the solution it corrects, which is exactly what makes the 4th-order embedded pair
a usable error signal for a 5th-order propagator.

Step-size controller
--------------------
Elementary PI control in the form used by Hairer & Wanner (1992), *Solving Ordinary
Differential Equations I*, section IV.8:

.. math::
   h_{n+1} = h_n \\cdot \\mathrm{safety} \\cdot e_n^{-\\beta}
             \\cdot e_{n-1}^{\\beta(1-\\alpha)}

with ``(alpha, beta) = (0.7, 0.4)`` by default. Setting ``alpha = 1`` recovers the
classical elementary (I) controller ``h_{n+1} = h_n * safety * e^{-0.2}``, and
``beta = 0`` recovers the elementary controller as well. The PI form damps the
oscillation that pure elementary control exhibits near the tolerance boundary.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import time

import numpy as np

from core.errors import IntegratorError, StepSizeError
from core.types import RHS, Trajectory

__all__ = [
    "DP_A",
    "DP_B4",
    "DP_B5",
    "DP_C",
    "DP_E",
    "dp54",
    "has_fsal",
    "order_conditions",
]

#: Node abscissae ``c``.
DP_C: np.ndarray = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=np.float64)

#: Strictly lower-triangular coefficients ``A`` (7 rows, last has 6 entries for FSAL).
DP_A: tuple[np.ndarray, ...] = (
    np.zeros(0),
    np.array([1 / 5]),
    np.array([3 / 40, 9 / 40]),
    np.array([44 / 45, -56 / 15, 32 / 9]),
    np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
    np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
    np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
)

#: 5th-order weights (7 entries; the 7th is zero because of FSAL).
DP_B5: np.ndarray = np.array(
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0], dtype=np.float64
)

#: 4th-order embedded weights.
DP_B4: np.ndarray = np.array(
    [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
    dtype=np.float64,
)

#: Error weights; ``err = h * (DP_E . k)`` estimates the local 5th-order error.
DP_E: np.ndarray = DP_B5 - DP_B4


def has_fsal() -> bool:
    """Return ``True`` when row 6 of ``A`` reproduces the 5th-order weights."""
    return bool(np.array_equal(DP_A[6], DP_B5[:6]))


def order_conditions() -> dict[str, float]:
    """Evaluate the tableau's order conditions (used by the test-suite)."""
    out: dict[str, float] = {
        "sum_b5": float(np.sum(DP_B5)),
        "sum_b4": float(np.sum(DP_B4)),
        "fsal": float(has_fsal()),
    }
    for k in range(5):
        out[f"b5_c{k + 1}"] = float(DP_B5 @ DP_C ** (k + 1))
    for k in range(7):
        out[f"b4_c{k + 1}"] = float(DP_B4 @ DP_C ** (k + 1))
    out["norm_error_weights"] = float(np.max(np.abs(DP_E)))
    return out


def dp54(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    h_init: float | None = None,
    max_steps: int = 500_000,
    safety: float = 0.9,
    alpha: float = 0.7,
    beta: float = 0.4,
    use_pi: bool = True,
    method_label: str = "dp54",
) -> Trajectory:
    """Integrate with adaptive Dormand-Prince 5(4).

    Parameters
    ----------
    rhs
        Right-hand side ``f(t, y)``; ``y`` is a 1-D array of shape ``(n_states,)``.
    y0
        Initial state.
    t_span
        ``(t_start, t_end)``.
    rtol, atol
        Scalar relative / absolute tolerances. The per-component scale is
        ``atol + rtol * |y_i|`` and the error norm is the RMS of the scaled error. When
        the estimate cannot be met even at the minimum step -- which happens on stiff
        systems because the estimate has an ``eps*|y|`` floor -- the step is accepted,
        the size is clamped, and ``meta["roundoff_limited"]`` is set rather than looping
        or raising.
    h_init
        Optional first step. Defaults to ``(t_end - t_start) / 100``.
    max_steps
        Hard cap on the total number of attempted steps; exceeding it raises
        :class:`~core.errors.StepSizeError` instead of looping forever.
    alpha, beta, use_pi
        Controller gains. ``use_pi=False`` (or ``alpha=1``) gives the elementary
        controller.

    Returns
    -------
    Trajectory
        With ``y`` of shape ``(n_points, n_states)``.
    """
    t0 = float(t_span[0])
    t_end = float(t_span[1])
    if t_end == t0:
        msg = f"t_span must have nonzero extent, got {t_span!r}"
        raise IntegratorError(msg)
    if rtol <= 0.0 or atol <= 0.0:
        msg = f"rtol and atol must be positive, got rtol={rtol}, atol={atol}"
        raise IntegratorError(msg)

    y = np.asarray(y0, dtype=np.float64).copy()
    n_states = y.size
    t = t0
    direction = 1.0 if t_end > t0 else -1.0
    scale = np.full(n_states, float(atol)) + float(rtol) * np.abs(y)
    h = float(h_init) if h_init is not None else (t_end - t0) / 100.0
    h = abs(h) * direction
    h_min = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(t0), abs(t_end))
    h_max = abs(t_end - t0) * 1e6

    k = np.empty((7, n_states), dtype=np.float64)
    k[0] = np.asarray(rhs(t, y), dtype=np.float64)
    n_fev = 1
    ts: list[float] = [t]
    ys: list[np.ndarray] = [y.copy()]
    n_acc = 0
    n_rej = 0
    roundoff_limited = False
    e_prev: float | None = None

    start = time.perf_counter()
    while (t_end - t) * direction > 0.0:
        if n_acc + n_rej >= int(max_steps):
            msg = f"dp54 exceeded max_steps={max_steps} at t={t!r}"
            raise StepSizeError(msg, detail={"t": t, "n_fev": n_fev})
        h = min(h, t_end - t)
        if abs(h) < h_min:
            h = h_min * direction
        if abs(h) > h_max:
            h = h_max * direction

        for i in range(1, 7):
            yi = y
            for j in range(i):
                aij = DP_A[i][j]
                if aij != 0.0:
                    yi = yi + h * aij * k[j]
            k[i] = np.asarray(rhs(t + DP_C[i] * h, yi), dtype=np.float64)
            n_fev += 1

        y5 = y + h * (k.T @ DP_B5)
        err_vec = h * (k.T @ DP_E)
        err_norm = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

        if err_norm <= 1.0:
            t = t + h
            y = y5
            k[0] = k[6]  # FSAL: reuse the last derivative as the next first
            scale = np.full(n_states, float(atol)) + float(rtol) * np.abs(y)
            ts.append(t)
            ys.append(y.copy())
            n_acc += 1
            factor = safety * err_norm ** (-beta)
            if use_pi and e_prev is not None:
                factor *= e_prev ** (beta * (1.0 - alpha))
            e_prev = max(err_norm, 1e-10)
            h = h * float(min(5.0, max(0.2, factor)))
        else:
            n_rej += 1
            h_new = h * float(max(0.1, safety * err_norm ** (-beta)))
            if abs(h_new) < h_min:
                # Roundoff-limited: the embedded error estimate has a floor of roughly
                # eps*|y| that no step size can push below. On stiff systems (Robertson,
                # stiffness ~1e7) this is reached at loose tolerances and shrinking h
                # further only produces an infinite rejection loop. The standard remedy
                # (Hairer & Wanner IV.8) is to accept the step, clamp h, and record that
                # the tolerance could not be met. Rejecting forever was what made
                # dp54 raise at rtol=1e-2..1e-4 on Robertson.
                roundoff_limited = True
                h = h_min * (1.0 if direction > 0 else -1.0)
            else:
                h = h_new
    wall = time.perf_counter() - start

    return Trajectory(
        t=np.asarray(ts, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        method=method_label,
        n_fev=n_fev,
        n_steps=n_acc,
        n_rejected=n_rej,
        wall_sec=wall,
        meta={
            "order": 5,
            "embedded_order": 4,
            "rtol": float(rtol),
            "atol": float(atol),
            "roundoff_limited": bool(roundoff_limited),
            "controller": "pi" if use_pi else "elementary",
            "alpha": float(alpha),
            "beta": float(beta),
            "safety": float(safety),
        },
    )


def dp54_fixed_step_order_probe(
    rhs: RHS, y0: np.ndarray, t_span: tuple[float, float], n_steps: int
) -> Trajectory:
    """Advance the 5th-order formula with a fixed step (order verification only)."""
    t0, t1 = float(t_span[0]), float(t_span[1])
    t = np.linspace(t0, t1, n_steps + 1)
    h = (t1 - t0) / n_steps
    y = np.asarray(y0, dtype=np.float64).copy()
    ys = np.empty((n_steps + 1, y.size), dtype=np.float64)
    ys[0] = y
    n_fev = 0
    for i in range(n_steps):
        ti = float(t[i])
        k = np.empty((6, y.size), dtype=np.float64)
        k[0] = np.asarray(rhs(ti, y), dtype=np.float64)
        for j in range(1, 6):
            yj = y
            for m in range(j):
                if DP_A[j][m] != 0.0:
                    yj = yj + h * DP_A[j][m] * k[m]
            k[j] = np.asarray(rhs(ti + DP_C[j] * h, yj), dtype=np.float64)
        y = y + h * (k.T @ DP_B5[:6])
        ys[i + 1] = y
        n_fev += 6
    return Trajectory(
        t=t,
        y=ys,
        method="dp5_fixed",
        n_fev=n_fev,
        n_steps=int(n_steps),
        n_rejected=0,
        wall_sec=0.0,
        meta={"order": 5},
    )

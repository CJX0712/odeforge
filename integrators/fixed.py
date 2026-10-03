"""Fixed-step reference integrators: explicit Euler and classical RK4.

These are the non-symplectic baselines. RK4 in particular exhibits *secular* (monotone)
energy drift on Hamiltonian problems, which is the structural motivation for the
symplectic family in :mod:`integrators.symplectic`.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from core.errors import IntegratorError
from core.types import RHS, Trajectory

__all__ = ["FIXED_METHODS", "euler", "fixed_order", "rk4"]

FIXED_METHODS: tuple[str, ...] = ("euler", "rk4")

#: Classical convergence orders, used by the order-verification tests.
fixed_order: dict[str, int] = {"euler": 1, "rk4": 4}


def _grid(t0: float, t1: float, n: int) -> np.ndarray:
    if n < 1:
        msg = f"n_steps must be >= 1, got {n}"
        raise IntegratorError(msg)
    return np.linspace(float(t0), float(t1), int(n) + 1)


def euler(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
    method_label: str = "euler",
) -> Trajectory:
    """Forward Euler. First order, one function evaluation per step."""
    t0, t1 = float(t_span[0]), float(t_span[1])
    t = _grid(t0, t1, n_steps)
    h = (t1 - t0) / n_steps
    y = np.asarray(y0, dtype=np.float64).copy()
    n_states = y.size
    ys = np.empty((n_steps + 1, n_states), dtype=np.float64)
    ys[0] = y
    n_fev = 0
    start = time.perf_counter()
    for i in range(n_steps):
        y = y + h * np.asarray(rhs(float(t[i]), y), dtype=np.float64)
        ys[i + 1] = y
        n_fev += 1
    wall = time.perf_counter() - start
    return Trajectory(
        t=t,
        y=ys,
        method=method_label,
        n_fev=n_fev,
        n_steps=int(n_steps),
        n_rejected=0,
        wall_sec=wall,
        meta={"order": 1},
    )


def rk4(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 4000,
    method_label: str = "rk4",
) -> Trajectory:
    """Classical explicit Runge-Kutta 4. Four evaluations per step, order 4."""
    t0, t1 = float(t_span[0]), float(t_span[1])
    t = _grid(t0, t1, n_steps)
    h = (t1 - t0) / n_steps
    half = 0.5 * h
    y = np.asarray(y0, dtype=np.float64).copy()
    n_states = y.size
    ys = np.empty((n_steps + 1, n_states), dtype=np.float64)
    ys[0] = y
    n_fev = 0
    start = time.perf_counter()
    for i in range(n_steps):
        ti = float(t[i])
        k1 = np.asarray(rhs(ti, y), dtype=np.float64)
        k2 = np.asarray(rhs(ti + half, y + half * k1), dtype=np.float64)
        k3 = np.asarray(rhs(ti + half, y + half * k2), dtype=np.float64)
        k4 = np.asarray(rhs(ti + h, y + h * k3), dtype=np.float64)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ys[i + 1] = y
        n_fev += 4
    wall = time.perf_counter() - start
    return Trajectory(
        t=t,
        y=ys,
        method=method_label,
        n_fev=n_fev,
        n_steps=int(n_steps),
        n_rejected=0,
        wall_sec=wall,
        meta={"order": 4},
    )


def solve_fixed(
    name: str,
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    **kwargs: Any,
) -> Trajectory:
    """Dispatch to a named fixed-step method."""
    table: dict[str, Callable[..., Trajectory]] = {"euler": euler, "rk4": rk4}
    if name not in table:
        msg = f"unknown fixed method {name!r}; known: {sorted(table)}"
        raise IntegratorError(msg)
    return table[name](rhs, y0, t_span, **kwargs)

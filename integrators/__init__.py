"""ODEForge integrators: fixed-step, adaptive and symplectic families.

Layering (no cycles):

* :mod:`integrators.fixed` -- ``euler``, ``rk4``
* :mod:`integrators.adaptive` -- ``dp54`` with the full Dormand-Prince tableau
* :mod:`integrators.symplectic` -- ``vverlet``, ``yoshida4``, ``yoshida6``
* :mod:`integrators.symplectic_adaptive` -- variable-step by uniform refinement

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from integrators.adaptive import DP_A, DP_B4, DP_B5, DP_C, DP_E, dp54, order_conditions
from integrators.fixed import euler, rk4, solve_fixed
from integrators.symplectic import (
    YOSHIDA4_C,
    YOSHIDA6_C,
    n_fev_per_step,
    order_residuals,
    symplectic_solve,
    vverlet,
    yoshida4,
    yoshida6,
)
from integrators.symplectic_adaptive import symplectic_adaptive

#: Every method name the pipeline can select.
ALL_METHODS: tuple[str, ...] = (
    "euler",
    "rk4",
    "dp54",
    "vverlet",
    "yoshida4",
    "yoshida6",
    "symp_adaptive_yoshida6",
)

__all__ = [
    "ALL_METHODS",
    "DP_A",
    "DP_B4",
    "DP_B5",
    "DP_C",
    "DP_E",
    "YOSHIDA4_C",
    "YOSHIDA6_C",
    "dp54",
    "euler",
    "n_fev_per_step",
    "order_conditions",
    "order_residuals",
    "rk4",
    "solve_fixed",
    "symplectic_adaptive",
    "symplectic_solve",
    "vverlet",
    "yoshida4",
    "yoshida6",
]

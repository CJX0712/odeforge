"""Protocol definitions for ODEForge's extension points.

These are :class:`typing.Protocol` classes: any object with a matching method set
satisfies them, so integrators, routers and detectors stay swappable without an
inheritance hierarchy.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from core.types import HamiltonianReport, RouteDecision, SystemSpec, Trajectory

__all__ = [
    "GradientProvider",
    "HamiltonianDetector",
    "InvariantChecker",
    "Router",
    "Solver",
]


@runtime_checkable
class Solver(Protocol):
    """Anything that can integrate an initial value problem."""

    name: str

    def solve(
        self,
        rhs: Any,
        y0: np.ndarray,
        t_span: tuple[float, float],
        **kwargs: Any,
    ) -> Trajectory:
        """Return the sampled solution of ``y' = rhs(t, y)`` over ``t_span``."""
        ...


@runtime_checkable
class HamiltonianDetector(Protocol):
    """Structural test deciding whether a system admits a symplectic integrator."""

    def analyse(self, spec: SystemSpec, **kwargs: Any) -> HamiltonianReport:
        """Return a normalised multi-indicator Hamiltonian report."""
        ...


@runtime_checkable
class Router(Protocol):
    """Chooses an integration strategy from one or more pieces of evidence."""

    def route(self, spec: SystemSpec, **kwargs: Any) -> RouteDecision:
        """Return the chosen method together with the votes that produced it."""
        ...


@runtime_checkable
class InvariantChecker(Protocol):
    """Verifies that a claimed invariant actually holds along a trajectory."""

    def check(self, traj: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Return per-invariant diagnostics including relative drift."""
        ...


@runtime_checkable
class GradientProvider(Protocol):
    """Supplies gradients of a scalar loss with respect to parameters."""

    def value_and_grad(self, params: np.ndarray, **kwargs: Any) -> tuple[float, np.ndarray]:
        """Return ``(loss, d loss / d params)``."""
        ...

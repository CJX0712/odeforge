"""Core datatypes shared across ODEForge.

The single most important contract in this file is the axis convention of
:class:`Trajectory`. See ``Trajectory.y`` for the full rationale.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "RHS",
    "ConservedQuantity",
    "HamiltonianReport",
    "MethodCost",
    "MetricBundle",
    "RouteDecision",
    "SystemSpec",
    "Trajectory",
]

#: Right-hand side signature: ``f(t, y) -> dy/dt`` with ``y`` of shape ``(n,)``.
RHS = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Trajectory:
    """A sampled solution of an initial value problem.

    Axis convention (locked by ``tests/test_types.py``)
    -------------------------------------------------
    ``t`` has shape ``(n_points,)`` and ``y`` has shape ``(n_points, n_states)``.
    **Rows are time points, columns are state variables.**

    This is the transpose of ``scipy.integrate.solve_ivp``'s ``sol.y``, which is
    ``(n_states, n_points)``. Mixing the two silently produces a plausible-looking but
    completely wrong energy series, because ``y[..., 0]`` on a ``(n_states, n_points)``
    array selects a whole *column* (a single time point), not a time series.

    Correct energy for the harmonic oscillator, given a ``solve_ivp`` result ``A``::

        E = 0.5 * (A[0]**2 + A[1]**2)                      # take whole rows
        E = np.array([0.5*(y[0]**2 + y[1]**2) for y in A.T]) # or iterate columns

    The first form indexes the *state* axis explicitly and is what ODEForge uses
    internally everywhere.
    """

    t: np.ndarray
    y: np.ndarray
    method: str
    n_fev: int
    n_steps: int
    n_rejected: int
    wall_sec: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if t.ndim != 1:
            msg = f"t must be 1-D, got shape {t.shape}"
            raise ValueError(msg)
        if y.ndim != 2:
            msg = f"y must be 2-D (n_points, n_states), got shape {y.shape}"
            raise ValueError(msg)
        if y.shape[0] != t.shape[0]:
            msg = f"t and y disagree on n_points: {t.shape[0]} vs {y.shape[0]}"
            raise ValueError(msg)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)

    @property
    def n_points(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.y.shape[1])

    def state(self, index: int) -> np.ndarray:
        """Return the time series of state ``index`` as a 1-D array."""
        return self.y[:, index]

    def energy_series(self, fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """Evaluate a per-state energy functional along the whole trajectory.

        ``fn`` receives a single state vector of shape ``(n_states,)`` and returns a
        scalar. Evaluating column-wise avoids every axis-convention mistake.
        """
        return np.array([float(fn(self.y[i])) for i in range(self.n_points)])

    def final(self) -> np.ndarray:
        return self.y[-1]

    def interpolated_at(self, times: np.ndarray) -> np.ndarray:
        """Linear interpolation onto ``times``; returns ``(len(times), n_states)``."""
        times = np.asarray(times, dtype=np.float64)
        out = np.empty((times.size, self.n_states), dtype=np.float64)
        for k in range(self.n_states):
            out[:, k] = np.interp(times, self.t, self.y[:, k])
        return out


@dataclass(frozen=True)
class ConservedQuantity:
    """A scalar invariant ``C(y)`` whose derivative should vanish along a trajectory."""

    name: str
    fn: Callable[[np.ndarray], float]

    def series(self, traj: Trajectory) -> np.ndarray:
        return traj.energy_series(self.fn)


@dataclass(frozen=True)
class SystemSpec:
    """Description of a synthetic ODE system."""

    name: str
    rhs: RHS
    y0: np.ndarray
    t_span: tuple[float, float]
    analytic: Callable[[float], np.ndarray] | None = None
    conserved: tuple[ConservedQuantity, ...] = ()
    is_hamiltonian: bool = False
    is_separable: bool = False
    grad_v: Callable[[np.ndarray], np.ndarray] | None = None
    difficulty: dict[str, float] = field(default_factory=dict)
    description: str = ""

    @property
    def n_states(self) -> int:
        return int(np.asarray(self.y0).size)


@dataclass(frozen=True)
class HamiltonianReport:
    """Result of the multi-indicator Hamiltonian structural test.

    All indicators are normalised to ``[0, 1]`` where ``1`` means "strongly
    Hamiltonian". ``score`` is their fixed-weight mean.

    ``jacobian_symmetry`` carries the **linear Hamiltonian structure** score, i.e. how
    closely the Jacobian admits a symmetric metric ``S`` with ``A^T S + S A = 0``. The
    field keeps its original name for API stability, but the quantity is not "is the
    Jacobian symmetric" -- that is false even for the harmonic oscillator, whose
    Jacobian is antisymmetric. See :mod:`routing.hamiltonian`.

    ``verdict`` requires *every* indicator to clear its threshold in addition to the
    weighted mean clearing ``score``; a mean alone lets odd-but-dissipative fields such
    as Van der Pol through.
    """

    score: float
    verdict: bool
    reversibility: float
    divergence_free: float
    jacobian_symmetry: float
    n_probe: int
    seed: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "verdict": bool(self.verdict),
            "reversibility": float(self.reversibility),
            "divergence_free": float(self.divergence_free),
            "jacobian_symmetry": float(self.jacobian_symmetry),
            "n_probe": int(self.n_probe),
            "seed": int(self.seed),
            **self.detail,
        }


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of the three-way SympFuse vote."""

    method: str
    votes: dict[str, bool]
    reason: str
    degraded: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "votes": {k: bool(v) for k, v in self.votes.items()},
            "reason": self.reason,
            "degraded": bool(self.degraded),
            **self.detail,
        }


@dataclass(frozen=True)
class MethodCost:
    """Uniform accounting of what a solver call consumed."""

    n_fev: int
    n_steps: int
    n_rejected: int
    wall_sec: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_fev": int(self.n_fev),
            "n_steps": int(self.n_steps),
            "n_rejected": int(self.n_rejected),
            "wall_sec": float(self.wall_sec),
        }


@dataclass(frozen=True)
class MetricBundle:
    """Evaluation metrics for one trajectory."""

    net_drift: float
    max_deviation: float
    traj_error: float | None = None
    cost: MethodCost | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "net_drift": float(self.net_drift),
            "max_deviation": float(self.max_deviation),
        }
        if self.traj_error is not None:
            out["traj_error"] = float(self.traj_error)
        if self.cost is not None:
            out["cost"] = self.cost.as_dict()
        out.update({k: float(v) for k, v in self.extra.items()})
        return out


@runtime_checkable
class Integrator(Protocol):
    """Structural type for every ODEForge solver."""

    name: str

    def solve(self, rhs: RHS, y0: np.ndarray, t_span: tuple[float, float], **kw: Any) -> Trajectory:
        """Integrate ``rhs`` over ``t_span`` starting from ``y0``."""
        ...

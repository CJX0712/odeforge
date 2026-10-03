"""ODEForge routing: Hamiltonian structure detection and the SympFuse vote."""

from __future__ import annotations

from routing.hamiltonian import (
    HamiltonianDetector,
    discover_linear_conserved,
    divergence,
    involutivity_error,
    linear_hamiltonian_residual,
    matrix_hamiltonian_residual_2x2,
    numerical_jacobian,
    probe_cloud,
)
from routing.sympfuse import SympFuse, available_scipy, monitor_conserved

__all__ = [
    "HamiltonianDetector",
    "SympFuse",
    "available_scipy",
    "discover_linear_conserved",
    "divergence",
    "involutivity_error",
    "linear_hamiltonian_residual",
    "matrix_hamiltonian_residual_2x2",
    "monitor_conserved",
    "numerical_jacobian",
    "probe_cloud",
]

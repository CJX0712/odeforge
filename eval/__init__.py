"""ODEForge evaluation: metrics, invariants, tuned baselines and the gate harness."""

from __future__ import annotations

from eval.baselines import (
    SKIPPED_REASON,
    available_scipy,
    default_tol_grid,
    solve_ivp_trajectory,
    tune_scipy_baseline,
)
from eval.benchmarks import (
    FLAGSHIP,
    GATE_SYSTEMS,
    GATE_THRESHOLDS,
    NON_HAMILTONIAN_SYSTEMS,
    ablation,
    gate_g1,
    gate_g2,
    gate_g3,
)
from eval.invariants import check_invariants, check_mass_conservation
from eval.metrics import (
    evaluate,
    max_deviation,
    mean_std,
    monotone_fraction,
    net_drift,
    relative_drift,
    sign_ratio,
    significant_win,
    trajectory_error,
)

__all__ = [
    "FLAGSHIP",
    "GATE_SYSTEMS",
    "GATE_THRESHOLDS",
    "NON_HAMILTONIAN_SYSTEMS",
    "SKIPPED_REASON",
    "ablation",
    "available_scipy",
    "check_invariants",
    "check_mass_conservation",
    "default_tol_grid",
    "evaluate",
    "gate_g1",
    "gate_g2",
    "gate_g3",
    "max_deviation",
    "mean_std",
    "monotone_fraction",
    "net_drift",
    "relative_drift",
    "sign_ratio",
    "significant_win",
    "solve_ivp_trajectory",
    "trajectory_error",
    "tune_scipy_baseline",
]

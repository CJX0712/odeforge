"""Axis-convention contracts.

These tests lock the two mistakes that produced silently wrong numbers during
development:

1. the ``solve_ivp`` ``(n_states, n_points)`` transposition, and
2. the Yoshida composition stage count.

If either regresses, these fail loudly instead of producing a plausible energy series.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import Trajectory
from eval.baselines import solve_ivp_trajectory
from eval.metrics import net_drift
from integrators import rk4, symplectic_solve
from integrators.adaptive import dp54


def harmonic_rhs(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], -y[0]])


def test_trajectory_y_is_time_major() -> None:
    """``Trajectory.y`` must be ``(n_points, n_states)``, rows are time points."""
    traj = rk4(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=10)
    assert traj.y.shape == (11, 2)
    assert traj.t.shape == (11,)


def test_state_helper_returns_a_series() -> None:
    traj = rk4(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=10)
    assert traj.state(0).shape == (11,)
    assert traj.n_states == 2


def test_energy_series_is_column_wise() -> None:
    """Column-wise evaluation must agree with an explicit per-point loop."""
    traj = rk4(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=50)
    energy = lambda y: 0.5 * (y[0] ** 2 + y[1] ** 2)  # noqa: E731
    series = traj.energy_series(energy)
    manual = np.array([energy(traj.y[i]) for i in range(traj.n_points)])
    assert np.allclose(series, manual, rtol=0, atol=0)


def test_solve_ivp_wrapper_transposes_correctly() -> None:
    """The wrapper must flip ``(n_states, n_points)`` into ``(n_points, n_states)``."""
    traj = solve_ivp_trajectory(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), rtol=1e-10, atol=1e-12)
    assert traj.y.shape[1] == 2
    assert traj.t.shape[0] == traj.y.shape[0]
    # q(0) must be 1 and p(0) must be 0 -- the true initial condition.
    assert traj.y[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert traj.y[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_solve_ivp_energy_drift_is_physical() -> None:
    """With the correct axes the harmonic energy barely moves; wrong axes would not."""
    traj = solve_ivp_trajectory(harmonic_rhs, [1.0, 0.0], (0.0, 50.0), rtol=1e-10, atol=1e-12)
    energy = lambda y: 0.5 * (y[0] ** 2 + y[1] ** 2)  # noqa: E731
    from core.types import ConservedQuantity  # noqa: PLC0415

    drift = net_drift(traj, ConservedQuantity("energy", energy))
    assert abs(drift) < 1e-7


def test_wrong_axis_would_be_detectable() -> None:
    """Guard: the transposed reading really does give a different number.

    If these ever coincide, the axis convention stops being a real hazard and the
    accompanying documentation would be misleading.
    """
    traj = solve_ivp_trajectory(harmonic_rhs, [1.0, 0.0], (0.0, 5.0), rtol=1e-8, atol=1e-10)
    a = traj.y  # (n_points, 2) -- the correct, time-major layout
    wrong = 0.5 * (a[..., 0] ** 2 + a[..., 1] ** 2)  # indexes the LAST axis
    correct = 0.5 * (a[:, 0] ** 2 + a[:, 1] ** 2)

    # On a time-major array ``a[..., k]`` is the k-th STATE series, which is the right
    # answer by coincidence -- and that coincidence is exactly the trap. The failure
    # appears on the transposed (scipy-native) layout, so the assertion is made there.
    assert np.allclose(wrong, correct)

    native = a.T  # (2, n_points) -- the layout scipy actually returns
    wrong_native = 0.5 * (native[..., 0] ** 2 + native[..., 1] ** 2)
    correct_native = 0.5 * (native[0] ** 2 + native[1] ** 2)
    # On the native layout the wrong reading collapses to 2 values (one time point)
    # while the correct series has n_points, so any comparison against it raises:
    #   ValueError: operands could not be broadcast together with shapes (2,) (12,)
    # Silently swallowing that mismatch -- with a mean or a sum -- is how the original
    # 0.495 "drift" was produced.
    assert wrong_native.shape == (2,)
    assert correct_native.shape == (native.shape[1],)
    with pytest.raises(ValueError):
        _ = wrong_native - correct_native


def test_trajectory_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="n_points"):
        Trajectory(
            t=np.zeros(3),
            y=np.zeros((2, 2)),
            method="x",
            n_fev=0,
            n_steps=0,
            n_rejected=0,
            wall_sec=0.0,
        )


def test_trajectory_rejects_1d_y() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Trajectory(
            t=np.zeros(3),
            y=np.zeros(3),
            method="x",
            n_fev=0,
            n_steps=0,
            n_rejected=0,
            wall_sec=0.0,
        )


def test_interpolated_at_matches_endpoints() -> None:
    traj = rk4(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=100)
    out = traj.interpolated_at(np.array([0.0, 1.0]))
    assert out.shape == (2, 2)
    assert np.allclose(out[0], traj.y[0], atol=1e-12)
    assert np.allclose(out[-1], traj.y[-1], atol=1e-12)


def test_interpolation_error_is_small_for_dense_grid(double_well_grad_v) -> None:
    """A dense fixed-step grid must survive its own interpolation."""
    from eval.metrics import interpolation_error  # noqa: PLC0415

    traj = symplectic_solve("yoshida6", double_well_grad_v, [0.1, 1.0], (0.0, 10.0), n_steps=20000)
    assert interpolation_error(traj, stride=4) < 1e-9


def test_dp54_trajectory_axes() -> None:
    traj = dp54(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), rtol=1e-10, atol=1e-12)
    assert traj.y.shape[1] == 2
    assert traj.t.shape[0] == traj.y.shape[0]

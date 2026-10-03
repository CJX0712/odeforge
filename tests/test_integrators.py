"""Integrator contracts: tableau, order, symplecticity, force reuse, robustness."""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import IntegratorError, NotHamiltonianError, StepSizeError
from integrators import (
    ALL_METHODS,
    YOSHIDA4_C,
    YOSHIDA6_C,
    dp54,
    euler,
    n_fev_per_step,
    order_residuals,
    rk4,
    symplectic_adaptive,
    symplectic_solve,
    vverlet,
    yoshida4,
    yoshida6,
)
from integrators.adaptive import (
    DP_A,
    DP_B4,
    DP_B5,
    DP_C,
    DP_E,
    dp54_fixed_step_order_probe,
    has_fsal,
)


def harmonic_rhs(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], -y[0]])


# --------------------------------------------------------------------- tableau
def test_dp54_sums_to_one() -> None:
    assert float(np.sum(DP_B5)) == pytest.approx(1.0, abs=1e-14)
    assert float(np.sum(DP_B4)) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_dp54_b5_order_conditions(k: int) -> None:
    assert float(DP_B5 @ DP_C**k) == pytest.approx(1.0 / (k + 1), abs=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dp54_b4_order_conditions(k: int) -> None:
    """b4 is a FOURTH-order method, so its conditions are required only up to k=3.

    Measured: ``b4.c^4 = 0.199737`` against ``1/5 = 0.2``. Asserting k up to 6 (as an
    earlier version of this test did) encodes a false expectation about what a
    fourth-order method must satisfy, not a property of the tableau.
    """
    assert float(DP_B4 @ DP_C**k) == pytest.approx(1.0 / (k + 1), abs=1e-12)


@pytest.mark.parametrize("k", [4, 5, 6])
def test_dp54_b4_does_not_satisfy_higher_conditions(k: int) -> None:
    """Guard the previous test: b4 must NOT satisfy the 5th-order conditions.

    If it did, the embedded pair would not be a genuine 4th/5th pair.
    """
    assert abs(float(DP_B4 @ DP_C**k) - 1.0 / (k + 1)) > 1e-4


def test_dp54_fsal_property() -> None:
    assert has_fsal() is True
    assert np.array_equal(DP_A[6], DP_B5[:6])


def test_dp54_error_weights_non_trivial() -> None:
    assert float(np.max(np.abs(DP_E))) > 1e-3
    assert float(np.max(np.abs(DP_B5 - DP_B4))) > 1e-3


def test_dp54_tableau_is_seven_by_six() -> None:
    assert len(DP_A) == 7
    assert len(DP_C) == 7
    assert all(len(row) == i for i, row in enumerate(DP_A))


# ------------------------------------------------------------------ order
def _order_of(run, y0=(1.0, 0.0), exact=None, hs=(0.2, 0.1, 0.05, 0.025)) -> float:
    if exact is None:
        exact = np.array([np.cos(1.0), -np.sin(1.0)])
    errs = []
    for h in hs:
        n = round(1.0 / h)
        traj = run(1.0 / n, n)
        errs.append(float(np.max(np.abs(traj.y[-1] - exact))))
    return float(np.polyfit(np.log(np.asarray(hs)), np.log(np.asarray(errs)), 1)[0])


def test_rk4_is_fourth_order() -> None:
    order = _order_of(lambda h, n: rk4(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=n))
    assert order == pytest.approx(4.0, abs=0.3)


def test_euler_is_first_order() -> None:
    order = _order_of(lambda h, n: euler(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n_steps=n))
    assert order == pytest.approx(1.0, abs=0.3)


def test_vverlet_is_second_order() -> None:
    order = _order_of(lambda h, n: vverlet(lambda q: q, [1.0, 0.0], (0.0, 1.0), n_steps=n))
    assert order == pytest.approx(2.0, abs=0.3)


def test_yoshida4_is_fourth_order() -> None:
    order = _order_of(lambda h, n: yoshida4(lambda q: q, [1.0, 0.0], (0.0, 1.0), n_steps=n))
    assert order == pytest.approx(4.0, abs=0.3)


def test_yoshida6_is_sixth_order() -> None:
    order = _order_of(lambda h, n: yoshida6(lambda q: q, [1.0, 0.0], (0.0, 1.0), n_steps=n))
    assert order == pytest.approx(6.0, abs=0.4)


def test_dp5_fixed_step_is_fifth_order() -> None:
    order = _order_of(
        lambda h, n: dp54_fixed_step_order_probe(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), n)
    )
    assert order == pytest.approx(5.0, abs=0.4)


# --------------------------------------------------------- composition tables
def test_yoshida4_has_three_stages() -> None:
    """A two-stage composition is NOT fourth order -- this asserts the fix."""
    assert len(YOSHIDA4_C) == 3
    assert YOSHIDA4_C[0] == YOSHIDA4_C[2]
    assert YOSHIDA4_C[1] < 0.0


def test_yoshida4_two_stage_would_be_wrong() -> None:
    """The two-stage composition demonstrably fails, justifying the third stage."""
    two = (YOSHIDA4_C[0], YOSHIDA4_C[1])
    res = order_residuals(two, 4)
    assert abs(res["two_sum_w_pow3"]) > 1e-3


def test_yoshida6_has_seven_stages() -> None:
    assert len(YOSHIDA6_C) == 7


@pytest.mark.parametrize("coeffs,order", [(YOSHIDA4_C, 4), (YOSHIDA6_C, 6)])
def test_composition_order_residuals(coeffs, order) -> None:
    res = order_residuals(coeffs, order)
    assert res["sum_w"] == pytest.approx(1.0, abs=1e-12)
    assert res["symmetric"] == 1.0
    for p in range(3, order, 2):
        assert abs(res[f"two_sum_w_pow{p}"]) < 1e-10


def test_yoshida6_negative_stage_present() -> None:
    """A negative stage is mandatory; without it the error term does not cancel."""
    assert any(c < 0 for c in YOSHIDA6_C)


# ------------------------------------------------------------- force reuse
def test_force_reuse_is_exact(double_well_grad_v) -> None:
    """Reusing the boundary force must not change the trajectory at all."""
    from integrators.symplectic import compose  # noqa: PLC0415

    q, p, a = 0.1, 1.0, double_well_grad_v(0.1)
    q1, p1, a1 = compose(q, p, 0.025, double_well_grad_v, YOSHIDA4_C, a)

    q2, p2 = q, p
    for c in YOSHIDA4_C:
        p2 = p2 - 0.5 * c * 0.025 * double_well_grad_v(q2)
        q2 = q2 + c * 0.025 * p2
        p2 = p2 - 0.5 * c * 0.025 * double_well_grad_v(q2)
    assert q1 == pytest.approx(q2, abs=1e-15)
    assert p1 == pytest.approx(p2, abs=1e-15)
    assert a1 == pytest.approx(double_well_grad_v(q2), abs=1e-15)


@pytest.mark.parametrize("method,stages", [("vverlet", 1), ("yoshida4", 3), ("yoshida6", 7)])
def test_amortised_cost_equals_stage_count(double_well_grad_v, method, stages) -> None:
    """Measured ``n_fev`` must be ``1 + stages * n_steps`` -- the reuse contract."""
    traj = symplectic_solve(method, double_well_grad_v, [0.1, 1.0], (0.0, 10.0), n_steps=500)
    assert traj.n_fev == 1 + stages * 500


def test_n_fev_per_step_helper_matches_stages() -> None:
    assert n_fev_per_step(YOSHIDA4_C) == 4
    assert n_fev_per_step(YOSHIDA6_C) == 8
    assert n_fev_per_step((1.0,)) == 2


def test_budget_steps_helper_matches_measured_cost() -> None:
    from hpo.search import budget_steps_for  # noqa: PLC0415

    budget = 7001
    assert budget_steps_for("yoshida6", budget) == budget // 7
    assert budget_steps_for("yoshida4", budget) == budget // 3
    assert budget_steps_for("vverlet", budget) == budget // 1
    assert budget_steps_for("rk4", budget) == budget // 4


# ----------------------------------------------------- symplectic behaviour
def test_symplectic_energy_error_is_bounded_not_secular(double_well_grad_v) -> None:
    """Over T=500 the symplectic drift must stay tiny."""
    traj = yoshida6(double_well_grad_v, [0.1, 1.0], (0.0, 500.0), n_steps=15000)
    energy = traj.energy_series(lambda y: 0.5 * y[1] ** 2 + 0.25 * y[0] ** 4 - 0.5 * y[0] ** 2)
    assert abs(energy[-1] - energy[0]) < 1e-9


def test_rk4_energy_loss_is_monotone_on_the_harmonic_oscillator() -> None:
    """RK4 bleeds energy monotonically on the linear oscillator.

    The monotone signature is a property of a *resolved* secular trend. It is clean here
    at a coarse step, where the per-step change is dominated by truncation error.
    """
    traj = rk4(harmonic_rhs, [1.0, 0.0], (0.0, 200.0), n_steps=20000)
    energy = traj.energy_series(lambda y: 0.5 * y[0] ** 2 + 0.5 * y[1] ** 2)
    diffs = np.diff(energy)
    diffs = diffs[diffs != 0]
    ratio = max(np.count_nonzero(diffs > 0), np.count_nonzero(diffs < 0)) / diffs.size
    assert ratio == pytest.approx(1.0, abs=1e-9)


def test_rk4_secular_drift_exceeds_its_oscillation_on_the_double_well() -> None:
    """On the double well the secular trend dominates even though signs are mixed.

    Measured at ``n=30000``: 14614 positive and 15386 negative increments, yet the net
    drift ``-3.81e-07`` far exceeds the typical increment. The per-step *sign* becomes
    noise-dominated long before the secular *trend* does, so the honest assertion is on
    the trend, not on the sign pattern.
    """
    traj = rk4(
        lambda t, y: np.array([y[1], -(y[0] ** 3) + y[0]]),
        [0.1, 1.0],
        (0.0, 500.0),
        n_steps=30000,
    )
    energy = traj.energy_series(lambda y: 0.5 * y[1] ** 2 + 0.25 * y[0] ** 4 - 0.5 * y[0] ** 2)
    diffs = np.diff(energy)
    diffs = diffs[diffs != 0]
    assert abs(energy[-1] - energy[0]) > 20.0 * float(np.mean(np.abs(diffs)))


def test_rk4_and_yoshida6_contrast_at_equal_budget(double_well_grad_v) -> None:
    """The headline structural claim, asserted directly at an equal budget."""
    from core.types import ConservedQuantity  # noqa: PLC0415
    from eval.metrics import net_drift  # noqa: PLC0415

    energy = ConservedQuantity(
        "energy", lambda y: 0.5 * y[1] ** 2 + 0.25 * y[0] ** 4 - 0.5 * y[0] ** 2
    )
    sym = yoshida6(double_well_grad_v, [0.1, 1.0], (0.0, 500.0), n_steps=15000)
    rk = rk4(
        lambda t, y: np.array([y[1], -(y[0] ** 3) + y[0]]),
        [0.1, 1.0],
        (0.0, 500.0),
        n_steps=30000,
    )
    assert abs(net_drift(sym, energy)) < abs(net_drift(rk, energy)) / 100.0


# ------------------------------------------------------------------- errors
def test_symplectic_rejects_non_separable_state() -> None:
    with pytest.raises(NotHamiltonianError):
        yoshida6(lambda q: q, [1.0, 0.0, 0.0], (0.0, 1.0), n_steps=10)


def test_unknown_symplectic_method_raises() -> None:
    with pytest.raises(IntegratorError):
        symplectic_solve("nope", lambda q: q, [1.0, 0.0], (0.0, 1.0), n_steps=10)


def test_dp54_step_budget_raises() -> None:
    def stiff(t, y):
        return np.array([-1e6 * y[0]])

    with pytest.raises(StepSizeError):
        dp54(stiff, [1.0], (0.0, 1e3), rtol=1e-14, atol=1e-16, max_steps=500)


def test_dp54_rejects_zero_span() -> None:
    with pytest.raises(IntegratorError):
        dp54(harmonic_rhs, [1.0, 0.0], (1.0, 1.0))


def test_dp54_rejects_bad_tolerance() -> None:
    with pytest.raises(IntegratorError):
        dp54(harmonic_rhs, [1.0, 0.0], (0.0, 1.0), rtol=0.0)


def test_all_methods_listed() -> None:
    assert "yoshida6" in ALL_METHODS
    assert "dp54" in ALL_METHODS
    assert "rk4" in ALL_METHODS


# ------------------------------------------------- variable-step symplectic
def test_symplectic_adaptive_runs_and_stays_finite(double_well_grad_v) -> None:
    traj = symplectic_adaptive(
        double_well_grad_v, [0.1, 1.0], (0.0, 20.0), method="yoshida6", rtol=1e-8
    )
    assert np.all(np.isfinite(traj.y))
    assert traj.n_steps > 0


def test_symplectic_adaptive_reports_its_method(double_well_grad_v) -> None:
    traj = symplectic_adaptive(double_well_grad_v, [0.1, 1.0], (0.0, 5.0), rtol=1e-6)
    assert traj.method == "symp_adaptive_yoshida6"
    assert traj.meta["variable_step"] is True
    assert traj.meta["symplectic_preserved_by"] == "uniform substep refinement"


def test_symplectic_adaptive_does_not_overflow(double_well_grad_v) -> None:
    """The controller must not walk outside the explicit method's stability interval."""
    traj = symplectic_adaptive(double_well_grad_v, [0.1, 1.0], (0.0, 50.0), rtol=1e-4, atol=1e-6)
    assert np.all(np.isfinite(traj.y))

"""Gate and pipeline integration tests.

These are the slowest tests in the suite because they actually run the acceptance
gates. They are kept in one file, and use reduced budgets, so the whole file runs in
seconds while still exercising every branch that produces a reported number.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.seed import set_all
from data.systems import build_system
from eval.baselines import (
    SKIPPED_REASON,
    best_baseline,
    default_tol_grid,
    solve_ivp_trajectory,
    tune_scipy_baseline,
)
from eval.benchmarks import (
    FLAGSHIP,
    GATE_THRESHOLDS,
    ablation,
    candidate_grid,
    equal_budget_trial,
    gate_g1,
    gate_g2,
    gate_g3,
    reference_self_consistency,
    run_method,
)
from pipeline import ODEForgePipeline

#: Gate runs that cost real wall-clock. They are the evidence behind the reported
#: numbers, so they are not skipped by default -- but they are marked so a contributor
#: can run ``pytest -m "not slow"`` for a fast inner loop and ``pytest -m slow`` to
#: reproduce the gates in isolation.
slow = pytest.mark.slow

#: A reduced budget keeps the gate integration tests fast. The ratios are what matter,
#: not the absolute counts, and the alignment logic is budget-independent.
#:
#: 60000 is the floor that works: the double well over T=500 needs h < 0.43, and
#: yoshida6 costs 7 evaluations per step, so a smaller budget implies an unstable step
#: and the solver now refuses it (by design) instead of overflowing. Below this the
#: gate cannot be measured at all, which is itself worth knowing.
FAST = Config(budget_n_fev=60_000, max_steps=200_000)
SMALL = 60_000


# ================================================================== baselines
def test_tol_grid_covers_the_mandated_powers() -> None:
    powers = sorted({round(-np.log10(rtol)) for rtol, _ in default_tol_grid()})
    assert powers == [2, 3, 4, 5, 6, 8, 10]


def test_solve_ivp_records_its_tolerances() -> None:
    spec = build_system("harmonic")
    traj = solve_ivp_trajectory(spec.rhs, spec.y0, (0.0, 1.0), rtol=1e-8, atol=1e-10)
    assert traj.meta["rtol"] == 1e-8
    assert traj.method == "scipy_DOP853"


def test_solve_ivp_propagates_failure() -> None:
    def blow_up(t, y):
        return np.array([np.inf])

    with pytest.raises(RuntimeError, match="solve_ivp failed"):
        solve_ivp_trajectory(blow_up, [1.0], (0.0, 1.0), rtol=1e-6, atol=1e-8)


def test_tune_baseline_sweeps_the_whole_grid() -> None:
    spec = build_system("harmonic")
    report = tune_scipy_baseline(
        spec.rhs, spec.y0, (0.0, 5.0), spec.conserved[0], grid=default_tol_grid()
    )
    assert report["skipped"] is False
    assert report["n_evaluated"] == len(default_tol_grid())
    assert "best_by_drift" in report
    # The trajectory-error winner only appears when a reference is supplied; without one
    # the tuner has no way to rank candidates on accuracy.
    assert "best_by_traj_error" not in report


def test_tune_baseline_ranks_accuracy_when_given_a_reference() -> None:
    spec = build_system("harmonic")
    ref = solve_ivp_trajectory(spec.rhs, spec.y0, (0.0, 5.0), rtol=1e-12, atol=1e-14)
    report = tune_scipy_baseline(
        spec.rhs,
        spec.y0,
        (0.0, 5.0),
        spec.conserved[0],
        grid=default_tol_grid(),
        ref_t=ref.t,
        ref_y=ref.y,
    )
    assert "best_by_traj_error" in report
    assert report["best_by_traj_error"]["traj_error"] >= 0.0


def test_tune_baseline_honours_a_budget() -> None:
    spec = build_system("harmonic")
    report = tune_scipy_baseline(
        spec.rhs, spec.y0, (0.0, 5.0), spec.conserved[0], budget_n_fev=5_000
    )
    assert report["best_by_drift"]["n_fev"] <= 5_000


def test_best_baseline_returns_none_when_skipped() -> None:
    assert best_baseline({"skipped": True}) is None
    assert best_baseline({"skipped": False, "best_by_drift": {"n_fev": 1}})["n_fev"] == 1


# ================================================================== run_method
@pytest.mark.parametrize("method", ["dp54", "rk4"])
def test_run_method_dispatches_generic_methods(method) -> None:
    traj = run_method(method, build_system("lorenz"), n_steps=200, config=FAST)
    assert np.all(np.isfinite(traj.y))


def test_run_method_dispatches_symplectic_methods() -> None:
    traj = run_method("yoshida6", build_system("double_well"), n_steps=8_000, config=FAST)
    assert traj.method == "yoshida6"
    assert np.all(np.isfinite(traj.y))


def test_symplectic_refuses_an_unstable_step_clearly() -> None:
    """A budget too small to stabilise the integrator must say so, not overflow.

    The double well over T=500 is stable only to about h=0.43; 500 steps implies
    h=1.0. Python floats raise OverflowError out of ``q**3 - q``, which surfaced as an
    opaque crash before the guard existed.
    """
    from core.errors import StepSizeError  # noqa: PLC0415

    with pytest.raises(StepSizeError, match="stability limit"):
        run_method("yoshida6", build_system("double_well"), n_steps=500, config=FAST)


def test_run_method_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        run_method("nope", build_system("harmonic"), config=FAST)


def test_run_method_reports_a_non_separable_system_clearly() -> None:
    with pytest.raises(ValueError, match="not separable"):
        run_method("yoshida6", build_system("kepler"), config=FAST)


# ================================================================== references
def test_reference_self_consistency_is_reported() -> None:
    out = reference_self_consistency(build_system("double_well"), steps=2_000)
    assert out["available"] is True
    assert out["self_disagreement"] >= 0.0


def test_reference_self_consistency_declines_for_non_separable() -> None:
    out = reference_self_consistency(build_system("van_der_pol"))
    assert out["available"] is False


# ================================================================== gate G1
@slow
def test_gate_g1_runs_and_reports_both_sides() -> None:
    set_all(FAST.seed)
    gate = gate_g1(config=FAST, systems=("double_well",))  # Hamiltonian: G1's domain
    assert gate["gate"] == "G1"
    assert gate["threshold"] == GATE_THRESHOLDS["G1"] == 0.10
    row = gate["rows"][0]
    assert row["status"] == "measured"
    assert row["baseline_n_fev"] > 0
    assert row["ratio"] >= 0.0
    assert "flagship_net_drift" in row
    assert "baseline_net_drift" in row


@slow
def test_gate_g1_passes_on_the_hamiltonian_systems() -> None:
    """The headline claim, asserted rather than reported: G1 must pass."""
    set_all(FAST.seed)
    gate = gate_g1(config=FAST, systems=("double_well", "pendulum"))
    assert gate["passed"] is True
    for row in gate["rows"]:
        assert row["ratio"] <= GATE_THRESHOLDS["G1"]


@slow
def test_gate_g1_records_an_unavailable_baseline_instead_of_failing() -> None:
    gate = gate_g1(config=FAST, systems=("double_well",))
    for row in gate["rows"]:
        if row["status"] != "measured":
            assert "skipped" in row
            assert "reason" in row


# ================================================================== gate G2
@slow
def test_gate_g2_aligns_both_budgets() -> None:
    """Every G2 row must show the two n_fev counts and their ratio.

    G2 measures on non-Hamiltonian systems at a budget where the adaptive solvers can
    actually spend what they are given; a much larger budget overshoots what ``dp54``
    can use and the two sides land far apart, which would make the alignment assertion
    meaningless rather than informative.
    """
    set_all(FAST.seed)
    budget_n = 20_000
    gate = gate_g2(config=FAST, systems=("van_der_pol",), budget_n_fev=budget_n)
    row = gate["rows"][0]
    assert row["status"] == "measured"
    budget = row["budget"]
    assert budget["target_n_fev"] == budget_n
    assert budget["flagship_n_fev"] > 0
    assert budget["baseline_n_fev"] > 0
    assert 0.95 < budget["n_fev_ratio"] < 1.05
    assert row["is_hamiltonian"] is False
    # The reference must be able to resolve what it is judging.
    assert row["reference"]["cross_family_disagreement"] < min(
        row["flagship_traj_error"], row["baseline_traj_error"]
    )


@slow
def test_gate_g2_threshold_is_non_inferiority() -> None:
    assert GATE_THRESHOLDS["G2"] == 1.0
    set_all(FAST.seed)
    gate = gate_g2(config=FAST, systems=("van_der_pol",), budget_n_fev=20_000)
    assert gate["threshold"] == 1.0
    assert isinstance(gate["passed"], bool)


@slow
def test_gate_g2_reports_skips_without_crashing() -> None:
    gate = gate_g2(config=FAST, systems=("van_der_pol",), budget_n_fev=20_000)
    assert gate["gate"] == "G2"
    assert isinstance(gate["n_skipped"], int)


# ================================================================== gate G3
@slow
def test_gate_g3_delegates_to_scipy() -> None:
    set_all(FAST.seed)
    gate = gate_g3(config=FAST, systems=("van_der_pol",), budget_n_fev=SMALL)
    row = gate["rows"][0]
    assert row["status"] == "measured"
    assert row["delegated"] is True
    assert row["routed_method"] == "scipy_DOP853"
    assert row["ratio"] == pytest.approx(1.0, abs=0.05)


@slow
def test_gate_g3_budgets_match_exactly_when_delegated() -> None:
    """A delegated flagship runs the same solver, so the budgets must agree exactly."""
    set_all(FAST.seed)
    gate = gate_g3(config=FAST, systems=("lorenz",), budget_n_fev=SMALL)
    budget = gate["rows"][0]["budget"]
    assert budget["flagship_n_fev"] == budget["baseline_n_fev"]


@slow
def test_gate_g3_passes_within_five_percent() -> None:
    set_all(FAST.seed)
    gate = gate_g3(config=FAST, systems=("van_der_pol", "lorenz"), budget_n_fev=SMALL)
    assert gate["passed"] is True
    for row in gate["rows"]:
        assert row["ratio"] <= 1.0 / GATE_THRESHOLDS["G3"]


# ============================================================== equal-budget
@slow
def test_equal_budget_trial_shape() -> None:
    set_all(FAST.seed)
    trial = equal_budget_trial(build_system("double_well"), budget_n_fev=SMALL, config=FAST)
    assert trial["flagship"] == FLAGSHIP
    assert trial["flagship_net_drift"]["n"] == 3
    assert trial["flagship_net_drift"]["std"] >= 0.0
    assert "rk4_net_drift" in trial
    assert "scipy_available" in trial


def test_candidate_grid_hits_the_budget() -> None:
    grid = candidate_grid(70_000)
    flagship = [c for c in grid if c.method == "yoshida6"]
    assert flagship, "the flagship method must appear in its own search grid"
    # The first entry spends the whole budget: 7 evaluations per step with force reuse.
    assert flagship[0].n_steps == 70_000 // 7
    # Every entry must stay inside the budget it was given.
    for cand in grid:
        per_step = {"yoshida6": 7, "yoshida4": 3, "rk4": 4, "vverlet": 1}[cand.method]
        assert (cand.n_steps or 0) * per_step <= 70_000


# ================================================================== ablation
@slow
def test_ablation_has_at_least_four_variants() -> None:
    set_all(FAST.seed)
    rows = ablation(config=FAST, system="double_well")
    assert len(rows) >= 4
    names = {r["variant"] for r in rows}
    assert {"full", "no_ham", "no_cons", "dp54_only"} <= names


@slow
def test_ablation_no_cons_and_dp54_only_are_the_same_path() -> None:
    """The two arms coincide by construction, and that must be explicit.

    The conserved-quantity monitor only ever adds a *vote*; removing it changes the
    tally but not the routed method on this system, so both arms execute the same code.
    Presenting them as two independent confirmations would misrepresent the evidence.
    """
    set_all(FAST.seed)
    rows = {r["variant"]: r for r in ablation(config=FAST, system="double_well")}
    no_cons, dp54_only = rows["no_cons"], rows["dp54_only"]
    assert no_cons["n_fev"] == dp54_only["n_fev"]
    assert no_cons["net_drift"] == dp54_only["net_drift"]


@slow
def test_ablation_full_beats_the_downgrade() -> None:
    set_all(FAST.seed)
    rows = {r["variant"]: r for r in ablation(config=FAST, system="double_well")}
    assert rows["full"]["net_drift"] < rows["y4_only"]["net_drift"]


# ============================================================== pipeline API
@slow
def test_pipeline_benchmark_report_shape() -> None:
    report = ODEForgePipeline(Config(budget_n_fev=60_000)).benchmark_report(quick=True)
    assert set(report["gates"]) == {"G1", "G2", "G3"}
    assert report["thresholds"] == GATE_THRESHOLDS
    assert "systems" in report
    assert report["ablation"]
    assert isinstance(report["passed"]["all"], bool)


@slow
def test_pipeline_quick_benchmark_covers_the_demo_systems() -> None:
    report = ODEForgePipeline(Config(budget_n_fev=60_000)).benchmark_report(quick=True)
    names = {row["system"] for row in report["systems"]}
    assert {"harmonic", "double_well", "lorenz"} <= names


@slow
def test_pipeline_records_errors_instead_of_raising() -> None:
    report = ODEForgePipeline(Config(budget_n_fev=60_000)).benchmark_report(quick=True)
    for row in report["systems"]:
        assert "error" in row or "metrics" in row


def test_skipped_reason_is_explicit() -> None:
    assert "scipy" in SKIPPED_REASON

"""Routing contracts: Hamiltonian detection, the SympFuse vote, and delegation."""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import HamiltonianError
from data.systems import SYSTEM_NAMES, build_system
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


def make_fuse(**kwargs) -> SympFuse:
    """A SympFuse with a default detector, so tests read as one line."""
    return SympFuse(detector=HamiltonianDetector(), **kwargs)


HAMILTONIAN = ("harmonic", "double_well", "pendulum")
NON_HAMILTONIAN = ("van_der_pol", "lorenz", "robertson", "exponential")


# ------------------------------------------------------------ structural tests
def test_linear_structure_residual_is_zero_for_a_hamiltonian_jacobian() -> None:
    """A = [[0, 1], [-1, 0]] admits S = I, so A^T S + S A = 0 exactly."""
    a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert linear_hamiltonian_residual(a) < 1e-12


def test_linear_structure_residual_is_large_for_a_dissipative_jacobian() -> None:
    a = np.array([[0.0, 1.0], [-11.0, -1.0]])  # Van der Pol
    assert linear_hamiltonian_residual(a) > 1e-2


def test_svd_residual_agrees_with_the_2x2_closed_form() -> None:
    """Two independent formulations of the same condition must agree."""
    for a in (
        np.array([[0.0, 1.0], [-1.0, 0.0]]),
        np.array([[0.0, 1.0], [-11.0, -1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.73, 0.0]]),
    ):
        svd = linear_hamiltonian_residual(a)
        closed = matrix_hamiltonian_residual_2x2(a)
        if closed < 1e-12:
            assert svd < 1e-8
        else:
            assert svd > 1e-3


def test_harmonic_jacobian_is_antisymmetric_not_symmetric() -> None:
    """The commonly quoted "Hamiltonian Jacobians are symmetric" is false.

    For ``H = p^2/2 + q^2/2`` the Jacobian is ``[[0, 1], [-1, 0]]``, whose symmetric part
    is exactly zero. Any test that assumes symmetry would reject the harmonic oscillator.
    """
    spec = build_system("harmonic")
    jac = numerical_jacobian(spec.rhs, 0.0, np.array([0.7, -0.3]))
    assert np.max(np.abs(jac + jac.T)) < 1e-9
    assert np.max(np.abs(jac - jac.T)) > 1.0


def test_divergence_vanishes_for_hamiltonian_and_not_otherwise() -> None:
    assert abs(divergence(build_system("harmonic").rhs, 0.0, np.array([1.0, 0.5]))) < 1e-9
    assert abs(divergence(build_system("van_der_pol").rhs, 0.0, np.array([2.0, 0.5]))) > 0.5


def test_involutivity_alone_misclassifies_van_der_pol() -> None:
    """Documented false-positive risk: VdP is odd, so involutivity is perfect.

    This is why the detector requires all three indicators rather than thresholding a
    weighted mean.
    """
    spec = build_system("van_der_pol")
    points = probe_cloud(spec.y0, 12)
    assert involutivity_error(spec.rhs, points) < 1e-12


# ------------------------------------------------------------------ detector
@pytest.mark.parametrize("name", HAMILTONIAN)
def test_detector_accepts_hamiltonian_systems(name) -> None:
    report = HamiltonianDetector().analyse(build_system(name))
    assert report.verdict is True
    assert report.score > 0.9


@pytest.mark.parametrize("name", NON_HAMILTONIAN)
def test_detector_rejects_non_hamiltonian_systems(name) -> None:
    report = HamiltonianDetector().analyse(build_system(name))
    assert report.verdict is False


def test_detector_agrees_with_the_declared_flag() -> None:
    detector = HamiltonianDetector()
    for name in SYSTEM_NAMES:
        spec = build_system(name)
        report = detector.analyse(spec)
        assert report.verdict == spec.is_hamiltonian, name


def test_detector_is_deterministic() -> None:
    spec = build_system("double_well")
    a = HamiltonianDetector().analyse(spec)
    b = HamiltonianDetector().analyse(spec)
    assert a.as_dict() == b.as_dict()


def test_detector_records_seed_and_probe_count() -> None:
    report = HamiltonianDetector(n_probe=17).analyse(build_system("harmonic"))
    assert report.n_probe == 17
    assert isinstance(report.seed, int)


def test_detector_rejects_bad_configuration() -> None:
    with pytest.raises(HamiltonianError):
        HamiltonianDetector(n_probe=1)
    with pytest.raises(HamiltonianError):
        HamiltonianDetector(threshold=1.5)


def test_probe_cloud_is_reproducible() -> None:
    a = probe_cloud(np.array([1.0, 0.0]), 8)
    b = probe_cloud(np.array([1.0, 0.0]), 8)
    assert np.array_equal(a, b)


def test_conserved_quantity_discovery_finds_mass_for_robertson() -> None:
    spec = build_system("robertson")
    points = probe_cloud(spec.y0, 16)
    found = discover_linear_conserved(spec.rhs, points)
    assert found["found"] is True
    assert found["residual"] < 1e-8


# ------------------------------------------------------------------- SympFuse
def test_route_separable_hamiltonian_to_symplectic() -> None:
    fuse = make_fuse()
    decision = fuse.route(build_system("double_well"))
    assert decision.method == "yoshida6"
    assert all(decision.votes.values())


def test_route_holds_back_hamiltonian_but_non_separable() -> None:
    """Kepler is Hamiltonian yet not separable, so a symplectic method cannot apply."""
    decision = make_fuse().route(build_system("kepler"))
    assert decision.method == "dp54"
    assert "not separable" in decision.reason


def test_route_non_hamiltonian_to_adaptive() -> None:
    decision = make_fuse().route(build_system("van_der_pol"))
    assert decision.method == "dp54"
    assert not any(decision.votes.values())


def test_route_is_deterministic() -> None:
    fuse = make_fuse()
    for name in ("double_well", "lorenz", "kepler"):
        spec = build_system(name)
        assert fuse.route(spec).as_dict() == fuse.route(spec).as_dict()


def test_route_accepts_precomputed_monitor() -> None:
    fuse = make_fuse()
    spec = build_system("double_well")
    monitor = monitor_conserved(spec)
    assert fuse.route(spec, conserved=monitor).method == "yoshida6"


def test_route_degrades_when_the_invariant_drifts() -> None:
    """A symplectic vote contradicted by invariant drift must degrade, not proceed."""
    fuse = make_fuse(conserved_tol=1e-12)
    decision = fuse.route(
        build_system("double_well"),
        conserved={
            "available": True,
            "max_rel_drift": 1.0,
            "n_quantities": 1,
            "per_quantity": {"energy": 1.0},
        },
    )
    assert decision.method == "dp54"
    assert decision.degraded is True
    assert "alarm" in decision.detail


def test_route_tie_resolves_conservatively() -> None:
    """A tie must fall back to the adaptive solver, which is never wrong, only slower."""
    fuse = make_fuse()
    decision = fuse.route(
        build_system("double_well"),
        conserved={"available": False, "max_rel_drift": 0.0, "n_quantities": 0, "per_quantity": {}},
    )
    assert decision.method == "dp54"
    assert "tie" in decision.reason or "split" in decision.reason


def test_route_rejects_an_unknown_symplectic_method() -> None:
    with pytest.raises(Exception):  # noqa: B017 -- IntegratorError subclass
        make_fuse(symplectic_method="euler")


def test_route_many_covers_every_system() -> None:
    fuse = make_fuse()
    specs = {n: build_system(n) for n in ("harmonic", "van_der_pol", "kepler")}
    decisions = fuse.route_many(specs)
    assert set(decisions) == set(specs)


# ---------------------------------------------------------------- delegation
def test_delegation_only_for_non_hamiltonian() -> None:
    fuse = make_fuse()
    for name in HAMILTONIAN:
        assert fuse.delegates_to_scipy(build_system(name)) is False
    for name in NON_HAMILTONIAN:
        assert fuse.delegates_to_scipy(build_system(name)) is True


def test_available_scipy_matches_importability() -> None:
    try:
        import scipy.integrate  # noqa: F401, PLC0415
    except Exception:
        assert available_scipy() is False
    else:
        assert available_scipy() is True


def test_monitor_reports_declared_invariants() -> None:
    mon = monitor_conserved(build_system("double_well"))
    assert mon["available"] is True
    assert mon["n_quantities"] == 1
    assert mon["max_rel_drift"] < 1e-6


def test_monitor_without_invariants_is_not_evidence() -> None:
    mon = monitor_conserved(build_system("lorenz"))
    assert mon["available"] is False
    assert mon["max_rel_drift"] == 0.0

"""Tests for the cross-validated reference solution.

The reference is the yardstick every accuracy gate is measured against, so its own
properties are asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import ODEForgeError
from data.systems import build_system
from eval.baselines import available_scipy
from eval.reference import StrictReference, build_strict_reference, require_resolved


def test_reference_reports_its_ingredients() -> None:
    ref = build_strict_reference(build_system("van_der_pol"), candidates_n_fev=5_000)
    payload = ref.as_dict()
    for key in (
        "reference_method",
        "reference_n_fev",
        "cross_family_disagreement",
        "candidates_n_fev",
        "resolved",
    ):
        assert key in payload


@pytest.mark.skipif(not available_scipy(), reason="cross-validation needs scipy")
def test_two_independent_families_are_used() -> None:
    """DOP853 (explicit) and Radau (implicit) must both appear.

    Two explicit RK methods would share an error estimator and a step controller, so
    their agreement is weaker evidence. Measured: dp54 + DOP853 agreed to only 1.24e-05
    after 17 s, while DOP853 + Radau agree to ~1e-12 in 3.4 s.
    """
    ref = build_strict_reference(build_system("van_der_pol"), candidates_n_fev=5_000)
    assert "DOP853" in ref.method
    assert "Radau" in ref.method


@pytest.mark.skipif(not available_scipy(), reason="needs scipy")
def test_reference_is_far_more_accurate_than_the_candidates() -> None:
    """The whole point: the yardstick must out-resolve what it measures.

    The candidates sit at ~1.3e-05. If the reference's own cross-family disagreement
    were of that order, the ratio would be a comparison of two noise floors -- which is
    exactly how an earlier revision read 52.1 instead of 1.04 on Lorenz.
    """
    ref = build_strict_reference(build_system("van_der_pol"), candidates_n_fev=5_000)
    assert ref.cross_family_disagreement < 1e-10, (
        f"reference disagreement {ref.cross_family_disagreement:.3e} is too coarse to "
        "adjudicate a ~1e-05 difference"
    )


@pytest.mark.skipif(not available_scipy(), reason="needs scipy")
def test_tighter_tolerance_tightens_agreement() -> None:
    """Monotonicity: a tighter reference must not be less trustworthy."""
    spec = build_system("van_der_pol")
    loose = build_strict_reference(spec, candidates_n_fev=5_000, rtol=1e-10)
    tight = build_strict_reference(spec, candidates_n_fev=5_000, rtol=1e-13)
    assert tight.cross_family_disagreement < loose.cross_family_disagreement


@pytest.mark.skipif(not available_scipy(), reason="needs scipy")
def test_reference_is_finite_and_on_a_monotone_grid() -> None:
    ref = build_strict_reference(build_system("lorenz"), candidates_n_fev=5_000)
    assert np.all(np.isfinite(ref.y))
    assert np.all(np.diff(ref.t) > 0)
    assert ref.t[0] == pytest.approx(build_system("lorenz").t_span[0])
    assert ref.t[-1] == pytest.approx(build_system("lorenz").t_span[1])


def test_require_resolved_raises_for_an_unresolvable_reference() -> None:
    bad = StrictReference(
        t=np.array([0.0, 1.0]),
        y=np.zeros((2, 2)),
        n_fev=10,
        method="none",
        cross_family_disagreement=float("inf"),
        candidates_n_fev=1_000_000,
        resolved=False,
        note="deliberately unresolvable",
    )
    with pytest.raises(ODEForgeError, match="not demonstrably more accurate"):
        require_resolved(bad)


def test_require_resolved_accepts_a_good_reference() -> None:
    good = StrictReference(
        t=np.array([0.0, 1.0]),
        y=np.zeros((2, 2)),
        n_fev=100_000,
        method="DOP853+Radau",
        cross_family_disagreement=1e-12,
        candidates_n_fev=1_000,
        resolved=True,
        note="fine",
    )
    require_resolved(good)  # must not raise

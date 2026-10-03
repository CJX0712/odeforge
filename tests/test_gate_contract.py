"""Guard against "metric hallucination": the prose must match the code.

The most dangerous class of defect in this project is not a wrong number -- it is a
*correct* number attached to a *stale* description. During review, the module docstring
still advertised G2 as "at most 50%" and G3 as "at least 95%" while ``GATE_THRESHOLDS``
held ``1.0`` and ``0.95``. Anyone reading the docstring would have drawn the wrong
conclusion from a passing run.

These tests parse the prose and cross-check it against the executable constants, so the
two cannot drift apart silently again.
"""

from __future__ import annotations

import inspect
import re

import pytest

from eval import benchmarks
from eval.benchmarks import (
    FLAGSHIP,
    G1_SYSTEMS,
    G2_SYSTEMS,
    GATE_THRESHOLDS,
    NON_HAMILTONIAN_SYSTEMS,
    gate_g1,
    gate_g2,
    gate_g3,
)
from eval.benchmarks import (
    G2_SYSTEMS as _G2,
)

MODULE_DOC = benchmarks.__doc__ or ""


def test_module_docstring_exists() -> None:
    assert len(MODULE_DOC) > 500, "the gate contract must be documented in prose"


def test_docstring_states_the_real_g2_threshold() -> None:
    """The docstring must say 100% (non-inferiority), not the retired 50%."""
    assert "**100%**" in MODULE_DOC, "docstring G2 threshold drifted from GATE_THRESHOLDS"
    assert "**50%**" not in MODULE_DOC, "the retired 0.50 G2 threshold is still in the prose"
    assert "at most **50%**" not in MODULE_DOC


def test_docstring_states_the_real_g1_threshold() -> None:
    assert "**10%**" in MODULE_DOC
    assert GATE_THRESHOLDS["G1"] == 0.10


def test_docstring_percentages_match_the_threshold_table() -> None:
    """Every percentage in the gate summary must correspond to a real threshold."""
    # G1 is 0.10 -> "10%", G2 is 1.0 -> "100%". G3 is 0.95 -> "5%" as a regression band.
    expected = {
        "G1": f"**{GATE_THRESHOLDS['G1'] * 100:.0f}%**",
        "G2": f"**{GATE_THRESHOLDS['G2'] * 100:.0f}%**",
    }
    for gate, token in expected.items():
        assert token in MODULE_DOC, f"{gate} threshold text {token!r} missing from docstring"


def test_no_gate_threshold_is_advertised_beyond_the_table() -> None:
    """A percentage in the docstring that matches no threshold is a drift smell."""
    allowed = {f"{GATE_THRESHOLDS[k] * 100:.0f}" for k in ("G1", "G2", "G3")}
    allowed.add("5")  # G3's 1/0.95 regression band, stated as "within 5%"
    found = set(re.findall(r"\*\*(\d+)%\*\*", MODULE_DOC))
    assert found <= allowed, f"docstring advertises percentages not in GATE_THRESHOLDS: {found}"


def test_gate_functions_document_their_own_threshold() -> None:
    """Each ``gate_gN`` docstring must mention its threshold value."""
    for name, key in (("gate_g1", "G1"), ("gate_g2", "G2"), ("gate_g3", "G3")):
        doc = inspect.getdoc(getattr(benchmarks, name)) or ""
        assert doc, f"{name} has no docstring"
        # G3's threshold is a regression band, so accept either form.
        thr = GATE_THRESHOLDS[key]
        if key == "G3":
            assert ("5%" in doc) or (f"{thr:.2f}" in doc), f"{name} docstring omits its threshold"
        else:
            assert f"{thr:g}" in doc or f"{thr * 100:.0f}%" in doc, (
                f"{name} docstring does not state threshold {thr}"
            )


# ------------------------------------------------------------- the G1/G2 split
def test_g1_and_g2_target_different_mathematical_objects() -> None:
    """G1 is Hamiltonian; G2 must be non-Hamiltonian. Mixing them is the bug under test."""
    from data.systems import build_system  # noqa: PLC0415

    assert set(G1_SYSTEMS).isdisjoint(set(G2_SYSTEMS)), "G1 and G2 must not share systems"
    for name in G1_SYSTEMS:
        assert build_system(name).is_hamiltonian, f"{name} is not Hamiltonian but is in G1"
    for name in G2_SYSTEMS:
        assert not build_system(name).is_hamiltonian, (
            f"{name} is Hamiltonian; putting it in G2 re-conflates the two gates"
        )


def test_g2_systems_are_non_hamiltonian() -> None:
    assert _G2 == G2_SYSTEMS
    assert "van_der_pol" in G2_SYSTEMS
    assert "lorenz" in G2_SYSTEMS


def test_g3_systems_superset_of_g2() -> None:
    """G3 covers the same family plus the stiff case."""
    assert set(G2_SYSTEMS) <= set(NON_HAMILTONIAN_SYSTEMS)


def test_module_docstring_explains_the_split() -> None:
    assert "G1_SYSTEMS" in MODULE_DOC
    assert "G2_SYSTEMS" in MODULE_DOC
    assert "different mathematical objects" in MODULE_DOC


def test_flagship_is_a_symplectic_method() -> None:
    assert FLAGSHIP.startswith("yoshida")


# ------------------------------------------------------------------- thresholds
def test_thresholds_are_the_agreed_values() -> None:
    """Pinned so a well-meaning edit cannot quietly relax a gate."""
    assert GATE_THRESHOLDS["G1"] == 0.10
    assert GATE_THRESHOLDS["G2"] == 1.0
    assert GATE_THRESHOLDS["G3"] == 0.95


def test_g2_threshold_is_non_inferiority_not_a_target() -> None:
    """G2 at 1.0 means 'must not be worse'. Anything below 1.0 would demand
    superiority, which the measured order gap makes unreachable -- and inflating it to
    1.05 to turn the gate green would be fitting the threshold to the result."""
    assert GATE_THRESHOLDS["G2"] == 1.0


def test_gate_entry_points_exist() -> None:
    for fn in (gate_g1, gate_g2, gate_g3):
        assert callable(fn)


@pytest.mark.parametrize("gate", ["G1", "G2", "G3"])
def test_every_gate_threshold_is_used_by_its_function(gate) -> None:
    """The table entry must actually be read by the gate it names."""
    src = inspect.getsource(getattr(benchmarks, f"gate_{gate.lower()}"))
    assert 'GATE_THRESHOLDS["' + gate + '"]' in src or f"GATE_THRESHOLDS[{gate!r}]" in src

"""Shared pytest fixtures and the determinism/autodiff guard rails."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.seed import set_all  # noqa: E402

#: Global test seed. Every test that touches randomness uses this.
TEST_SEED = 20260101


@pytest.fixture(autouse=True)
def _seeded() -> None:
    """Re-seed before every test so order cannot change results."""
    set_all(TEST_SEED)


@pytest.fixture
def cfg() -> Config:
    """A default configuration for tests."""
    return Config()


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator for test inputs (never the global legacy state)."""
    return np.random.default_rng(TEST_SEED)


@pytest.fixture
def harmonic_grad_v():
    """dV/dq for the unit harmonic oscillator."""
    return lambda q: q


@pytest.fixture
def double_well_grad_v():
    """dV/dq for the double well."""
    return lambda q: q**3 - q

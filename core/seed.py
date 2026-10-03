"""The single global seed entry point.

Rule enforced throughout ODEForge: **all** randomness originates from
:func:`set_all`. No module calls ``np.random.seed`` or ``random.seed`` directly, and
no code path uses an entropy source that bypasses :class:`numpy.random.Generator`.

Determinism contract
--------------------
Two runs of ``examples/run_demo.py`` in the same environment must produce
bit-identical numbers in ``benchmark.json``, except for ``wall_sec`` fields.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["get_seed", "rng", "seed_state", "set_all"]

_CURRENT_SEED: int | None = None


def set_all(seed: int) -> int:
    """Seed every stochastic source ODEForge uses and record the seed.

    This is the *only* supported way to seed the library. Calling it twice is
    harmless and returns the seed both times.
    """
    global _CURRENT_SEED  # noqa: PLW0603 -- module-level registry, intentional
    value = int(seed)
    if value < 0:
        msg = f"seed must be non-negative, got {value}"
        raise ValueError(msg)
    np.random.seed(value)  # legacy global state, kept for third-party callbacks
    random.seed(value)
    _CURRENT_SEED = value
    os.environ.setdefault("PYTHONHASHSEED", str(value))
    return value


def get_seed() -> int | None:
    """Return the seed most recently passed to :func:`set_all`, or ``None``."""
    return _CURRENT_SEED


def rng(offset: int = 0) -> np.random.Generator:
    """Return a generator derived deterministically from the global seed.

    ``offset`` lets a subsystem take its own independent stream without changing any
    other component's numbers. This is the *only* sanctioned way to draw random probe
    points, and the offset is always recorded alongside the result.
    """
    if _CURRENT_SEED is None:
        set_all(0)
    assert _CURRENT_SEED is not None
    return np.random.default_rng([_CURRENT_SEED, int(offset)])


def seed_state() -> dict[str, Any]:
    """Snapshot of the seed state, for inclusion in benchmark artefacts."""
    return {
        "global_seed": _CURRENT_SEED,
        "numpy_bit_generator": "MT19937",
        "legacy_seeded": _CURRENT_SEED is not None,
    }

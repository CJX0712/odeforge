"""ODEForge differentiable integration.

* :mod:`autodiff.tape` -- reverse-mode taping over a :class:`~autodiff.tape.Value` scalar
* :mod:`autodiff.vector` -- vectorised ops on ``Value`` (batch-friendly helpers)
* :mod:`autodiff.adjoint` -- adjoint (Lagrangian) method for O(1)-memory gradients

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from autodiff.tape import (
    GradCheckError,
    Tape,
    Value,
    central_difference_check,
    leaf_gradients,
    reset_tape,
    tape_stats,
)

__all__ = [
    "GradCheckError",
    "Tape",
    "Value",
    "central_difference_check",
    "leaf_gradients",
    "reset_tape",
    "tape_stats",
]

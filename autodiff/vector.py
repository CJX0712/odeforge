"""Vectorised helpers for :class:`~autodiff.tape.Value`.

These wrap the scalar tape in operations that treat a leading axis as a batch, so a
whole ensemble of initial conditions can be integrated in one call. The point is
*throughput*, not new autodiff capability: every function here is expressible with the
scalar primitives and produces identical gradients.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from autodiff.tape import Value

__all__ = [
    "as_value",
    "batch_map",
    "grad_vector_norm",
    "mean_square_error",
    "stack_batch",
    "vdot",
    "weighted_sum_square",
]


def as_value(data: object, *, requires_grad: bool = True) -> Value:
    """Coerce ``data`` to a :class:`Value` (no-op if already one)."""
    return data if isinstance(data, Value) else Value(data, requires_grad=requires_grad)


def stack_batch(parts: Sequence[Value]) -> Value:
    """Stack a list of ``Value`` leaves along a new leading axis."""
    return Value.stack([as_value(p) for p in parts], axis=0)


def batch_map(fn: Callable[[Value], Value], batch: Value) -> Value:
    """Apply ``fn`` independently to every element of a batched ``Value``.

    ``batch`` is treated as a flat list of scalars regardless of its shape, so a caller
    can integrate many initial conditions with one graph.
    """
    n = batch.data.size
    outs = [fn(batch[i]) for i in range(n)]
    return Value.stack(outs, axis=0)


def weighted_sum_square(residual: Value, weights: np.ndarray | None = None) -> Value:
    """Return ``sum(w_i * r_i^2)``, the standard least-squares objective."""
    r = as_value(residual)
    sq = r * r
    total = sq.sum()
    if weights is None:
        return total
    w = np.asarray(weights, dtype=np.float64)
    if w.size != r.data.size:
        msg = f"weights size {w.size} does not match residual size {r.data.size}"
        raise ValueError(msg)
    # Weighting is a constant scaling, so fold it in without extra graph nodes.
    return total * float(np.sum(w * r.data**2) / max(1e-300, float(np.sum(r.data**2))))


def vdot(a: Value, b: np.ndarray | Value) -> Value:
    """Inner product of two vectors, taped through ``a``."""
    x = as_value(a)
    y = as_value(b)
    return (x * y).sum()


def mean_square_error(pred: Value, target: np.ndarray) -> Value:
    """``mean((pred - target)^2)`` with ``target`` a constant array."""
    p = as_value(pred)
    t = np.asarray(target, dtype=np.float64)
    if t.shape != p.data.shape:
        msg = f"shape mismatch: prediction {p.data.shape} vs target {t.shape}"
        raise ValueError(msg)
    diff = p - t
    return (diff * diff).mean()


def grad_vector_norm(loss: Value) -> np.ndarray:
    """Return the accumulated gradient of every leaf, concatenated.

    Convenience for optimisers that want a single flat parameter vector.
    """
    parts: list[np.ndarray] = []
    for node in loss._tape.leaves:
        parts.append(np.zeros_like(node.data) if node.grad is None else node.grad.ravel())
    return np.concatenate(parts) if parts else np.zeros(0)

"""Reverse-mode automatic differentiation by taping.

A :class:`Value` records every primitive operation it performs onto a tape. When the
final scalar is :meth:`Value.backward`-ed, the tape is walked in reverse and each node
contributes ``vjp @ grad_out`` to its inputs' adjoint accumulators.

Supported primitives
--------------------
===================== ==================================================
category              operations
===================== ==================================================
arithmetic            ``+ - * / neg``, ``pow``, ``reciprocal``
transcendental        ``exp log sin cos tanh sqrt abs``
matrices              ``matmul``, ``T`` (transpose)
reductions            ``sum``, ``mean``, ``max``
shapes                ``reshape``, ``squeeze``, ``unsqueeze``
indexing              ``__getitem__`` (integer, slice, tuple, boolean mask)
combination           ``concat``, ``stack``
norms                 ``linalg.norm`` (Frobenius and 2-norm)
===================== ==================================================

Why taping rather than forward mode
-----------------------------------
Gradients of a scalar loss with respect to ``n`` parameters cost one reverse sweep
regardless of ``n``, whereas forward mode costs ``n`` sweeps. The training task in
:mod:`training.task` differentiates through a whole ODE integration, so reverse mode is
the right choice; the memory cost is the number of live nodes, which is bounded by the
integration length and is reported by :func:`tape_stats`.

The golden standard enforced by the test-suite is a **central finite difference** check
with relative error below ``1e-6``.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "GradCheckError",
    "Tape",
    "Value",
    "central_difference_check",
    "leaf_gradients",
    "reset_tape",
    "tape_stats",
]

_TOL = 1e-6

#: numpy ufunc name -> taped unary method, installed after the class body.
_UFUNC_DISPATCH: dict[str, Any] = {}


class GradCheckError(AssertionError):
    """Raised when an analytic gradient disagrees with finite differences."""


class _Node:
    """One taped operation: inputs, output, and the vector-Jacobian closure."""

    __slots__ = ("inputs", "output", "vjp")

    def __init__(
        self, inputs: tuple[Value, ...], output: Value, vjp: Callable[[np.ndarray], None]
    ) -> None:
        self.inputs = inputs
        self.output = output
        self.vjp = vjp


class Tape:
    """A linear list of taped nodes plus the leaves that must outlive it.

    ``backward`` clears the node list to release memory, so leaf gradients are
    additionally retained in :attr:`leaves`. Without that, a caller could not read
    ``d loss / d theta`` after the sweep, which is the only reason to run it.
    """

    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self.leaves: list[Value] = []

    def append(
        self, inputs: tuple[Value, ...], output: Value, vjp: Callable[[np.ndarray], None]
    ) -> None:
        self.nodes.append(_Node(inputs, output, vjp))

    def clear(self) -> None:
        self.nodes.clear()

    def __len__(self) -> int:
        return len(self.nodes)

    def stats(self) -> dict[str, int]:
        return {"n_nodes": len(self.nodes), "n_leaves": len(self.leaves)}


#: The tape used when a ``Value`` is constructed without an explicit one.
_GLOBAL_TAPE = Tape()


class Value:
    """A scalar-or-array node participating in reverse-mode autodiff."""

    __slots__ = ("_prev", "_scalar", "_tape", "data", "grad", "requires_grad")

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = True,
        _prev: tuple[Value, ...] = (),
        _tape: Tape | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=np.float64)
        # A genuine 0-d input is a SCALAR node. Promoting it to shape (1,) made
        # ``y[0]`` return shape (1,) instead of a scalar, which broke arithmetic that
        # mixes indexed states (a 2-vector rhs) and silently changed broadcasting.
        self._scalar = arr.ndim == 0
        if self._scalar:
            arr = arr.reshape(1)
        self.data = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = bool(requires_grad)
        self._prev = _prev
        self._tape = _tape if _tape is not None else _GLOBAL_TAPE
        if not _prev and self.requires_grad:
            # A parentless node is a leaf: it receives gradient directly and must
            # remain reachable after the node list is released.
            self._tape.leaves.append(self)

    # ------------------------------------------------------------------ plumbing
    @property
    def shape(self) -> tuple[int, ...]:
        return () if self._scalar else self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def _record(
        self, out: np.ndarray, inputs: tuple[Value, ...], vjp: Callable[[np.ndarray], None]
    ) -> Value:
        node = Value(
            out, requires_grad=any(i.requires_grad for i in inputs), _prev=inputs, _tape=self._tape
        )
        node._scalar = inputs[0]._scalar
        self._tape.append(inputs, node, vjp)
        return node

    def _accum(self, g: np.ndarray) -> None:
        """Accumulate an incoming adjoint, normalising shapes to this node's storage.

        A scalar node stores shape ``(1,)`` but vjp closures naturally produce shape
        ``()`` for it (numpy scalar arithmetic). Without this normalisation the ``+=``
        raises ``non-broadcastable output operand``, which aborted the whole reverse
        sweep and left every parameter gradient at zero.
        """
        if not self.requires_grad:
            return
        arr = np.asarray(g, dtype=np.float64)
        if arr.shape != self.data.shape:
            arr = arr.reshape(self.data.shape)
        if self.grad is None:
            self.grad = np.array(arr, copy=True)
        else:
            self.grad = self.grad + arr

    def zero_grad(self) -> None:
        """Reset this node's adjoint accumulator."""
        self.grad = None

    def _as_arr(self, other: Any) -> np.ndarray:
        return other.data if isinstance(other, Value) else np.asarray(other, dtype=np.float64)

    def backward(self, seed: np.ndarray | float | None = None) -> None:
        """Reverse-accumulate gradients of this node back through the tape.

        The adjoint of ``self`` is set to ``seed`` (default ``1.0``) and the tape is
        then walked in reverse. Intermediate adjoints are cleared first so a reused tape
        cannot carry stale values into the result.
        """
        out_grad = np.array(seed if seed is not None else 1.0, dtype=np.float64).reshape(
            self.data.shape
        )
        for leaf in self._tape.leaves:
            leaf.grad = None
        for node in self._tape.nodes:
            node.output.grad = None
        self.grad = out_grad
        for node in reversed(self._tape.nodes):
            g = node.output.grad
            if g is None:
                continue
            node.vjp(g.copy())
        # Node list is released, but leaf gradients persist for the caller to read.
        self._tape.nodes.clear()

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other: Any) -> Value:
        o = self._as_arr(other)
        inputs = (self,) if not isinstance(other, Value) else (self, other)

        def vjp(g: np.ndarray, a=self, b=other) -> None:
            a._accum(g)
            if isinstance(b, Value):
                b._accum(g)

        return self._record(self.data + o, inputs, vjp)

    __radd__ = __add__

    def __neg__(self) -> Value:
        return self._record(-self.data, (self,), lambda g, a=self: a._accum(-g))

    def __sub__(self, other: Any) -> Value:
        o = self._as_arr(other)
        inputs = (self,) if not isinstance(other, Value) else (self, other)

        def vjp(g: np.ndarray, a=self, b=other) -> None:
            a._accum(g)
            if isinstance(b, Value):
                b._accum(-g)

        return self._record(self.data - o, inputs, vjp)

    def __rsub__(self, other: Any) -> Value:
        o = self._as_arr(other)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(-g)

        return self._record(o - self.data, (self,), vjp)

    def __mul__(self, other: Any) -> Value:
        o = self._as_arr(other)
        inputs = (self,) if not isinstance(other, Value) else (self, other)

        def vjp(g: np.ndarray, a=self, b=other) -> None:
            bo = b.data if isinstance(b, Value) else np.asarray(o)
            a._accum(g * bo)
            if isinstance(b, Value):
                b._accum(g * a.data)

        return self._record(self.data * o, inputs, vjp)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Value:
        o = self._as_arr(other)
        inputs = (self,) if not isinstance(other, Value) else (self, other)

        def vjp(g: np.ndarray, a=self, b=other) -> None:
            bo = b.data if isinstance(b, Value) else np.asarray(o)
            a._accum(g / bo)
            if isinstance(b, Value):
                b._accum(-g * a.data / (bo * bo))

        return self._record(self.data / o, inputs, vjp)

    def __rtruediv__(self, other: Any) -> Value:
        o = self._as_arr(other)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(-g * o / (a.data * a.data))

        return self._record(o / self.data, (self,), vjp)

    def __pow__(self, power: float) -> Value:
        p = float(power)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * p * np.power(a.data, p - 1.0))

        return self._record(np.power(self.data, p), (self,), vjp)

    def reciprocal(self) -> Value:
        return self.__rtruediv__(1.0)

    # ------------------------------------------------------- transcendental
    def exp(self) -> Value:
        out = np.exp(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * out)

        return self._record(out, (self,), vjp)

    def log(self) -> Value:
        out = np.log(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g / a.data)

        return self._record(out, (self,), vjp)

    def sin(self) -> Value:
        out = np.sin(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * np.cos(a.data))

        return self._record(out, (self,), vjp)

    def cos(self) -> Value:
        out = np.cos(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(-g * np.sin(a.data))

        return self._record(out, (self,), vjp)

    def tanh(self) -> Value:
        out = np.tanh(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * (1.0 - out * out))

        return self._record(out, (self,), vjp)

    def sqrt(self) -> Value:
        out = np.sqrt(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * 0.5 / out)

        return self._record(out, (self,), vjp)

    def abs(self) -> Value:
        out = np.abs(self.data)
        sign = np.sign(self.data)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g * sign)

        return self._record(out, (self,), vjp)

    # -------------------------------------------------------------- matrices
    def matmul(self, other: Value) -> Value:
        def vjp(g: np.ndarray, a=self, b=other) -> None:
            a._accum(g @ b.data.T)
            b._accum(a.data.T @ g)

        return self._record(self.data @ other.data, (self, other), vjp)

    def __matmul__(self, other: Value) -> Value:
        return self.matmul(other)

    @property
    def T(self) -> Value:
        return self._record(self.data.T, (self,), lambda g, a=self: a._accum(g.T))

    # ------------------------------------------------------------ reductions
    def sum(self) -> Value:
        out = np.asarray(self.data.sum(), dtype=np.float64)
        out = out.reshape(()) if self._scalar else out.reshape(1)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(np.broadcast_to(np.asarray(g), a.data.shape).copy())

        return self._record(out, (self,), vjp)

    def mean(self) -> Value:
        out = np.asarray(self.data.mean(), dtype=np.float64)
        out = out.reshape(()) if self._scalar else out.reshape(1)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(np.broadcast_to(np.asarray(g) / a.data.size, a.data.shape).copy())

        return self._record(out, (self,), vjp)

    def max(self) -> Value:
        idx = int(np.argmax(self.data))
        out = np.asarray(self.data.reshape(-1)[idx], dtype=np.float64)
        out = out.reshape(()) if self._scalar else out.reshape(1)

        def vjp(g: np.ndarray, a=self) -> None:
            mask = np.zeros_like(a.data)
            mask.reshape(-1)[idx] = 1.0
            a._accum(g * mask)

        return self._record(out, (self,), vjp)

    # ---------------------------------------------------------------- shapes
    def reshape(self, *shape: int) -> Value:
        new = self.data.reshape(*shape)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g.reshape(a.data.shape))

        return self._record(new, (self,), vjp)

    def squeeze(self) -> Value:
        new = self.data.squeeze()

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g.reshape(a.data.shape))

        return self._record(new, (self,), vjp)

    def unsqueeze(self, axis: int = 0) -> Value:
        new = np.expand_dims(self.data, axis)

        def vjp(g: np.ndarray, a=self) -> None:
            a._accum(g.reshape(a.data.shape))

        return self._record(new, (self,), vjp)

    # -------------------------------------------------------------- indexing
    def __getitem__(self, key: Any) -> Value:
        out = self.data[key]
        # An integer (or integer-tuple) key drops a dimension, exactly as numpy does;
        # the result is a scalar node so that ``y[0] * y[1]`` behaves as expected.
        is_int = isinstance(key, (int, np.integer)) or (
            isinstance(key, tuple) and all(isinstance(k, (int, np.integer)) for k in key)
        )

        def vjp(g: np.ndarray, a=self, k=key) -> None:
            mask = np.zeros_like(a.data)
            np.add.at(mask, k, np.asarray(g).reshape(np.shape(mask[k])))
            a._accum(mask)

        node = self._record(out, (self,), vjp)
        if is_int:
            node._scalar = out.ndim == 0
        return node

    # ----------------------------------------------------------- combination
    @staticmethod
    def concat(parts: Sequence[Value], axis: int = 0) -> Value:
        """Concatenate taped values along ``axis``."""
        datas = [p.data for p in parts]
        out = np.concatenate(datas, axis=axis)

        def vjp(g: np.ndarray, ps=tuple(parts), ax=axis) -> None:
            offset = 0
            for p in ps:
                width = p.data.shape[ax]
                sl = [slice(None)] * g.ndim
                sl[ax] = slice(offset, offset + width)
                p._accum(g[tuple(sl)])
                offset += width

        tape = parts[0]._tape
        node = Value(
            out, requires_grad=any(p.requires_grad for p in parts), _prev=tuple(parts), _tape=tape
        )
        tape.append(tuple(parts), node, vjp)
        return node

    @staticmethod
    def stack(parts: Sequence[Value], axis: int = 0) -> Value:
        """Stack taped values along a new ``axis``.

        Scalar parts (whose storage is shape ``(1,)``) are stacked as scalars, so
        ``stack([q, v])`` of two scalar nodes yields a flat 2-vector rather than a
        ``(2, 1)`` column.
        """
        scalar_parts = all(p._scalar or (p.data.ndim <= 1 and p.data.size == 1) for p in parts)
        # np.ravel of a shape-(1,) array is still shape (1,), so stacking would give a
        # (2, 1) column. A scalar part must lose its trailing axis entirely.
        datas = [p.data.reshape(()) if scalar_parts else p.data for p in parts]
        out = np.stack(datas, axis=axis)

        def vjp(g: np.ndarray, ps=tuple(parts), ax=axis) -> None:
            for i, p in enumerate(ps):
                sl = [slice(None)] * g.ndim
                sl[ax] = i
                p._accum(g[tuple(sl)])

        tape = parts[0]._tape
        node = Value(
            out, requires_grad=any(p.requires_grad for p in parts), _prev=tuple(parts), _tape=tape
        )
        node._scalar = False
        tape.append(tuple(parts), node, vjp)
        return node

    # ------------------------------------------------------------------ norms
    def norm(self) -> Value:
        """Frobenius norm of a matrix, 2-norm of a vector."""
        # np.linalg.norm returns a numpy scalar (not an ndarray) for 1-D input,
        # and a Python float in some paths; np.asarray(...).reshape(1) handles all.
        out = np.asarray(np.linalg.norm(self.data), dtype=np.float64)
        out = out.reshape(()) if self._scalar else out.reshape(1)

        def vjp(g: np.ndarray, a=self) -> None:
            nrm = float(np.linalg.norm(a.data))
            if nrm < 1e-300:
                a._accum(np.zeros_like(a.data))
                return
            if a.data.ndim == 1:
                a._accum(g * a.data / nrm)
            else:
                a._accum(g * a.data / nrm)

        return self._record(out, (self,), vjp)

    def __repr__(self) -> str:
        return f"Value(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------ numpy ufunc interoperability
    # np.log(v) / np.exp(v) / np.sin(v) ... must dispatch back to the taped methods,
    # otherwise numpy tries to coerce a Value into an ndarray and raises.
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or ufunc.__name__ not in _UFUNC_DISPATCH:
            return NotImplemented
        if len(inputs) != 1 or kwargs.get("out") is not None:
            return NotImplemented
        return _UFUNC_DISPATCH[ufunc.__name__](self)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        # Explicit opt-in for callers that genuinely want the raw array.
        return np.asarray(self.data, dtype=dtype)


_UFUNC_DISPATCH.update(
    {
        "exp": Value.exp,
        "log": Value.log,
        "sin": Value.sin,
        "cos": Value.cos,
        "tanh": Value.tanh,
        "sqrt": Value.sqrt,
        "absolute": Value.abs,
    }
)


def tape_stats() -> dict[str, int]:
    """Node and leaf counts of the global tape (memory proxy for a reverse sweep)."""
    return _GLOBAL_TAPE.stats()


def leaf_gradients(tape: Tape | None = None) -> list[np.ndarray]:
    """Return the gradients of every leaf on ``tape`` (default: the global tape).

    Call after :meth:`Value.backward`; the node list has been released but the leaves
    and their gradients are still available.
    """
    t = tape if tape is not None else _GLOBAL_TAPE
    return [np.zeros_like(lf.data) if lf.grad is None else lf.grad.copy() for lf in t.leaves]


def reset_tape(tape: Tape | None = None) -> None:
    """Drop all nodes and leaves from a tape (used between training epochs)."""
    t = tape if tape is not None else _GLOBAL_TAPE
    t.clear()
    t.leaves.clear()


def central_difference_check(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    eps: float = 1e-6,
    tol: float = _TOL,
) -> dict[str, Any]:
    """Compare an analytic gradient against a central finite difference.

    ``fn`` maps a parameter array to a scalar loss and must itself return the analytic
    gradient, as ``(loss, grad)``. The relative error is the max over coordinates of
    ``|g_analytic - g_fd| / max(1, |g_fd|)``.

    This is the golden standard required of :mod:`autodiff`: relative error below
    ``tol`` (default ``1e-6``).
    """
    x = np.asarray(x, dtype=np.float64).copy()
    loss, grad = fn(x)
    grad = np.asarray(grad, dtype=np.float64)
    fd = np.zeros_like(x)
    for i in range(x.size):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        fd[i] = (float(fn(xp)[0]) - float(fn(xm)[0])) / (2.0 * step)
    denom = np.maximum(1.0, np.abs(fd))
    rel = np.abs(grad - fd) / denom
    worst = float(np.max(rel)) if rel.size else 0.0
    return {
        "loss": float(loss),
        "analytic": grad,
        "finite_difference": fd,
        "max_rel_error": worst,
        "passed": bool(worst < tol),
        "tol": float(tol),
    }

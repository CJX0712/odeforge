"""Autodiff contracts: the tape, scalar semantics, the adjoint, and the training task.

The golden standard throughout is a central-difference check with relative error
below ``1e-6``.
"""

from __future__ import annotations

import numpy as np
import pytest

from autodiff.adjoint import adjoint_gradient, integrate_with_adjoint, rk4_step
from autodiff.tape import (
    Tape,
    Value,
    central_difference_check,
    leaf_gradients,
    reset_tape,
    tape_stats,
)
from autodiff.vector import as_value, batch_map, mean_square_error, stack_batch, vdot
from core.seed import set_all
from training.task import TaskConfig, fit, generate_observations, loss_and_grad, true_params


def _check(build, x0, tol=1e-6):
    """Backward a tape built by ``build`` and compare against central differences."""

    def fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        reset_tape()
        leaf = Value(np.asarray(x, dtype=np.float64).copy())
        out = build(leaf)
        out.backward()
        grads = leaf_gradients()
        return out.item(), grads[0].copy()

    return central_difference_check(fn, np.asarray(x0, dtype=np.float64), tol=tol)


# ------------------------------------------------------------- scalar semantics
def test_integer_index_yields_a_scalar_node() -> None:
    """``y[0]`` must be a scalar, exactly as numpy does.

    Promoting every 0-d node to shape ``(1,)`` made ``y[0]`` return shape ``(1,)``, which
    broke arithmetic mixing indexed states and silently changed broadcasting.
    """
    y = Value(np.array([1.0, 2.0]))
    assert y[0].shape == ()
    assert y[1].shape == ()
    assert y[0].item() == pytest.approx(1.0)


def test_stack_of_scalars_gives_a_flat_vector() -> None:
    y = Value(np.array([1.0, 2.0]))
    assert Value.stack([y[0], y[1]], axis=0).shape == (2,)


def test_stack_of_vectors_keeps_its_shape() -> None:
    a = Value(np.array([1.0, 2.0]))
    b = Value(np.array([3.0, 4.0]))
    assert Value.stack([a, b], axis=0).shape == (2, 2)


def test_summation_of_a_scalar_node_stays_scalar() -> None:
    s = Value(3.0)
    assert s.sum().shape == ()


# --------------------------------------------------------------- primitives
#: One builder per primitive family under test. A dispatch table rather than a chain of
#: ``if`` returns keeps each case on its own line and avoids an eleven-branch function.
_BUILDERS = {
    "poly": lambda a: (a * a * a + 2.0 * a - 1.0).sum(),
    "trig": lambda a: (a.sin() * a.cos()).sum(),
    "exp": lambda a: np.log(a.exp().sum()),
    "tanh": lambda a: (a.tanh() * a).sum(),
    "sqrt": lambda a: (a * a + 1.0).sqrt().sum(),
    "div": lambda a: (a / (a * a + 2.0)).sum(),
    "norm": lambda a: a.norm(),
    "reshape": lambda a: (a.reshape(2, 2) * a.reshape(2, 2)).sum(),
    "mean": lambda a: a.mean(),
    "max": lambda a: a.max(),
    "pow": lambda a: (a**3).sum(),
}


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_primitive_gradients_match_finite_differences(rng, name) -> None:
    res = _check(_BUILDERS[name], rng.standard_normal(4))
    assert res["passed"], f"{name}: {res['max_rel_error']:.3e}"


def test_matmul_gradient_matches_finite_differences(rng) -> None:
    mat = rng.standard_normal((3, 3))

    def build(a: Value) -> Value:
        return Value(mat).matmul(a.reshape(3, 1)).norm()

    res = _check(build, rng.standard_normal(3))
    assert res["passed"]


def test_indexing_gradient_matches_finite_differences(rng) -> None:
    def build(a: Value) -> Value:
        return (a[0:2] * a[2:4]).sum()

    assert _check(build, rng.standard_normal(4))["passed"]


def test_stack_and_concat_gradients(rng) -> None:
    def build_stack(a: Value) -> Value:
        return Value.stack([a[0:2], a[2:4]]).sum()

    def build_concat(a: Value) -> Value:
        return Value.concat([a[0:2], a[2:4]]).sum()

    assert _check(build_stack, rng.standard_normal(4))["passed"]
    assert _check(build_concat, rng.standard_normal(4))["passed"]


def test_numpy_ufuncs_dispatch_back_to_the_tape(rng) -> None:
    """``np.log(v)`` must tape, not try to coerce a Value into an array."""
    res = _check(lambda a: np.log(a.exp().sum()), rng.standard_normal(3))
    assert res["passed"]


# ----------------------------------------------------------------- machinery
def test_backward_assigns_the_seed_adjoint() -> None:
    """Regression: backward() once cleared the adjoint without assigning the seed,
    so every gradient came out exactly zero."""
    reset_tape()
    leaf = Value(np.array([3.0]))
    out = (leaf * leaf).sum()
    out.backward()
    assert out.grad is not None
    assert np.allclose(leaf.grad, [6.0])


def test_leaf_gradients_survive_tape_release() -> None:
    """``backward`` clears the node list to free memory; leaf gradients must persist."""
    reset_tape()
    leaf = Value(np.array([2.0, 3.0]))
    out = (leaf * leaf).sum()
    out.backward()
    assert tape_stats()["n_nodes"] == 0
    assert np.allclose(leaf_gradients()[0], [4.0, 6.0])


def test_explicit_tape_is_supported() -> None:
    tape = Tape()
    leaf = Value(np.array([2.0]), _tape=tape)
    out = (leaf * 3.0).sum()
    out.backward()
    assert np.allclose(leaf.grad, [3.0])


def test_reset_tape_clears_leaves() -> None:
    Value(np.array([1.0]))
    reset_tape()
    assert len(leaf_gradients()) == 0


def test_requires_grad_false_blocks_accumulation() -> None:
    reset_tape()
    leaf = Value(np.array([2.0]), requires_grad=False)
    out = (leaf * leaf).sum()
    out.backward()
    assert leaf.grad is None


# ------------------------------------------------------------------- vector
def test_vector_helpers(rng) -> None:
    reset_tape()
    x = Value(rng.standard_normal(4))
    assert as_value(x) is x
    assert stack_batch([Value(np.array([1.0])), Value(np.array([2.0]))]).shape == (2,)
    assert vdot(x, np.ones(4)).shape == (1,)
    assert mean_square_error(x, np.zeros(4)).shape == (1,)
    assert batch_map(lambda v: v * 2.0, x).shape == (4,)


# ------------------------------------------------------------------ adjoint
def test_adjoint_gradient_matches_finite_differences() -> None:
    """Regression: the pull-back is ``M^T lam``, not ``lam - M^T lam``."""

    def make_rhs(theta: np.ndarray):
        k, c = theta

        def f(t, y):
            return np.array([y[1], -k * y[0] - c * y[1]])

        return f

    def loss(yf, th):
        return float(0.5 * (yf[0] - 1.0) ** 2 + 0.5 * yf[1] ** 2)

    th0 = np.array([1.0, 0.1])

    def fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        return adjoint_gradient(
            make_rhs, np.asarray(x, float), [1.0, 0.0], (0.0, 4.0), loss, n_steps=200
        )

    res = central_difference_check(fn, th0, tol=1e-6)
    assert res["passed"], f"{res['max_rel_error']:.3e}"


def test_adjoint_step_map_jacobian_is_exact() -> None:
    """``M`` must reproduce central differences of the whole RK4 step."""
    f = lambda t, y: np.array([y[1], -y[0]])  # noqa: E731
    y0 = np.array([0.7, -0.3])
    t0, h = 1.3, 0.1
    jac = np.empty((2, 2))
    for j in range(2):
        yp, ym = y0.copy(), y0.copy()
        yp[j] += 1e-7
        ym[j] -= 1e-7
        jac[:, j] = (rk4_step(f, t0, yp, h) - rk4_step(f, t0, ym, h)) / 2e-7
    from autodiff.adjoint import _step_map_jacobian  # noqa: PLC0415

    analytic, _ = _step_map_jacobian(f, t0, y0, h)
    assert np.max(np.abs(analytic - jac)) < 1e-9


def test_adjoint_reports_constant_memory() -> None:
    def make_rhs(theta: np.ndarray):
        def f(t, y):
            return np.array([y[1], -theta[0] * y[0]])

        return f

    res = integrate_with_adjoint(
        make_rhs,
        np.array([1.0]),
        [1.0, 0.0],
        (0.0, 1.0),
        lambda yf, th: float(yf[0]),
        n_steps=20,
    )
    assert res["memory_slots"] == 2


# ----------------------------------------------------------------- training
def test_training_gradient_matches_finite_differences() -> None:
    """Regression: the loss once stored plain floats, severing the tape."""
    cfg = TaskConfig(n_steps=60, n_obs=12, n_iters=0)
    _, q_obs = generate_observations(cfg)

    def fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        return loss_and_grad(np.asarray(x, float), cfg, np.zeros_like(q_obs), q_obs)

    res = central_difference_check(fn, np.array([0.3, 0.9, 0.35, 1.2]), tol=1e-6)
    assert res["passed"], f"{res['max_rel_error']:.3e}"


def test_training_reduces_the_loss() -> None:
    cfg = TaskConfig(n_iters=25, n_obs=20)
    result = fit(cfg)
    assert result.final_loss < result.loss_curve[0]


def test_training_is_deterministic() -> None:
    set_all(20260101)
    a = fit(TaskConfig(n_iters=8)).as_dict()
    b = fit(TaskConfig(n_iters=8)).as_dict()
    assert a == b


def test_observations_are_reproducible() -> None:
    set_all(20260101)
    t1, q1 = generate_observations(TaskConfig())
    set_all(20260101)
    t2, q2 = generate_observations(TaskConfig())
    assert np.array_equal(t1, t2)
    assert np.array_equal(q1, q2)


def test_true_params_shape() -> None:
    assert true_params(TaskConfig()).shape == (4,)

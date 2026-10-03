"""A real trainable task: fit ODE parameters to noisy observations.

Task
----
Recover the parameters of a damped, driven oscillator

.. math:: \\ddot q + c \\dot q + k q = \\alpha \\sin(\\omega t)

from noisy position samples. The model is integrated with ODEForge's own RK4 at a fixed
step, and the parameters are fitted by gradient descent using the **tape** gradients from
:mod:`autodiff.tape` -- so the gradient flows through every RK4 stage without anyone
writing a derivative by hand.

This is a genuine end-to-end test of "differentiable integration": the loss is a function
of a trajectory, and the trajectory's dependence on the parameters is what the optimiser
needs.

Determinism
-----------
The observation noise is drawn once from :func:`core.seed.rng` with a recorded offset, so
:func:`fit` returns identical numbers for identical inputs.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autodiff.tape import Value, reset_tape
from core.seed import rng

__all__ = [
    "FitResult",
    "TaskConfig",
    "fit",
    "generate_observations",
    "loss_and_grad",
    "make_rhs",
    "true_params",
]


@dataclass(frozen=True)
class TaskConfig:
    """Hyperparameters of the parameter-identification task."""

    n_obs: int = 60
    t_end: float = 12.0
    n_steps: int = 240
    noise_std: float = 5e-3
    # lr chosen by a measured sweep (0.30 diverges, 0.15 gives loss 0.267 -> 0.00106
    # in 60 iterations with max parameter error 0.06; 0.04 and below converge too slowly).
    lr: float = 0.15
    n_iters: int = 60
    seed_offset: int = 23
    true_params: tuple[float, float, float, float] = (0.25, 1.0, 0.4, 1.3)
    init_params: tuple[float, float, float, float] = (0.10, 0.70, 0.20, 1.00)


def true_params(cfg: TaskConfig) -> np.ndarray:
    """The ground-truth parameter vector ``(c, k, alpha, omega)``."""
    return np.asarray(cfg.true_params, dtype=np.float64)


def make_rhs(theta: Value) -> Any:
    """Build a taped right-hand side from a parameter :class:`Value`.

    Returns a 2-vector ``Value`` of shape ``(2,)``: ``[q', v']``.
    """
    c, k, alpha, omega = theta[0], theta[1], theta[2], theta[3]

    def rhs(t: float, y: Value) -> Value:
        # Index the 2-state into two genuine scalar nodes. Indexing a length-2 Value
        # with [0]/[1] yields a scalar node, so the arithmetic below stays scalar and
        # no accidental broadcasting to shape (2,) can occur.
        q = y[0]
        v = y[1]
        drive = alpha * np.sin(omega * t)
        acc = -c * v - k * q + drive
        # stack expects equal shapes; both entries are shape (1,) scalar nodes.
        return Value.stack([v, acc], axis=0)

    return rhs


def _integrate_numeric(theta: np.ndarray, cfg: TaskConfig) -> np.ndarray:
    """Plain-numpy RK4 integration, used to synthesise the observations."""
    c, k, alpha, omega = (float(x) for x in theta)

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -c * y[1] - k * y[0] + alpha * np.sin(omega * t)])

    t = np.linspace(0.0, cfg.t_end, cfg.n_steps + 1)
    h = cfg.t_end / cfg.n_steps
    y = np.array([1.0, 0.0])
    out = np.empty((cfg.n_steps + 1, 2))
    out[0] = y
    for i in range(cfg.n_steps):
        ti = t[i]
        k1 = f(ti, y)
        k2 = f(ti + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(ti + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(ti + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    return out


def generate_observations(cfg: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_obs, q_obs)``: noisy position samples from the true system.

    The noise is drawn from ``numpy.random.default_rng`` seeded through
    :func:`core.seed.rng` with :attr:`TaskConfig.seed_offset`, so the dataset is
    reproducible.
    """
    traj = _integrate_numeric(true_params(cfg), cfg)
    idx = np.unique(np.linspace(0, cfg.n_steps, cfg.n_obs).astype(int))
    t_obs = idx * (cfg.t_end / cfg.n_steps)
    clean = traj[idx, 0]
    gen = rng(offset=cfg.seed_offset)
    noise = gen.standard_normal(clean.size) * cfg.noise_std
    return t_obs, clean + noise


def loss_and_grad(
    theta: np.ndarray, cfg: TaskConfig, t_obs: np.ndarray, q_obs: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean-squared error of a taped RK4 integration, plus its parameter gradient."""
    reset_tape()
    leaf = Value(np.asarray(theta, dtype=np.float64).copy())
    rhs = make_rhs(leaf)

    h = cfg.t_end / cfg.n_steps
    ts = np.linspace(0.0, cfg.t_end, cfg.n_steps + 1)
    state = Value(np.array([1.0, 0.0]))
    preds: list[Value] = []
    obs_idx = set(np.unique(np.linspace(0, cfg.n_steps, cfg.n_obs).astype(int)).tolist())

    for i in range(cfg.n_steps):
        ti = float(ts[i])
        k1 = rhs(ti, state)
        k2 = rhs(ti + 0.5 * h, state + (0.5 * h) * k1)
        k3 = rhs(ti + 0.5 * h, state + (0.5 * h) * k2)
        k4 = rhs(ti + h, state + h * k3)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if i + 1 in obs_idx:
            # Keep the TAPE: storing a plain float here would detach the loss from the
            # trajectory and every parameter gradient would come out exactly zero.
            preds.append(state[0])

    resid = Value.stack(preds, axis=0)
    diff = resid - np.asarray(q_obs[: len(preds)])
    loss_node = (diff * diff).mean()
    loss_node.backward()
    # Read the gradient off the *parameter* leaf directly. leaf_gradients()[0] is not
    # reliable here: the state trajectory creates further parentless Value nodes on the
    # same tape, so positional indexing picked the wrong one and returned zeros.
    grad = np.zeros_like(theta) if leaf.grad is None else leaf.grad.copy()
    return float(np.asarray(loss_node.data).reshape(-1)[0]), grad


@dataclass
class FitResult:
    """Outcome of a parameter fit."""

    params: np.ndarray
    loss_curve: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    n_iters: int = 0

    @property
    def final_loss(self) -> float:
        return float(self.loss_curve[-1]) if self.loss_curve else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": [float(x) for x in self.params],
            "final_loss": self.final_loss,
            "n_iters": int(self.n_iters),
            "loss_curve": [float(x) for x in self.loss_curve],
        }


def fit(cfg: TaskConfig | None = None, **kwargs: Any) -> FitResult:
    """Fit the oscillator parameters by gradient descent on the taped loss.

    Plain gradient descent (no momentum) keeps the run deterministic and the loss curve
    monotone enough to be a meaningful sanity signal.
    """
    cfg = cfg or TaskConfig(**kwargs) if kwargs else cfg or TaskConfig()
    t_obs, q_obs = generate_observations(cfg)
    theta = np.asarray(cfg.init_params, dtype=np.float64).copy()
    curve: list[float] = []
    gnorms: list[float] = []
    for _ in range(cfg.n_iters):
        loss, grad = loss_and_grad(theta, cfg, t_obs, q_obs)
        curve.append(loss)
        gnorms.append(float(np.linalg.norm(grad)))
        theta = theta - cfg.lr * grad
    return FitResult(params=theta, loss_curve=curve, grad_norms=gnorms, n_iters=cfg.n_iters)

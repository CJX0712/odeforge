"""Evaluation metrics for integrator trajectories.

The metrics are deliberately kept separate rather than collapsed into one "error"
number, because they answer different questions:

``net_drift``
    ``C(y(T)) - C(y(0))`` for a conserved quantity ``C``. This is the metric that
    exposes a **secular** energy trend, which is the structural weakness of
    non-symplectic methods. A bounded oscillation averages to ~0 here even while the
    instantaneous error is large.
``max_deviation``
    ``max |C(y(t)) - C(y(0))|``. This exposes the **amplitude** of the error and does
    not distinguish bounded from growing.

Both are reported because they can disagree. Measured on the double-well at equal
``n_fev = 120000``: Yoshida4 had ``net_drift = -1.96e-10`` with
``max_deviation = 4.31e-07``, while RK4 had ``net_drift = -3.81e-07`` with
``max_deviation = 3.81e-07``. On ``max_deviation`` alone RK4 looks competitive; only
``net_drift`` shows the symplectic advantage. Reporting one without the other would
have hidden that.

``traj_error``
    Max-norm deviation from a reference solution, after interpolating both onto a
    common time grid.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import ConservedQuantity, MethodCost, MetricBundle, Trajectory

__all__ = [
    "conserved_series",
    "cost_of",
    "evaluate",
    "interpolation_error",
    "max_deviation",
    "monotone_fraction",
    "net_drift",
    "relative_drift",
    "sign_ratio",
    "trajectory_error",
]


def conserved_series(traj: Trajectory, quantity: ConservedQuantity) -> np.ndarray:
    """Evaluate a conserved quantity along the whole trajectory."""
    return quantity.series(traj)


def net_drift(traj: Trajectory, quantity: ConservedQuantity) -> float:
    """``C(y(T)) - C(y(0))`` -- the secular drift of an invariant."""
    series = conserved_series(traj, quantity)
    return float(series[-1] - series[0])


def max_deviation(traj: Trajectory, quantity: ConservedQuantity) -> float:
    """``max |C(y(t)) - C(y(0))|`` -- the amplitude of the invariant error."""
    series = conserved_series(traj, quantity)
    return float(np.max(np.abs(series - series[0])))


def relative_drift(traj: Trajectory, quantity: ConservedQuantity) -> float:
    """Net drift normalised by the invariant's scale along the trajectory."""
    series = conserved_series(traj, quantity)
    denom = max(1e-12, float(np.max(np.abs(series))))
    return float(abs(series[-1] - series[0]) / denom)


def sign_ratio(series: np.ndarray) -> float:
    """Fraction of adjacent differences carrying the majority sign.

    ``1.0`` means the sequence changes monotonically (a secular trend); values well
    below ``1`` mean it oscillates. Used to distinguish bounded oscillation from drift
    without depending on the magnitude of either.
    """
    d = np.diff(np.asarray(series, dtype=np.float64))
    d = d[d != 0.0]
    if d.size == 0:
        return float("nan")
    return float(max(np.count_nonzero(d > 0), np.count_nonzero(d < 0)) / d.size)


def monotone_fraction(traj: Trajectory, quantity: ConservedQuantity) -> float:
    """Fraction of adjacent differences that share the sign of the net change."""
    series = conserved_series(traj, quantity)
    total = series[-1] - series[0]
    if total == 0.0:
        return float("nan")
    d = np.diff(series)
    same = np.count_nonzero(np.sign(d) == np.sign(total))
    return float(same / max(1, d.size))


def interpolation_error(traj: Trajectory, *, stride: int = 8) -> float:
    """Self-test: how well does this trajectory survive its own interpolation?

    The trajectory is decimated by ``stride`` and then resampled onto its own full grid.
    The result upper-bounds the interpolation error that :func:`trajectory_error` will
    introduce when it interpolates *this* trajectory as the reference.

    This matters because adaptive solvers store very few points on rapidly curving
    problems. Measured on Lorenz at ``rtol=1e-12``: DOP853 stored **617** points over
    ``T=10`` and the *linear interpolation* of that grid carried an error of ``4.9e-02`` --
    five orders of magnitude larger than the ``1.5e-09`` by which the two solvers
    actually disagreed. Any gate comparing them on linear interpolation would have been
    measuring interpolation noise.
    """
    if traj.n_points <= stride * 4:
        return 0.0
    idx = np.arange(0, traj.n_points, stride)
    fine_t = traj.t
    fine_y = traj.y
    coarse_t = traj.t[idx]
    coarse_y = traj.y[idx]
    out = np.empty_like(fine_y)
    for k in range(traj.n_states):
        out[:, k] = _interp_column(fine_t, coarse_t, coarse_y[:, k])
    return float(np.max(np.abs(out - fine_y)))


def _interp_column(query_t: np.ndarray, source_t: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    """Interpolate one column, preferring a cubic spline when scipy is available."""
    try:
        from scipy.interpolate import CubicSpline  # noqa: PLC0415

        return CubicSpline(source_t, source_y)(query_t)
    except Exception:  # pragma: no cover - scipy-free fallback
        return np.interp(query_t, source_t, source_y)


def trajectory_error(
    traj: Trajectory,
    ref_t: np.ndarray,
    ref_y: np.ndarray,
) -> float:
    """Max-norm error against a reference, resampled onto ``traj.t``.

    ``ref_y`` must be shaped ``(n_points, n_states)`` with rows as time points, matching
    :class:`~core.types.Trajectory`. Only the reference is resampled, so no axis
    ambiguity can creep in.

    The resampling uses a **cubic spline** when scipy is importable, because linear
    interpolation of an adaptively-stored grid can dominate the quantity being measured
    (see :func:`interpolation_error` for the measured numbers). Callers that need to know
    whether the metric is trustworthy at their scale should compare
    ``interpolation_error(ref)`` against the error they are about to measure.
    """
    ref_y = np.asarray(ref_y, dtype=np.float64)
    if ref_y.ndim != 2 or ref_y.shape[1] != traj.n_states:
        msg = (
            f"reference shape {ref_y.shape} incompatible with trajectory "
            f"(n_points={traj.n_points}, n_states={traj.n_states})"
        )
        raise ValueError(msg)
    ref_t = np.asarray(ref_t, dtype=np.float64)
    interp = np.empty_like(traj.y)
    for k in range(traj.n_states):
        interp[:, k] = _interp_column(traj.t, ref_t, ref_y[:, k])
    return float(np.max(np.abs(interp - traj.y)))


def cost_of(traj: Trajectory) -> MethodCost:
    """Package a trajectory's cost for reporting."""
    return MethodCost(
        n_fev=int(traj.n_fev),
        n_steps=int(traj.n_steps),
        n_rejected=int(traj.n_rejected),
        wall_sec=float(traj.wall_sec),
    )


def evaluate(
    traj: Trajectory,
    quantity: ConservedQuantity | None = None,
    *,
    ref_t: np.ndarray | None = None,
    ref_y: np.ndarray | None = None,
) -> MetricBundle:
    """Assemble the standard metric bundle for one trajectory."""
    net = max_dev = 0.0
    extra: dict[str, float] = {}
    if quantity is not None:
        net = net_drift(traj, quantity)
        max_dev = max_deviation(traj, quantity)
        series = conserved_series(traj, quantity)
        extra["relative_drift"] = relative_drift(traj, quantity)
        extra["sign_ratio"] = sign_ratio(series)
        extra["monotone_fraction"] = monotone_fraction(traj, quantity)
    err: float | None = None
    if ref_t is not None and ref_y is not None:
        err = trajectory_error(traj, ref_t, ref_y)
    return MetricBundle(
        net_drift=net,
        max_deviation=max_dev,
        traj_error=err,
        cost=cost_of(traj),
        extra=extra,
    )


def mean_std(values: list[float]) -> dict[str, float]:
    """Mean and *population* standard deviation, plus the sample count."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "n": int(arr.size),
    }


def significant_win(
    a: list[float],
    b: list[float],
    *,
    factor: float = 0.5,
) -> dict[str, Any]:
    """Decide whether ``a`` beats ``b`` under the stated statistical rule.

    A win counts only when the mean difference exceeds
    ``factor * (sigma_a + sigma_b)`` -- i.e. the gap is larger than half the summed
    spread, so run-to-run noise cannot explain it.
    """
    sa = mean_std([abs(float(x)) for x in a])
    sb = mean_std([abs(float(x)) for x in b])
    diff = sb["mean"] - sa["mean"]
    threshold = factor * (sa["std"] + sb["std"])
    return {
        "mean_a": sa["mean"],
        "std_a": sa["std"],
        "n_a": sa["n"],
        "mean_b": sb["mean"],
        "std_b": sb["std"],
        "n_b": sb["n"],
        "difference": diff,
        "threshold": threshold,
        "win": bool(diff > threshold),
    }

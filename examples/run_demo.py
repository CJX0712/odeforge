"""End-to-end ODEForge demonstration.

Runs the flagship through routing, integration, the acceptance gates, the ablation study
and a gradient check, then writes ``results/benchmark.json``.

Running it twice must produce identical numbers (excluding ``wall_sec`` fields); that is
the determinism contract, asserted by ``tools/check_determinism.py`` and enforced in CI.

Budget: the demo completes in roughly 40 s on CPU. Robertson is excluded from the demo
path because its stiffness costs seconds per solve; it still runs in the full
``odeforge benchmark`` command and in the test-suite.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Running this file directly puts examples/ on sys.path rather than the project root,
# so the top-level packages (core, data, integrators, ...) would not be importable.
# This is a no-op when the package is installed or run through pytest with
# pythonpath = ["."].
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from core.config import Config  # noqa: E402
from core.seed import set_all  # noqa: E402
from pipeline import DEMO_SYSTEMS, ODEForgePipeline  # noqa: E402

__all__ = ["OUTPUT_PATH", "main", "run_demo"]

OUTPUT_PATH = Path("results/benchmark.json")


def _gradient_check() -> dict[str, Any]:
    """Central-difference check of the tape gradient on the training loss.

    This is the project's gradient golden standard: relative error below ``1e-6``.
    """
    from autodiff.tape import central_difference_check  # noqa: PLC0415
    from training.task import TaskConfig, generate_observations, loss_and_grad  # noqa: PLC0415

    cfg = TaskConfig(n_steps=60, n_obs=12, n_iters=0)
    _, q_obs = generate_observations(cfg)
    theta0 = np.array([0.3, 0.9, 0.35, 1.2])

    def fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        return loss_and_grad(np.asarray(x, dtype=np.float64), cfg, np.zeros_like(q_obs), q_obs)

    res = central_difference_check(fn, theta0, tol=1e-6)
    return {
        "max_rel_error": float(res["max_rel_error"]),
        "tol": float(res["tol"]),
        "passed": bool(res["passed"]),
    }


def _order_check() -> dict[str, Any]:
    """Measure the empirical convergence order of each integrator on the harmonic case."""
    from integrators import rk4, vverlet, yoshida4, yoshida6  # noqa: PLC0415

    grad_v = lambda q: q  # noqa: E731
    rhs = lambda t, y: np.array([y[1], -y[0]])  # noqa: E731
    exact = np.array([np.cos(1.0), -np.sin(1.0)])
    hs = np.array([0.2, 0.1, 0.05, 0.025])
    out: dict[str, Any] = {}
    for name, fn, theory in (
        ("vverlet", vverlet, 2),
        ("yoshida4", yoshida4, 4),
        ("yoshida6", yoshida6, 6),
    ):
        errs = []
        for h in hs:
            traj = fn(grad_v, [1.0, 0.0], (0.0, 1.0), n_steps=round(1.0 / h))
            errs.append(float(np.max(np.abs(traj.y[-1] - exact))))
        slope = float(np.polyfit(np.log(hs), np.log(np.asarray(errs)), 1)[0])
        out[name] = {"empirical_order": slope, "nominal_order": theory}
    errs = []
    for h in hs:
        traj = rk4(rhs, [1.0, 0.0], (0.0, 1.0), n_steps=round(1.0 / h))
        errs.append(float(np.max(np.abs(traj.y[-1] - exact))))
    out["rk4"] = {
        "empirical_order": float(np.polyfit(np.log(hs), np.log(np.asarray(errs)), 1)[0]),
        "nominal_order": 4,
    }
    return out


def _print_gate_rows(gate: dict[str, Any]) -> None:
    """Print one gate's rows, including the aligned budgets where they exist."""
    for row in gate.get("rows", []):
        if row.get("status") != "measured":
            print(f"          {row.get('system', '?'):<14s} skipped: {str(row.get('reason'))[:48]}")
            continue
        budget = row.get("budget")
        # Print the achieved n_fev of BOTH sides wherever they were aligned, so the
        # equal-budget claim is auditable from stdout alone.
        if budget:
            print(
                f"          {row['system']:<14s} B={budget['target_n_fev']} "
                f"flag_nfev={budget['flagship_n_fev']} "
                f"base_nfev={budget['baseline_n_fev']} "
                f"nfev_ratio={budget['n_fev_ratio']:.4f}"
            )
        if "flagship_net_drift" in row:
            key, bkey = "flagship_net_drift", "baseline_net_drift"
        else:
            key, bkey = "flagship_traj_error", "baseline_traj_error"
        print(
            f"          {row['system']:<14s} flagship={row[key]:.4e} "
            f"baseline={row[bkey]:.4e} ratio={row['ratio']:.4e} thr={row['threshold']:.4f}"
        )


def run_demo(*, quick: bool = True, output: Path | None = None) -> dict[str, Any]:
    """Execute the demonstration and return the report it wrote."""
    started = time.perf_counter()
    cfg = Config()
    set_all(cfg.seed)
    pipe = ODEForgePipeline(cfg)
    meta = pipe.meta()

    print("ODEForge demo")
    print("=" * 68)
    print(f"  author   : {meta['author']} <{meta['email']}>")
    print(f"  python   : {meta['python']}   numpy {meta['numpy']}")
    print(f"  seed     : {cfg.seed}")
    print()

    print("[1/5] convergence orders")
    orders = _order_check()
    for name, row in orders.items():
        print(
            f"      {name:<9s} empirical={row['empirical_order']:.4f} "
            f"(nominal {row['nominal_order']})"
        )
    print()

    print("[2/5] routing decisions")
    for name in DEMO_SYSTEMS:
        decision = pipe.route(name)
        votes = decision.votes
        flags = "".join("1" if votes[k] else "0" for k in ("hamiltonian", "conserved", "jacobian"))
        print(f"      {name:<16s} -> {decision.method:<9s} votes={flags}  {decision.reason[:52]}")
    print()

    print("[3/5] invariants")
    inv = pipe.invariants_report()
    for row in inv["rows"]:
        if "error" in row:
            print(f"      {row['system']:<16s} error: {row['error'][:48]}")
            continue
        for check in row["invariants"]:
            mark = "ok  " if check["passed"] else "FAIL"
            print(
                f"      {check['system']:<16s} {check['quantity']:<18s} "
                f"rel={check['relative_drift']:.3e} tol={check['tol']:.0e} {mark}"
            )
    print()

    print("[4/5] gradient check (tape vs central difference)")
    grad = _gradient_check()
    print(
        f"      max_rel_error={grad['max_rel_error']:.3e} tol={grad['tol']:.0e} "
        f"{'PASS' if grad['passed'] else 'FAIL'}"
    )
    print()

    print("[5/5] acceptance gates")
    report = pipe.benchmark_report(quick=quick)
    for name, gate in report["gates"].items():
        print(f"      {name}: {'PASS' if gate.get('passed') else 'FAIL'}")
        _print_gate_rows(gate)
    print()

    report["convergence_orders"] = orders
    report["gradient_check"] = grad
    report["demo_wall_sec"] = float(time.perf_counter() - started)

    path = pipe.write_report(report, output or OUTPUT_PATH)
    print("=" * 68)
    print(f"  report written to : {path}")
    print(f"  demo wall time    : {report['demo_wall_sec']:.2f}s")
    print(f"  gates             : {'ALL PASS' if report['passed']['all'] else 'SEE REPORT'}")
    print("=" * 68)
    return report


def main() -> int:
    """Entry point used by ``python examples/run_demo.py`` and ``odeforge demo``."""
    run_demo(quick=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

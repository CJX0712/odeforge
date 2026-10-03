"""Command-line interface for ODEForge.

Subcommands
-----------
=========== =====================================================================
``run``        route and integrate one system, print a summary
``benchmark``  run the gates + ablations, write ``benchmark.json``
``invariants`` verify declared invariants across the family
``demo``       end-to-end demonstration (same as ``examples/run_demo.py``)
``info``       environment, versions and the solver inventory
=========== =====================================================================

Every subcommand is deterministic: the seed comes from ``--seed`` or the
``ENV_ODEFORGE_SEED`` environment variable, and no subcommand draws from any other
entropy source.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from core.config import ENV_PREFIX, Config
from core.errors import ODEForgeError
from data.systems import SYSTEM_NAMES
from eval import available_scipy
from eval.benchmarks import FLAGSHIP, GATE_THRESHOLDS
from integrators import ALL_METHODS
from pipeline import DEMO_SYSTEMS, ODEForgePipeline

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="odeforge",
        description="ODEForge: continuous-time dynamics solvers and differentiable integration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"global seed; overrides {ENV_PREFIX}SEED",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="route and integrate one system")
    p_run.add_argument("system", choices=list(SYSTEM_NAMES))
    p_run.add_argument(
        "--method",
        default=None,
        choices=list(ALL_METHODS),
        help="bypass the router and force a method",
    )

    p_bench = sub.add_parser("benchmark", help="run the acceptance gates")
    p_bench.add_argument("--out", default="results/benchmark.json")
    p_bench.add_argument(
        "--quick", action="store_true", help="narrow G3 to the two cheap non-Hamiltonian systems"
    )

    p_inv = sub.add_parser("invariants", help="verify declared invariants")
    p_inv.add_argument("--systems", nargs="*", default=list(DEMO_SYSTEMS))

    sub.add_parser("demo", help="run the end-to-end demonstration")
    sub.add_parser("info", help="print environment and solver inventory")
    return parser


def _cmd_run(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    pipe = ODEForgePipeline(cfg)
    if args.method:
        decision = pipe.route(args.system)
        method = args.method
    else:
        decision = pipe.route(args.system)
        method = decision.method
    spec_dict = {
        "system": args.system,
        "method": method,
        "route": decision.as_dict(),
    }
    result = pipe.evaluate_system(args.system)
    spec_dict["result"] = result
    return spec_dict


def _cmd_benchmark(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    pipe = ODEForgePipeline(cfg)
    report = pipe.benchmark_report(quick=args.quick)
    path = pipe.write_report(report, args.out)
    report["written_to"] = str(path)
    return report


def _cmd_invariants(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    pipe = ODEForgePipeline(cfg)
    report = pipe.invariants_report(systems=tuple(args.systems))
    failures: list[str] = []
    for row in report["rows"]:
        for check in row.get("invariants", []):
            if not check.get("passed", True):
                failures.append(f"{check.get('system')}:{check.get('quantity')}")
    report["failures"] = failures
    report["passed"] = not failures
    return report


def _cmd_info(cfg: Config) -> dict[str, Any]:
    pipe = ODEForgePipeline(cfg)
    return {
        "meta": pipe.meta(),
        "config": cfg.as_dict(),
        "flagship": FLAGSHIP,
        "gate_thresholds": GATE_THRESHOLDS,
        "methods": list(ALL_METHODS),
        "systems": list(SYSTEM_NAMES),
        "scipy_available": available_scipy(),
        "env_prefix": ENV_PREFIX,
    }


def _fmt(payload: dict[str, Any]) -> str:
    """Human-readable rendering; ``--json`` bypasses this entirely."""
    if "gates" in payload:
        lines = ["ODEForge benchmark", "=" * 60]
        for name, gate in payload["gates"].items():
            mark = "PASS" if gate.get("passed") else "FAIL"
            lines.append(f"{name}: {mark}  ({gate.get('description', '')})")
            for row in gate.get("rows", []):
                if row.get("status") != "measured":
                    lines.append(f"    {row.get('system', '?'):<14s} skipped: {row.get('reason')}")
                    continue
                if "net_drift" in row:
                    lines.append(
                        f"    {row['system']:<14s} flagship={row['flagship_net_drift']:.4e} "
                        f"baseline={row['baseline_net_drift']:.4e} "
                        f"ratio={row['ratio']:.4e} thr={row['threshold']:.2f}"
                    )
                else:
                    lines.append(
                        f"    {row['system']:<14s} flagship={row['flagship_traj_error']:.4e} "
                        f"baseline={row['baseline_traj_error']:.4e} "
                        f"ratio={row['ratio']:.4f} thr={row['threshold']:.4f}"
                    )
        lines.append("=" * 60)
        lines.append(f"overall: {'PASS' if payload['passed'].get('all') else 'FAIL'}")
        return "\n".join(lines)
    if "rows" in payload and "invariants" in payload["rows"][0] if payload.get("rows") else False:
        lines = ["ODEForge invariants", "=" * 60]
        for row in payload["rows"]:
            if "error" in row:
                lines.append(f"  {row['system']:<14s} error: {row['error']}")
                continue
            for check in row["invariants"]:
                mark = "ok" if check["passed"] else "FAIL"
                lines.append(
                    f"  {check['system']:<14s} {check['quantity']:<18s} "
                    f"rel={check['relative_drift']:.3e} tol={check['tol']:.1e} {mark}"
                )
        return "\n".join(lines)
    if "meta" in payload and "methods" in payload:
        meta = payload["meta"]
        lines = [
            "ODEForge",
            f"  author      : {meta['author']} <{meta['email']}>",
            f"  python      : {meta['python']}",
            f"  platform    : {meta['platform']}",
            f"  numpy       : {meta['numpy']}",
            f"  scipy       : {'available' if meta['scipy_available'] else 'unavailable (pure numpy mode)'}",
            f"  flagship    : {payload['flagship']}",
            f"  methods     : {', '.join(payload['methods'])}",
            f"  systems     : {len(payload['systems'])} -> {', '.join(payload['systems'])}",
            f"  seed        : {payload['config']['seed']}",
            f"  budget n_fev: {payload['config']['budget_n_fev']}",
        ]
        return "\n".join(lines)
    if "system" in payload:
        result = payload.get("result", {})
        metrics = result.get("metrics", {})
        return (
            f"{payload['system']} via {payload['method']}\n"
            f"  route    : {payload['route']['reason']}\n"
            f"  n_fev    : {result.get('n_fev')}\n"
            f"  net drift: {metrics.get('net_drift', float('nan')):.6e}\n"
            f"  max dev  : {metrics.get('max_deviation', float('nan')):.6e}"
        )
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = Config.from_env(seed=args.seed) if args.seed is not None else Config.from_env()
        if args.command == "run":
            payload = _cmd_run(args, cfg)
        elif args.command == "benchmark":
            payload = _cmd_benchmark(args, cfg)
        elif args.command == "invariants":
            payload = _cmd_invariants(args, cfg)
        elif args.command == "info":
            payload = _cmd_info(cfg)
        else:  # demo
            from examples.run_demo import main as demo_main  # noqa: PLC0415

            return int(demo_main())
    except ODEForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(_fmt(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

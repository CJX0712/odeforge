"""The ODEForge pipeline: route, solve, evaluate, benchmark.

Layering is strictly one-directional and acyclic::

    cli -> pipeline -> {data, integrators, routing, autodiff, hpo, eval} -> core

``pipeline`` depends on every leaf package and on ``core``; nothing below depends on
``pipeline``. There is no import cycle, and no module reaches around another to shortcut
the DAG.

Naming discipline
-----------------
Report-producing methods are suffixed ``_report`` (``benchmark_report``,
``ablation_report``) rather than named ``benchmark``/``ablation``. A plain attribute
``self.benchmark = None`` shadows the ``benchmark()`` method and turns a typo into a
silent ``None``; the suffix removes the collision by construction.

Determinism
-----------
:meth:`ODEForgePipeline.run` calls :func:`core.seed.set_all` exactly once with the
configured seed. Two runs produce bit-identical numbers except for ``wall_sec``.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.config import Config
from core.seed import get_seed, seed_state, set_all
from core.types import Trajectory
from data.systems import SYSTEM_NAMES, build_system
from eval import (
    ablation as _ablation,
)
from eval import (
    available_scipy,
    check_invariants,
    evaluate,
    gate_g1,
    gate_g2,
    gate_g3,
)
from eval.benchmarks import FLAGSHIP, G2_SYSTEMS, GATE_THRESHOLDS, run_method
from eval.metrics import net_drift
from hpo.search import Candidate, GridSearch, budget_steps_for
from integrators import symplectic_solve
from routing.hamiltonian import HamiltonianDetector
from routing.sympfuse import SympFuse

__all__ = ["ODEForgePipeline"]

#: Systems used by the demo, chosen so the whole demo stays inside its time budget.
#: Robertson is excluded from the fast path because its stiffness costs ~7 s per solve at
#: a useful tolerance; it still runs inside :meth:`ODEForgePipeline.benchmark_report`.
DEMO_SYSTEMS: tuple[str, ...] = (
    "harmonic",
    "double_well",
    "pendulum",
    "kepler",
    "van_der_pol",
    "lorenz",
)


class ODEForgePipeline:
    """End-to-end driver: route a system to a solver, then evaluate and report."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.detector = HamiltonianDetector(
            n_probe=self.config.n_probe,
            threshold=self.config.hamiltonian_threshold,
        )
        self.fuse = SympFuse.from_config(self.config)
        self._runs: dict[str, Trajectory] = {}
        self._routes: dict[str, Any] = {}

    # ------------------------------------------------------------------ core
    def run(self, system: str) -> Trajectory:
        """Route ``system`` to a method and integrate it.

        The single entry point for "solve this problem". It seeds the global RNG first,
        so the result depends only on the configured seed.
        """
        set_all(self.config.seed)
        spec = build_system(system)
        decision = self.fuse.route(spec)
        self._routes[system] = decision
        traj = run_method(
            decision.method,
            spec,
            n_steps=(
                budget_steps_for(decision.method, self.config.budget_n_fev)
                if decision.method not in {"dp54"}
                else None
            ),
            config=self.config,
        )
        self._runs[system] = traj
        return traj

    def route(self, system: str) -> Any:
        """Routing decision for a system, without integrating."""
        return self.fuse.route(build_system(system))

    def evaluate_system(self, system: str) -> dict[str, Any]:
        """Integrate a system and collect its metrics and invariant checks."""
        spec = build_system(system)
        traj = self.run(system)
        quantity = spec.conserved[0] if spec.conserved else None
        metrics = evaluate(traj, quantity)
        return {
            "system": system,
            "method": traj.method,
            "route": self._routes[system].as_dict() if system in self._routes else None,
            "n_fev": int(traj.n_fev),
            "n_steps": int(traj.n_steps),
            "n_rejected": int(traj.n_rejected),
            "wall_sec": float(traj.wall_sec),
            "metrics": metrics.as_dict(),
            "invariants": [dict(c) for c in check_invariants(spec, traj)],
        }

    def search(self, system: str, *, budget: int | None = None) -> dict[str, Any]:
        """Grid-search the flagship's own configuration under a fixed budget."""
        spec = build_system(system)
        if spec.grad_v is None:
            msg = f"system {system!r} is not separable; grid search needs grad_v"
            raise ValueError(msg)
        b = budget if budget is not None else self.config.budget_n_fev
        quantity = spec.conserved[0] if spec.conserved else None

        def run(cand: Candidate) -> tuple[float, Trajectory | None]:
            traj = symplectic_solve(
                cand.method,
                spec.grad_v,
                spec.y0,
                spec.t_span,
                n_steps=cand.n_steps or budget_steps_for(cand.method, b),
            )
            score = abs(net_drift(traj, quantity)) if quantity is not None else float(traj.n_fev)
            return score, traj

        grid = GridSearch(
            [
                Candidate("yoshida6", budget_steps_for("yoshida6", b)),
                Candidate("yoshida6", budget_steps_for("yoshida6", b) // 2),
                Candidate("yoshida6", budget_steps_for("yoshida6", b) // 4),
                Candidate("yoshida4", budget_steps_for("yoshida4", b)),
                Candidate("vverlet", budget_steps_for("vverlet", b)),
            ]
        )
        result = grid.search(run)
        return {"system": system, "budget_n_fev": int(b), **result.as_dict()}

    # ------------------------------------------------------------- reporting
    def benchmark_report(self, *, quick: bool = False) -> dict[str, Any]:
        """Run the three gates plus ablations and return a JSON-ready report.

        ``quick=True`` narrows G3 to the two cheap non-Hamiltonian systems, which keeps
        the demo inside its time budget while leaving G1 and G2 -- the gates that decide
        the flagship -- at full strength.
        """
        cfg = self.config
        set_all(cfg.seed)
        started = time.perf_counter()

        # The demo must finish inside 60 s, and the cross-validated reference is the most
        # expensive object in the pipeline. The demo therefore runs G2 on Van der Pol
        # alone; `odeforge benchmark` keeps both systems. The two gates share one cached
        # reference per system, so G3 costs nothing beyond G2.
        g2_systems = ("van_der_pol",) if quick else G2_SYSTEMS
        gates: dict[str, Any] = {
            "G1": gate_g1(config=cfg),
            "G2": gate_g2(config=cfg, systems=g2_systems),
        }
        if quick:
            # Robertson is excluded from the demo path on cost grounds: its stiffness
            # costs several seconds per solve and its cross-validated reference costs
            # several more, so including it would dominate the demo's runtime. The full
            # ``odeforge benchmark`` command still runs it.
            gates["G3"] = gate_g3(config=cfg, systems=("van_der_pol", "lorenz"))
        else:
            gates["G3"] = gate_g3(config=cfg)

        systems_report = []
        for name in DEMO_SYSTEMS if quick else SYSTEM_NAMES:
            try:
                systems_report.append(self.evaluate_system(name))
            except (ValueError, RuntimeError) as exc:
                systems_report.append({"system": name, "error": str(exc)})

        report: dict[str, Any] = {
            "meta": self.meta(),
            "config": cfg.as_dict(),
            "gates": gates,
            "thresholds": GATE_THRESHOLDS,
            "flagship": FLAGSHIP,
            "systems": systems_report,
            "ablation": _ablation(config=cfg),
            "passed": {k: bool(v.get("passed")) for k, v in gates.items()},
            "wall_sec_total": float(time.perf_counter() - started),
        }
        report["passed"]["all"] = bool(all(report["passed"].values()))
        return report

    def ablation_report(self, *, system: str = "double_well") -> dict[str, Any]:
        """Ablations isolating each routing and flagship component."""
        set_all(self.config.seed)
        rows = _ablation(config=self.config, system=system)
        return {
            "system": system,
            "rows": rows,
            "n_variants": len(rows),
            "degradation": {
                r["variant"]: r.get("degradation_x") for r in rows if "degradation_x" in r
            },
        }

    def invariants_report(self, *, systems: tuple[str, ...] = DEMO_SYSTEMS) -> dict[str, Any]:
        """Invariant checks across the family, each with its measured value."""
        set_all(self.config.seed)
        rows: list[dict[str, Any]] = []
        for name in systems:
            spec = build_system(name)
            try:
                traj = self.run(name)
            except (ValueError, RuntimeError) as exc:
                rows.append({"system": name, "error": str(exc)})
                continue
            checks = [dict(c) for c in check_invariants(spec, traj)]
            report = self.detector.analyse(spec)
            rows.append(
                {
                    "system": name,
                    "method": traj.method,
                    "invariants": checks,
                    "hamiltonian": report.as_dict(),
                }
            )
        return {"rows": rows, "n_passed": sum(1 for r in rows if r.get("invariants"))}

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict[str, Any]:
        """Environment and provenance information recorded with every report."""
        return {
            "author": "晨星 (Chen Xing)",
            "email": "CJX0712@users.noreply.github.com",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy_available": available_scipy(),
            "seed_state": seed_state(),
            "seed": get_seed(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def write_report(self, report: dict[str, Any], path: str | Path) -> Path:
        """Write a report as JSON, then **read it back and verify** the round trip.

        Reading back is not ceremonial: a report is the only evidence the run happened,
        so a silently truncated or non-serialisable artefact would otherwise look like a
        result. The function raises if any gate verdict does not survive the round trip.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(
                report, fh, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default
            )
        with out.open("r", encoding="utf-8") as fh:
            back = json.load(fh)
        for gate, payload in back.get("gates", {}).items():
            original = report["gates"][gate].get("passed")
            if bool(payload.get("passed")) != bool(original):
                msg = f"gate {gate} verdict did not survive the JSON round trip"
                raise ValueError(msg)
        return out


def _json_default(obj: Any) -> Any:
    """Fallback encoder so numpy scalars and arrays survive serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    msg = f"object of type {type(obj).__name__} is not JSON serialisable"
    raise TypeError(msg)

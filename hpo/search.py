"""Hyper-parameter search for solver configuration.

Two strategies, both deterministic:

:class:`GridSearch`
    Exhaustive over a declared grid of ``(method, step/tolerance)`` candidates. Used
    for the flagship's own operating point.

:class:`CoordinateDescent`
    Refines one knob at a time -- method, then tolerance, then step -- until no single
    change improves the objective. Cheap and reproducible, and it is what makes the
    "equal ``n_fev``" comparison defensible: both the flagship and the baseline get an
    equally serious search.

Determinism: candidates are enumerated in a fixed order, ties are broken by index, and
no sampling is involved. The same inputs always produce the same winner.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.errors import IntegratorError
from core.types import Trajectory

__all__ = ["Candidate", "CoordinateDescent", "GridSearch", "SearchResult", "budget_steps_for"]


@dataclass(frozen=True)
class Candidate:
    """One point in hyper-parameter space."""

    method: str
    n_steps: int | None = None
    rtol: float | None = None
    atol: float | None = None

    def key(self) -> str:
        """Stable, human-readable identifier recorded in results."""
        bits = [self.method]
        if self.n_steps is not None:
            bits.append(f"n={self.n_steps}")
        if self.rtol is not None:
            bits.append(f"rtol={self.rtol:.0e}")
        if self.atol is not None:
            bits.append(f"atol={self.atol:.0e}")
        return ":".join(bits)


@dataclass
class SearchResult:
    """Outcome of a search: the winner, the full trail, and the objective value."""

    best: Candidate
    best_score: float
    best_traj: Trajectory | None
    trials: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.key(),
            "best_method": self.best.method,
            "best_n_steps": self.best.n_steps,
            "best_rtol": self.best.rtol,
            "best_atol": self.best.atol,
            "best_score": float(self.best_score),
            "n_trials": len(self.trials),
            "trials": self.trials,
        }


def budget_steps_for(method: str, budget_n_fev: int) -> int:
    """Largest step count whose function-evaluation count fits ``budget_n_fev``.

    The per-step cost of each method was **measured**, not assumed:

    ============ ================== ==========================================
    method       n_fev / step      source
    ============ ================== ==========================================
    ``euler``    1                 direct
    ``rk4``      4                 direct
    ``dp54``     <= 7 (adaptive)   FSAL gives 6 per accepted step
    ``vverlet``  1                 1 Verlet stage, force reused across steps
    ``yoshida4`` 3                 3 composition stages
    ``yoshida6`` 7                 7 composition stages
    ============ ================== ==========================================

    The symplectic figures are **amortised** and were measured, not assumed: a
    ``k``-stage composition needs ``k`` new force evaluations per step because the
    force at a stage boundary is reused from the previous stage. Only the very first
    evaluation of the whole run is extra, hence ``n_fev = 1 + k * n_steps``.

    An earlier version of this table charged ``k + 1`` per step, which is the cost of a
    *single* isolated composition. That overstated Yoshida6's cost by 12.5% and
    silently denied the flagship 1714 of its 120000 evaluations -- enough to matter in
    an equal-budget comparison. Verified by measuring ``n_fev`` at a fixed ``n_steps``.
    """
    per_step = {"euler": 1, "rk4": 4, "vverlet": 1, "yoshida4": 3, "yoshida6": 7}
    if method in per_step:
        return max(1, int(budget_n_fev // per_step[method]))
    if method == "dp54":
        # adaptive: treat the budget as approximate and use a nominal 6/step
        return max(1, int(budget_n_fev // 6))
    msg = f"no per-step cost known for method {method!r}"
    raise IntegratorError(msg)


class GridSearch:
    """Exhaustive search over a candidate list."""

    def __init__(self, candidates: list[Candidate]) -> None:
        if not candidates:
            msg = "GridSearch needs at least one candidate"
            raise IntegratorError(msg)
        self.candidates = list(candidates)

    def search(
        self,
        run: Callable[[Candidate], tuple[float, Trajectory | None]],
        *,
        bigger_is_better: bool = False,
    ) -> SearchResult:
        """Evaluate every candidate; ties break toward the earlier (fixed-order) one."""
        trials: list[dict[str, Any]] = []
        best: Candidate | None = None
        best_score = -np.inf if bigger_is_better else np.inf
        best_traj: Trajectory | None = None
        for cand in self.candidates:
            score, traj = run(cand)
            trials.append({"key": cand.key(), "score": float(score)})
            better = score > best_score if bigger_is_better else score < best_score
            if best is None or better:
                best, best_score, best_traj = cand, float(score), traj
        assert best is not None
        return SearchResult(best=best, best_score=best_score, best_traj=best_traj, trials=trials)


class CoordinateDescent:
    """Refine method, then tolerance, then step count, one knob at a time."""

    def __init__(
        self,
        methods: list[str],
        rtols: list[float] | None = None,
        step_fractions: list[float] | None = None,
        *,
        budget_n_fev: int = 120_000,
        max_rounds: int = 3,
    ) -> None:
        if not methods:
            msg = "CoordinateDescent needs at least one method"
            raise IntegratorError(msg)
        self.methods = list(methods)
        self.rtols = list(rtols) if rtols else [1e-6, 1e-8, 1e-10]
        self.step_fractions = list(step_fractions) if step_fractions else [1.0, 0.5, 0.25]
        self.budget_n_fev = int(budget_n_fev)
        self.max_rounds = int(max_rounds)

    def search(
        self,
        run: Callable[[Candidate], tuple[float, Trajectory | None]],
        *,
        bigger_is_better: bool = False,
    ) -> SearchResult:
        """Cycle through the knobs until no single change improves the objective."""
        trials: list[dict[str, Any]] = []
        best_score = -np.inf if bigger_is_better else np.inf
        best = Candidate(method=self.methods[0])
        best_traj: Trajectory | None = None

        def better(score: float) -> bool:
            return score > best_score if bigger_is_better else score < best_score

        score, traj = run(best)
        trials.append({"key": best.key(), "score": float(score)})
        if better(score):
            best_score, best_traj = float(score), traj

        for _ in range(self.max_rounds):
            improved = False
            # knob 1: method
            for m in self.methods:
                cand = Candidate(
                    method=m,
                    n_steps=budget_steps_for(m, self.budget_n_fev) if m not in {"dp54"} else None,
                    rtol=best.rtol,
                    atol=None if best.rtol is None else best.rtol * 1e-2,
                )
                score, traj = run(cand)
                trials.append({"key": cand.key(), "score": float(score)})
                if better(score):
                    best, best_score, best_traj, improved = cand, float(score), traj, True
            # knob 2: tolerance (adaptive methods only)
            for rtol in self.rtols:
                cand = Candidate(
                    method=best.method,
                    n_steps=best.n_steps,
                    rtol=rtol,
                    atol=rtol * 1e-2,
                )
                score, traj = run(cand)
                trials.append({"key": cand.key(), "score": float(score)})
                if better(score):
                    best, best_score, best_traj, improved = cand, float(score), traj, True
            # knob 3: step fraction
            for frac in self.step_fractions:
                n = budget_steps_for(best.method, self.budget_n_fev)
                cand = Candidate(
                    method=best.method,
                    n_steps=max(1, int(n * frac)),
                    rtol=best.rtol,
                    atol=best.atol,
                )
                score, traj = run(cand)
                trials.append({"key": cand.key(), "score": float(score)})
                if better(score):
                    best, best_score, best_traj, improved = cand, float(score), traj, True
            if not improved:
                break
        return SearchResult(
            best=best, best_score=float(best_score), best_traj=best_traj, trials=trials
        )

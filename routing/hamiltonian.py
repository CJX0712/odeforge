"""Numerical detection of Hamiltonian structure.

Three independent indicators, each normalised to ``[0, 1]`` where ``1`` means
"strongly Hamiltonian". No single one is trusted:

1. **Involutivity (reversibility) test** -- the defining property
   ``f(x) = -f(-x)`` of a Hamiltonian vector field. Evaluated as the maximum relative
   deviation over a probe cloud. This is the strongest and most direct indicator.
2. **Divergence-free test** -- ``tr J(f) = 0`` for a Hamiltonian field. Checked for
   *consistency* across probes rather than at a single point, and normalised against
   the field's own scale so that systems with large ``||f||`` are not penalised.
3. **Linear Hamiltonian structure test** -- the rigorous condition. A matrix ``A`` is
   the Jacobian of a Hamiltonian vector field iff there exists a symmetric matrix ``S``
   with ``A^T S + S A = 0``. This is *tested*, not assumed: the linear map
   ``vec(S) -> vec(A^T S + S A)`` is built and its smallest singular value is taken as
   the residual, normalised by ``||A||``.

   .. note::

      A commonly repeated shortcut is "the Jacobian of a Hamiltonian system is
      symmetric". **That is false.** For ``H = p^2/2 + q^2/2`` the Jacobian is
      ``[[0, 1], [-1, 0]]``, which is *antisymmetric* (``A + A^T = 0``), and its
      asymmetry is a perfect ``2.0``. The correct statement involves a metric ``S`` as
      above. For ``2x2`` systems the condition collapses to ``(tr A)(det A) = 0``,
      which is what makes the test cheap and exact for the separable family.

A fourth, optional signal is the **automatic discovery of conserved quantities**: a
greedy search for a linear-in-``y`` functional ``C(y) = c.y`` with
``grad C . f ~ 0``, solved as a small least-squares/nullspace problem over probe
points. It is reported separately and does not enter the score, because a nullspace
direction can exist for accidental reasons.

Determinism
-----------
Probe points come from :func:`core.seed.rng`, i.e. ``numpy.random.default_rng`` seeded
through the single global entry point :func:`core.seed.set_all`. The seed and the probe
count are recorded in every :class:`~core.types.HamiltonianReport`, so any score can be
reproduced exactly. No module ever calls ``np.random.randn`` directly.

Why not just ``trace(J)``?
--------------------------
``trace(J) = 0`` is necessary but far from sufficient, and the symmetric case is
degenerate: for the harmonic oscillator ``J = [[0, 1], [-1, 0]]`` has trace exactly
zero, but so do many non-Hamiltonian fields whose divergence happens to vanish at the
sampled point. Sampling consistency plus the involutivity test removes that
degeneracy.

A measured false positive worth recording
-----------------------------------------
A first implementation scored Van der Pol at **0.859** and Lorenz at **0.766** -- both
"Hamiltonian" -- while only Lotka-Volterra (0.368) and Robertson (0.274) were rejected.
The cause was that the involutivity test alone is far too weak: Van der Pol's field
``(p, q(1-q^2) - p)`` is an *odd* function, so ``f(x) = -f(-x)`` holds exactly and
earns a perfect reversibility score, even though its divergence is ``-1`` everywhere and
it is plainly not Hamiltonian. The fix was to make the linear-Hamiltonian-structure test
decisive rather than advisory, and to require *all three* indicators before voting yes.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.errors import HamiltonianError
from core.seed import get_seed, rng
from core.types import HamiltonianReport, SystemSpec

__all__ = [
    "WEIGHTS",
    "HamiltonianDetector",
    "discover_linear_conserved",
    "divergence",
    "involutivity_error",
    "linear_hamiltonian_residual",
    "numerical_jacobian",
    "probe_cloud",
]

#: Fixed indicator weights. They sum to 1 so ``score`` is a convex combination.
WEIGHTS: dict[str, float] = {
    "reversibility": 0.50,
    "divergence_free": 0.30,
    "linear_structure": 0.20,
}

#: A relative deviation at or below this counts as perfect agreement.
_SCALE_FLOOR = 1e-12


def probe_cloud(y0: np.ndarray, n_probe: int, spread: float = 0.35) -> np.ndarray:
    """Deterministic probe points around ``y0``.

    Uses ``numpy.random.default_rng`` via :func:`core.seed.rng` with a fixed offset, so
    the same global seed always yields the same cloud.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    gen = rng(offset=17)
    return y0 + spread * gen.standard_normal((int(n_probe), y0.size))


def numerical_jacobian(
    rhs: Any, t: float, y: np.ndarray, *, eps: float | None = None
) -> np.ndarray:
    """Central-difference Jacobian ``df/dy`` of shape ``(n, n)``."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if eps is None:
        eps = 1e-6
    jac = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        step = eps * max(1.0, abs(float(y[j])))
        yp = y.copy()
        ym = y.copy()
        yp[j] += step
        ym[j] -= step
        jac[:, j] = (
            np.asarray(rhs(t, yp), dtype=np.float64) - np.asarray(rhs(t, ym), dtype=np.float64)
        ) / (2.0 * step)
    return jac


def divergence(rhs: Any, t: float, y: np.ndarray) -> float:
    """``tr(df/dy)`` -- the phase-space volume expansion rate."""
    return float(np.trace(numerical_jacobian(rhs, t, y)))


def involutivity_error(rhs: Any, points: np.ndarray, t: float = 0.0) -> float:
    """Maximum relative violation of ``f(x) == -f(-x)`` over ``points``.

    Returns ``0.0`` for a field that is exactly involutive, and grows towards ``1`` as
    the violation becomes comparable to the field magnitude.
    """
    worst = 0.0
    for x in points:
        fx = np.asarray(rhs(t, x), dtype=np.float64)
        fm = np.asarray(rhs(t, -x), dtype=np.float64)
        denom = max(_SCALE_FLOOR, float(np.max(np.abs(fx))), float(np.max(np.abs(fm))))
        rel = float(np.max(np.abs(fx + fm))) / denom
        worst = max(worst, rel)
    return worst


def _sym_basis(n: int) -> np.ndarray:
    """Orthonormal basis of symmetric ``n x n`` matrices, shape ``(n(n+1)/2, n*n)``."""
    rows = []
    for i in range(n):
        for j in range(i, n):
            m = np.zeros((n, n))
            m[i, j] = 1.0
            m[j, i] = 1.0
            if i == j:
                m[i, i] = np.sqrt(2.0)  # orthonormal under the Frobenius inner product
            rows.append(m.ravel())
    return np.asarray(rows, dtype=np.float64)


def linear_hamiltonian_residual(matrix: np.ndarray) -> float:
    """Smallest singular value of ``S -> A^T S + S A`` restricted to symmetric ``S``.

    ``A`` is the Jacobian of a Hamiltonian vector field (with some symmetric metric ``S``)
    iff this map has a nontrivial kernel. The residual is normalised by ``||A||_F`` so it
    is scale free: ``0`` means exactly Hamiltonian, larger means further from it.

    For ``n = 2`` this condition collapses to ``(tr A)(det A) = 0``, which gives an
    independent cross-check used by the test-suite.
    """
    a = np.asarray(matrix, dtype=np.float64)
    n = a.shape[0]
    basis = _sym_basis(n)  # (m, n*n)
    nrm = float(np.linalg.norm(a, "fro"))
    if nrm <= _SCALE_FLOOR:
        return 0.0
    out = np.empty((n * n, basis.shape[0]), dtype=np.float64)
    for col in range(basis.shape[0]):
        s_mat = basis[col].reshape(n, n)
        out[:, col] = (a.T @ s_mat + s_mat @ a).ravel()
    sigma = np.linalg.svd(out, compute_uv=False)
    smallest = float(sigma[-1]) if sigma.size else 0.0
    return smallest / nrm


def matrix_hamiltonian_residual_2x2(matrix: np.ndarray) -> float:
    """Independent 2x2 shortcut: ``|tr A * det A| / ||A||_F^3``.

    Mathematically equivalent to :func:`linear_hamiltonian_residual` for ``n = 2`` and
    far cheaper, so it is used to cross-check the SVD-based residual.
    """
    a = np.asarray(matrix, dtype=np.float64)
    nrm = float(np.linalg.norm(a, "fro"))
    if nrm <= _SCALE_FLOOR:
        return 0.0
    return float(abs(np.trace(a) * np.linalg.det(a))) / (nrm**3)


def _consistency(values: np.ndarray) -> float:
    """How tightly a quantity agrees across probes: ``1`` when perfectly constant."""
    scale = max(_SCALE_FLOOR, float(np.max(np.abs(values))))
    spread = float(np.max(values) - np.min(values)) / scale
    return float(1.0 / (1.0 + spread))


def _as_score(error: float, reference: float) -> float:
    """Map a non-negative relative error onto ``[0, 1]`` with ``error=0 -> 1``."""
    if not np.isfinite(error):
        return 0.0
    return float(1.0 / (1.0 + error / reference)) if reference > 0 else 0.0


def discover_linear_conserved(
    rhs: Any, points: np.ndarray, t: float = 0.0, *, tol: float = 1e-8
) -> dict[str, Any]:
    """Greedily look for a conserved quantity of the form ``C(y) = c.y``.

    For such a functional, conservation is exactly ``c . f(y) = 0`` for every state, i.e.
    ``c`` lies in the nullspace of the matrix whose rows are the stacked ``f(y_i)``.
    The SVD is used to find that nullspace, then each candidate direction is scored by
    its actual relative drift along a short probe integration.

    Reported for information only -- a nullspace direction is *necessary* for a linear
    invariant but not *sufficient*, so it does not enter the Hamiltonian score.
    """
    n = points.shape[1]
    mat = np.array([np.asarray(rhs(t, p), dtype=np.float64) for p in points])
    _, singular, vt = np.linalg.svd(mat, full_matrices=False)
    if vt.shape[0] < n:
        return {"found": False, "vector": None, "residual": float("inf"), "n_candidates": 0}
    threshold = max(1e-12, float(singular[0]) * tol) if singular.size else 1e-12
    candidates = [
        row
        for row in vt
        if float(np.linalg.norm(row)) <= threshold
        or float(np.max(np.abs(mat @ row))) <= max(1e-12, float(np.max(np.abs(mat))) * tol)
    ]
    if not candidates:
        return {"found": False, "vector": None, "residual": float("inf"), "n_candidates": 0}
    best: dict[str, Any] = {
        "found": False,
        "vector": None,
        "residual": float("inf"),
        "n_candidates": len(candidates),
    }
    for cand in candidates:
        norm = float(np.linalg.norm(cand))
        if norm < _SCALE_FLOOR:
            continue
        resid = float(np.max(np.abs(mat @ cand))) / norm
        if resid < best["residual"]:
            best = {
                "found": True,
                "vector": cand / norm,
                "residual": resid,
                "n_candidates": len(candidates),
            }
    return best


class HamiltonianDetector:
    """Multi-indicator numerical Hamiltonian structure detector."""

    def __init__(
        self,
        *,
        n_probe: int = 24,
        threshold: float = 0.72,
        spread: float = 0.35,
        vote_threshold: float = 0.60,
    ) -> None:
        if n_probe < 2:
            msg = f"n_probe must be >= 2, got {n_probe}"
            raise HamiltonianError(msg)
        if not 0.0 < threshold < 1.0:
            msg = f"threshold must lie in (0, 1), got {threshold}"
            raise HamiltonianError(msg)
        self.n_probe = int(n_probe)
        self.threshold = float(threshold)
        self.spread = float(spread)
        self.vote_threshold = float(vote_threshold)

    def analyse(self, spec: SystemSpec, *, t_probe: float = 0.0) -> HamiltonianReport:
        """Score ``spec`` for Hamiltonian structure.

        Returns a :class:`~core.types.HamiltonianReport` whose ``verdict`` is
        ``score >= threshold``. ``separability`` (whether the system is in canonical
        ``(q, p)`` form with ``dq/dt = p``) is reported in ``detail`` because it, not
        Hamiltonicity alone, decides whether a symplectic integrator may be used.
        """
        points = probe_cloud(spec.y0, self.n_probe, self.spread)
        rhs = spec.rhs

        inv_err = involutivity_error(rhs, points, t_probe)
        f_mag = float(
            np.mean([np.max(np.abs(np.asarray(rhs(t_probe, p), dtype=np.float64))) for p in points])
        )
        reversibility = _as_score(inv_err, 1.0)

        divs = np.array([divergence(rhs, t_probe, p) for p in points])
        div_scale = max(_SCALE_FLOOR, f_mag)
        div_rel = float(np.max(np.abs(divs))) / div_scale
        divergence_free = _as_score(div_rel, 1.0) * _consistency(divs / div_scale)

        lin_res = float(
            np.mean(
                [linear_hamiltonian_residual(numerical_jacobian(rhs, t_probe, p)) for p in points]
            )
        )
        linear_structure = _as_score(lin_res, 1e-3)

        score = (
            WEIGHTS["reversibility"] * reversibility
            + WEIGHTS["divergence_free"] * divergence_free
            + WEIGHTS["linear_structure"] * linear_structure
        )
        # A field is only accepted as Hamiltonian when EVERY indicator agrees. The
        # weighted mean alone is too forgiving: Van der Pol is odd (reversibility 1.0)
        # yet has divergence -1 and no symmetric metric S, and a mean would let it pass.
        all_indicators_agree = bool(
            reversibility >= self.vote_threshold
            and divergence_free >= self.vote_threshold
            and linear_structure >= self.vote_threshold
        )
        verdict = bool(all_indicators_agree and score >= self.threshold)
        conserved = discover_linear_conserved(rhs, points, t_probe)

        detail = {
            "involutivity_error": float(inv_err),
            "max_rel_divergence": float(div_rel),
            "divergence_consistency": float(_consistency(divs / div_scale)),
            "linear_hamiltonian_residual": float(lin_res),
            "field_magnitude": f_mag,
            "separability": bool(spec.is_separable),
            "linear_conserved_found": bool(conserved["found"]),
            "linear_conserved_residual": float(conserved["residual"]),
        }
        return HamiltonianReport(
            score=float(score),
            verdict=verdict,
            reversibility=float(reversibility),
            divergence_free=float(divergence_free),
            jacobian_symmetry=float(linear_structure),
            n_probe=self.n_probe,
            seed=int(get_seed() or 0),
            detail=detail,
        )

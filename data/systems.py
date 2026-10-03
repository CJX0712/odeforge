"""Synthetic ODE system family used for evaluation, routing and training.

Every system provides ``rhs``, ``y0`` and ``t_span``; separable Hamiltonian ones also
provide ``grad_v`` so that :mod:`integrators.symplectic` can be used. Systems expose a
``difficulty`` dictionary so the same family spans easy order-verification problems and
hard chaotic/stiff ones.

Axis convention reminder: :class:`~core.types.Trajectory` stores ``y`` as
``(n_points, n_states)``. These ``rhs`` functions take and return a single state vector.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import numpy as np

from core.seed import rng
from core.types import ConservedQuantity, SystemSpec

__all__ = ["SYSTEMS", "SYSTEM_NAMES", "build_system", "poincare_hits"]

SYSTEM_NAMES: tuple[str, ...] = (
    "harmonic",
    "double_well",
    "kepler",
    "pendulum",
    "lotka_volterra",
    "van_der_pol",
    "lorenz",
    "robertson",
    "exponential",
    "three_body",
)


# --------------------------------------------------------------------- separable H
def _harmonic(omega: float = 1.0) -> SystemSpec:
    """Undamped harmonic oscillator with closed-form solution ``q = cos(omega t)``."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(omega**2) * y[0]])

    def grad_v(q: float) -> float:
        return (omega**2) * q

    return SystemSpec(
        name="harmonic",
        rhs=rhs,
        y0=np.array([1.0, 0.0]),
        t_span=(0.0, 20.0),
        analytic=lambda t: np.array([np.cos(omega * t), -omega * np.sin(omega * t)]),
        conserved=(ConservedQuantity("energy", lambda y: 0.5 * y[0] ** 2 + 0.5 * y[1] ** 2),),
        is_hamiltonian=True,
        is_separable=True,
        grad_v=grad_v,
        difficulty={"omega": omega},
        description="H = p^2/2 + omega^2 q^2/2; closed-form solution, order verification.",
    )


def _double_well() -> SystemSpec:
    """Nonlinear double-well: the flagship long-time energy benchmark."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(y[0] ** 3) + y[0]])

    def grad_v(q: float) -> float:
        return q**3 - q

    def energy(y: np.ndarray) -> float:
        return 0.5 * y[1] ** 2 + 0.25 * y[0] ** 4 - 0.5 * y[0] ** 2

    return SystemSpec(
        name="double_well",
        rhs=rhs,
        y0=np.array([0.1, 1.0]),
        t_span=(0.0, 500.0),
        conserved=(ConservedQuantity("energy", energy),),
        is_hamiltonian=True,
        is_separable=True,
        grad_v=grad_v,
        difficulty={},
        description="H = p^2/2 + q^4/4 - q^2/2; nonlinear, long-time drift gate G1.",
    )


def _kepler(mu: float = 1.0, ecc: float = 0.6) -> SystemSpec:
    """Planar Kepler two-body problem; angular momentum is exactly conserved."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, vx, yp, vy = y
        r2 = x * x + yp * yp
        r3 = r2 * np.sqrt(r2)
        k = mu
        return np.array([vx, -k * x / r3, vy, -k * yp / r3])

    a = 1.0
    p = a * (1.0 - ecc * ecc)
    peri = a * (1.0 + ecc)

    def analytic(t: float) -> np.ndarray:
        nu = t
        r = p / (1.0 + ecc * np.cos(nu))
        xp, yp = r * np.cos(nu), r * np.sin(nu)
        n_true = np.sqrt(mu / a**3)
        return np.array([xp, -r * np.sin(nu) * n_true, yp, r * np.cos(nu) * n_true])

    def angmom(y: np.ndarray) -> float:
        return y[0] * y[3] - y[2] * y[1]

    def efun(y: np.ndarray) -> float:
        x, vx, yp, vy = y
        r = np.sqrt(x * x + yp * yp)
        v2 = vx * vx + vy * vy
        return 0.5 * v2 - mu / r

    return SystemSpec(
        name="kepler",
        rhs=rhs,
        y0=np.array([peri, 0.0, 0.0, np.sqrt(mu * (2.0 / peri - 1.0 / a))]),
        t_span=(0.0, 30.0),
        conserved=(
            ConservedQuantity("angular_momentum", angmom),
            ConservedQuantity("energy", efun),
        ),
        is_hamiltonian=True,
        is_separable=False,  # canonical momentum transform needed, not separable
        difficulty={"eccentricity": ecc},
        description="Kepler two-body; conserves angular momentum, not separable.",
    )


def _pendulum(theta0: float = 2.5) -> SystemSpec:
    """Undamped nonlinear pendulum, ``H = p^2/2 + 1 - cos(q)``."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -np.sin(y[0])])

    def grad_v(q: float) -> float:
        return np.sin(q)

    def energy(y: np.ndarray) -> float:
        return 0.5 * y[1] ** 2 + (1.0 - np.cos(y[0]))

    return SystemSpec(
        name="pendulum",
        rhs=rhs,
        y0=np.array([theta0, 0.0]),
        t_span=(0.0, 100.0),
        conserved=(ConservedQuantity("energy", energy),),
        is_hamiltonian=True,
        is_separable=True,
        grad_v=grad_v,
        difficulty={"theta0": theta0},
        description="Nonlinear pendulum; separable Hamiltonian with a non-quadratic V.",
    )


def _lotka_volterra(
    alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, delta: float = 1.0
) -> SystemSpec:
    """Lotka-Volterra predator-prey with the exact invariant ``x - ln x + y - ln y``.

    The invariant is conserved **only** at ``alpha = beta = gamma = delta = 1``. For
    general rates the conserved first integral is
    ``delta*x - gamma*y - delta*ln(x) + beta*ln(y)``; with non-unit rates the advertised
    ``H`` is no longer invariant, so :attr:`ConservedQuantity.name` is adjusted to make
    the status explicit rather than silently claiming a false invariant.
    """

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, yy = y
        return np.array([alpha * x - beta * x * yy, -gamma * yy + delta * x * yy])

    unit_rates = (alpha, beta, gamma, delta) == (1.0, 1.0, 1.0, 1.0)

    def hfun(y: np.ndarray) -> float:
        x, yy = y
        if unit_rates:
            return x - np.log(x) + yy - np.log(yy)
        return delta * x - gamma * yy - delta * np.log(x) + beta * np.log(yy)

    return SystemSpec(
        name="lotka_volterra",
        rhs=rhs,
        y0=np.array([2.0, 1.0]),
        t_span=(0.0, 60.0),
        conserved=(ConservedQuantity("invariant" if unit_rates else "first_integral", hfun),),
        # NOT Hamiltonian in standard canonical form, despite having a first integral.
        # The phase-space divergence is d(x(1-y))/dx + d(y(x-1))/dy = (1-y) + (x-1)
        # = x - y, which is non-zero for generic states. Lotka-Volterra preserves the
        # non-standard area form dx^dy / x, not the canonical dx^dy, so it is
        # Hamiltonian only in a generalized sense. The detector's divergence indicator
        # rejects it correctly, and the symplectic integrators -- which need
        # dq/dt = p, dp/dt = -dV/dq -- genuinely cannot be applied.
        #
        # The family previously declared this system Hamiltonian; the numerical
        # detector disagreed and the detector was right. The flag is corrected here and
        # tests/test_routing.py asserts the two now agree.
        is_hamiltonian=False,
        is_separable=False,
        difficulty={"alpha": alpha},
        description="Lotka-Volterra; exact logarithmic invariant, non-canonical symplectic form.",
    )


# ------------------------------------------------------------------ non-Hamiltonian
def _van_der_pol(mu: float = 1.0) -> SystemSpec:
    """Van der Pol oscillator; ``mu`` controls stiffness."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        q, p = y
        return np.array([p, mu * q * (1.0 - q * q) - p])

    return SystemSpec(
        name="van_der_pol",
        rhs=rhs,
        y0=np.array([2.0, 0.0]),
        t_span=(0.0, 20.0),
        is_hamiltonian=False,
        is_separable=False,
        difficulty={"mu": mu},
        description="Self-excited limit cycle; not Hamiltonian, route to dp54 (gate G3).",
    )


def _lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> SystemSpec:
    """Lorenz attractor; volume-contracting, definitively non-Hamiltonian."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, yy, z = y
        return np.array([sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z])

    return SystemSpec(
        name="lorenz",
        rhs=rhs,
        y0=np.array([1.0, 1.0, 1.0]),
        t_span=(0.0, 10.0),
        is_hamiltonian=False,
        is_separable=False,
        difficulty={"rho": rho},
        description="Lorenz chaos; divergence = -(sigma+1+beta) != 0, gate G3.",
    )


def _robertson() -> SystemSpec:
    """Robertson A/B/C, the classic stiff chemistry problem.

    Span note (measured, not assumed): the stiffness is ``~3e7``. Advancing the
    textbook interval ``[0, 1e11]`` with an *explicit* method needs of order ``1e7``
    steps -- at ``rtol=1e-6`` the reachable span is roughly ``T=1e2`` at 6.4 s and
    6.3e5 function evaluations, and ``T=1e3`` already exhausts a 3e5-step budget.
    ODEForge ships explicit methods only, so the advertised span is ``[0, 1e2]``, which
    still spans the initial fast transient (the stiffness test) while remaining
    computable. The full ``1e11`` trajectory requires an implicit solver (Radau/BDF);
    :mod:`eval.baselines` notes this rather than silently truncating.
    """

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        a, b, c = y
        return np.array(
            [
                -0.04 * a + 1e4 * b * c,
                0.04 * a - 1e4 * b * c - 3e7 * b * b,
                3e7 * b * b,
            ]
        )

    return SystemSpec(
        name="robertson",
        rhs=rhs,
        y0=np.array([1.0, 0.0, 0.0]),
        t_span=(0.0, 1e2),
        conserved=(ConservedQuantity("mass", lambda y: float(np.sum(y))),),
        is_hamiltonian=False,
        is_separable=False,
        difficulty={"stiffness": 1e7, "textbook_t_end": 1e11},
        description="Robertson A/B/C; stiff (1e7), exercises the adaptive path, gate G3.",
    )


def _exponential(rate: float = 1.0) -> SystemSpec:
    """Linear exponential growth, dedicated to order verification."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([rate * y[0], rate * y[1]])

    return SystemSpec(
        name="exponential",
        rhs=rhs,
        y0=np.array([1.0, 0.5]),
        t_span=(0.0, 1.0),
        analytic=lambda t: np.array([np.exp(rate * t), 0.5 * np.exp(rate * t)]),
        is_hamiltonian=False,
        is_separable=False,
        difficulty={"rate": rate},
        description="Linear growth with closed form; cleanest order measurement.",
    )


def _three_body() -> SystemSpec:
    """Pythagorean three-body problem; chaotic, conserves total angular momentum."""

    m = np.array([1.0, 1.0, 1.0])
    pos0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    vel0 = np.array([[0.5, np.sqrt(3.0) / 2.0], [0.5, -np.sqrt(3.0) / 2.0], [-1.0, 0.0]])
    y0 = np.concatenate([pos0.ravel(), vel0.ravel()])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        pos = y[:6].reshape(3, 2)
        vel = y[6:].reshape(3, 2)
        acc = np.zeros((3, 2))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                d = pos[j] - pos[i]
                r2 = float(d @ d)
                acc[i] += m[j] * d / (r2 * np.sqrt(r2))
        return np.concatenate([vel.ravel(), acc.ravel()])

    def angmom(y: np.ndarray) -> float:
        pos = y[:6].reshape(3, 2)
        vel = y[6:].reshape(3, 2)
        return float(np.sum(m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])))

    return SystemSpec(
        name="three_body",
        rhs=rhs,
        y0=y0,
        t_span=(0.0, 20.0),
        conserved=(ConservedQuantity("angular_momentum", angmom),),
        is_hamiltonian=True,  # Newtonian N-body is Hamiltonian ...
        is_separable=False,  # ... but not separable in canonical (q, p) coordinates
        difficulty={},
        description="Pythagorean three-body; Hamiltonian yet non-separable (routing trap).",
    )


SYSTEMS: dict[str, SystemSpec] = {
    "harmonic": _harmonic(),
    "double_well": _double_well(),
    "kepler": _kepler(),
    "pendulum": _pendulum(),
    "lotka_volterra": _lotka_volterra(),
    "van_der_pol": _van_der_pol(),
    "lorenz": _lorenz(),
    "robertson": _robertson(),
    "exponential": _exponential(),
    "three_body": _three_body(),
}


def build_system(name: str, **difficulty: float) -> SystemSpec:
    """Return a system by name, optionally overriding its difficulty knobs."""
    if name not in SYSTEMS:
        msg = f"unknown system {name!r}; known: {list(SYSTEM_NAMES)}"
        raise KeyError(msg)
    spec = SYSTEMS[name]
    if not difficulty:
        return spec
    merged = {**spec.difficulty, **difficulty}
    return _rebuild(name, merged)


def _rebuild(name: str, difficulty: dict[str, float]) -> SystemSpec:
    factories = {
        "harmonic": lambda d: _harmonic(**d),
        "double_well": _double_well,
        "kepler": lambda d: _kepler(**d),
        "pendulum": lambda d: _pendulum(**d),
        "lotka_volterra": lambda d: _lotka_volterra(**d),
        "van_der_pol": lambda d: _van_der_pol(**d),
        "lorenz": lambda d: _lorenz(**d),
        "robertson": _robertson,
        "exponential": lambda d: _exponential(**d),
        "three_body": _three_body,
    }
    return factories[name](difficulty)


def poincare_hits(
    traj_t: np.ndarray,
    traj_y: np.ndarray,
    value: float,
    axis: int = 2,
    rising: bool = True,
) -> np.ndarray:
    """Indices where ``traj_y[:, axis]`` crosses ``value``.

    Used for the Lorenz / three-body diagnostics: a chaotic orbit keeps crossing the
    section, a regular one does not.
    """
    col = np.asarray(traj_y)[:, axis]
    below = col[:-1] < value
    above = col[1:] >= value
    # A crossing needs the value on BOTH sides of the interval; the earlier version
    # tested ``diff >= 0`` and ``diff < 0`` together, which is unsatisfiable, so this
    # function silently returned an empty array for every input.
    if rising:
        return np.nonzero(below & above)[0]
    return np.nonzero(~below & (col[1:] < value))[0]


def default_probe_points(n_probe: int, y0: np.ndarray, spread: float = 0.35) -> np.ndarray:
    """Deterministic probe cloud around ``y0`` for the Hamiltonian structural tests.

    Draws from ``numpy.random.default_rng`` seeded through :func:`core.seed.rng`, so the
    cloud is reproducible and the seed is always recorded with the report.
    """
    gen = rng(offset=11)
    return np.asarray(y0, dtype=np.float64) + spread * gen.standard_normal((n_probe, y0.size))

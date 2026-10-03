"""Typed error hierarchy for ODEForge.

Every error carries a stable ``code`` in the ``E1xx`` .. ``E5xx`` range so that
callers (and the test-suite) can assert on failure modes without string matching.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

__all__ = [
    "ERR_CODES",
    "BaselineUnavailableError",
    "BudgetError",
    "ConfigError",
    "GradientCheckError",
    "HamiltonianError",
    "IntegratorError",
    "NotHamiltonianError",
    "ODEForgeError",
    "StepSizeError",
    "SystemError_",
]


class ODEForgeError(Exception):
    """Base class for every ODEForge error."""

    code = "E000"

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, object] = dict(detail or {})

    def __str__(self) -> str:
        if not self.detail:
            return f"[{self.code}] {self.message}"
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.detail.items()))
        return f"[{self.code}] {self.message} ({items})"


class ConfigError(ODEForgeError):
    """Configuration value missing, malformed, or failing schema validation."""

    code = "E100"


class SystemError_(ODEForgeError):
    """An ODE system specification is invalid or internally inconsistent."""

    code = "E200"


class HamiltonianError(ODEForgeError):
    """Hamiltonian structural analysis failed or was applied to a non-Hamiltonian system."""

    code = "E300"


class NotHamiltonianError(HamiltonianError):
    """A symplectic method was requested for a system that is not separable Hamiltonian."""

    code = "E301"


class IntegratorError(ODEForgeError):
    """An integrator could not complete the requested integration."""

    code = "E400"


class StepSizeError(IntegratorError):
    """The step size collapsed below the usable floor, or the budget was exhausted."""

    code = "E401"


class BudgetError(IntegratorError):
    """The ``n_fev`` budget was exhausted before the integration finished."""

    code = "E402"


class GradientCheckError(ODEForgeError):
    """An analytic gradient failed to agree with a finite-difference reference."""

    code = "E500"


class BaselineUnavailableError(ODEForgeError):
    """A required comparison baseline is unavailable (e.g. scipy not installed)."""

    code = "E501"


#: Stable registry of every error code, used by the CLI to print helpful hints.
ERR_CODES: dict[str, type[ODEForgeError]] = {
    cls.code: cls
    for cls in (
        ODEForgeError,
        ConfigError,
        SystemError_,
        HamiltonianError,
        NotHamiltonianError,
        IntegratorError,
        StepSizeError,
        BudgetError,
        GradientCheckError,
        BaselineUnavailableError,
    )
}

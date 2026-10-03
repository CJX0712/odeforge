"""Configuration with environment-variable overrides and schema validation.

Precedence (highest first):

1. explicit keyword arguments
2. ``ENV_ODEFORGE_*`` environment variables
3. built-in defaults

Every field is declared in :data:`SCHEMA` with a type and a validator, so a bad
override fails loudly at construction instead of producing a subtly wrong benchmark.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from core.errors import ConfigError

__all__ = ["ENV_PREFIX", "SCHEMA", "Config", "load_config"]

ENV_PREFIX = "ENV_ODEFORGE_"


@dataclass(frozen=True)
class FieldSpec:
    """Declarative schema entry for one configuration field."""

    default: Any
    kind: type
    validate: Callable[[Any], Any] | None = None
    doc: str = ""

    def check(self, value: Any) -> Any:
        if self.kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if self.kind is int and isinstance(value, bool):
            msg = f"expected int, got bool ({value!r})"
            raise ConfigError(msg)
        if not isinstance(value, self.kind):
            msg = f"expected {self.kind.__name__}, got {type(value).__name__} ({value!r})"
            raise ConfigError(msg, detail={"field": self.kind.__name__})
        if self.validate is not None:
            out = self.validate(value)
            if out is False:
                msg = f"value failed validation: {value!r}"
                raise ConfigError(msg)
        return value


def _positive(value: float) -> bool:
    return value > 0.0


def _positive_int(value: int) -> bool:
    return value > 0


def _unit_interval(value: float) -> bool:
    return 0.0 <= value <= 1.0


#: Declarative schema. Adding a field here is enough to make it configurable.
SCHEMA: dict[str, FieldSpec] = {
    "seed": FieldSpec(20260101, int, _positive_int, "Global seed for numpy and random."),
    "n_probe": FieldSpec(
        24, int, _positive_int, "Deterministic probe points for Hamiltonian tests."
    ),
    "hamiltonian_threshold": FieldSpec(
        0.72, float, _unit_interval, "Score above which a system counts as Hamiltonian."
    ),
    "budget_n_fev": FieldSpec(120_000, int, _positive_int, "Equal-n_fev comparison budget."),
    "max_steps": FieldSpec(2_000_000, int, _positive_int, "Hard cap on adaptive solver steps."),
    "rtol": FieldSpec(1e-8, float, _positive, "Relative tolerance for adaptive solvers."),
    "atol": FieldSpec(1e-10, float, _positive, "Absolute tolerance for adaptive solvers."),
    "tol_grid_power": FieldSpec(
        (2, 3, 4, 5, 6, 8, 10), tuple, None, "Exponents for the baseline tolerance grid."
    ),
    "noise_std": FieldSpec(1e-3, float, _positive, "Observation noise for the training task."),
    "n_seeds": FieldSpec(3, int, _positive_int, "Repeats for mean +/- std statistics."),
    "steps_default": FieldSpec(4000, int, _positive_int, "Default fixed-step count."),
    "output_dir": FieldSpec("results", str, None, "Where benchmark artefacts are written."),
    "allow_scipy": FieldSpec(True, bool, None, "Whether scipy baselines may be used."),
}


@dataclass(frozen=True)
class Config:
    """Validated ODEForge configuration."""

    seed: int = 20260101
    n_probe: int = 24
    hamiltonian_threshold: float = 0.72
    budget_n_fev: int = 120_000
    max_steps: int = 2_000_000
    rtol: float = 1e-8
    atol: float = 1e-10
    tol_grid_power: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)
    noise_std: float = 1e-3
    n_seeds: int = 3
    steps_default: int = 4000
    output_dir: str = "results"
    allow_scipy: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides: Any) -> Config:
        """Build a config from ``ENV_ODEFORGE_*`` variables then explicit overrides."""
        source = dict(os.environ if env is None else env)
        values: dict[str, Any] = {}
        for name, spec in SCHEMA.items():
            raw = source.get(ENV_PREFIX + name.upper())
            if raw is None:
                continue
            values[name] = _coerce(raw, spec, name)
        for name, value in overrides.items():
            if value is None:
                continue
            if name not in SCHEMA:
                msg = f"unknown configuration key {name!r}"
                raise ConfigError(msg, detail={"known": sorted(SCHEMA)})
            values[name] = value
        return cls.build(**values)

    @classmethod
    def build(cls, **values: Any) -> Config:
        """Validate every supplied value against the schema and construct."""
        cleaned: dict[str, Any] = {}
        for name, spec in SCHEMA.items():
            raw = values.get(name, spec.default)
            cleaned[name] = spec.check(raw)
        unknown = set(values) - set(SCHEMA)
        if unknown:
            msg = f"unknown configuration keys: {sorted(unknown)}"
            raise ConfigError(msg, detail={"known": sorted(SCHEMA)})
        return cls(**cleaned)

    def replace(self, **overrides: Any) -> Config:
        """Return a copy with ``overrides`` applied and re-validated."""
        base = {f.name: getattr(self, f.name) for f in fields(self)}
        base.update({k: v for k, v in overrides.items() if v is not None})
        return Config.build(**base)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _coerce(raw: str, spec: FieldSpec, name: str) -> Any:
    """Parse a raw environment string into the declared type."""
    text = raw.strip()
    try:
        if spec.kind is bool:
            low = text.lower()
            if low in {"1", "true", "yes", "on"}:
                return True
            if low in {"0", "false", "no", "off"}:
                return False
            msg = f"cannot parse {raw!r} as bool"
            raise ConfigError(msg, detail={"key": name})
        if spec.kind is int:
            return int(text)
        if spec.kind is float:
            return float(text)
        if spec.kind is tuple:
            parts = [p for p in text.replace(" ", "").split(",") if p]
            return tuple(int(p) for p in parts)
    except ConfigError:
        raise
    except ValueError as exc:
        msg = f"cannot parse {raw!r} as {spec.kind.__name__}"
        raise ConfigError(msg, detail={"key": name, "reason": str(exc)}) from exc
    return text


def load_config(**overrides: Any) -> Config:
    """Convenience wrapper around :meth:`Config.from_env`."""
    return Config.from_env(**overrides)

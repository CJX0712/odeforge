"""Synthetic ODE system family for ODEForge."""

from __future__ import annotations

from data.systems import SYSTEM_NAMES, SYSTEMS, build_system, default_probe_points, poincare_hits

__all__ = ["SYSTEMS", "SYSTEM_NAMES", "build_system", "default_probe_points", "poincare_hits"]

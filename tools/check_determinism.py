"""Assert that two demo runs produce identical numbers, except for wall-clock.

Run after two consecutive ``examples/run_demo.py`` invocations. The report is written
to ``results/benchmark.json`` each time, so the check compares the current file against
the copy taken before the second run.

Author: 晨星 (Chen Xing)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "benchmark.json"
PREVIOUS = ROOT / "results" / "benchmark.prev.json"

#: Keys whose value legitimately differs between runs. Everything else must match
#: bit for bit. Timing is the only genuine source of run-to-run variation, and
#: ``timestamp_utc`` differs by construction.
VOLATILE_KEYS = frozenset({"wall_sec", "wall_sec_total", "demo_wall_sec", "timestamp_utc"})


def strip_volatile(node: Any, key: str | None = None) -> Any:
    """Recursively drop timing fields so the rest can be compared exactly."""
    if key in VOLATILE_KEYS:
        return "<volatile>"
    if isinstance(node, dict):
        return {k: strip_volatile(v, k) for k, v in sorted(node.items())}
    if isinstance(node, list):
        return [strip_volatile(v) for v in node]
    return node


def snapshot() -> None:
    """Copy the current report aside. Call this after the FIRST run."""
    if not REPORT.exists():
        msg = f"{REPORT} not found; run examples/run_demo.py first"
        raise SystemExit(msg)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT, PREVIOUS)
    print(f"snapshot -> {PREVIOUS}")


def compare() -> int:
    """Compare the snapshot against the current report. Call after the SECOND run."""
    if not PREVIOUS.exists():
        msg = f"{PREVIOUS} not found; run tools/check_determinism.py --snapshot first"
        raise SystemExit(msg)
    if not REPORT.exists():
        msg = f"{REPORT} not found; run examples/run_demo.py first"
        raise SystemExit(msg)

    with PREVIOUS.open(encoding="utf-8") as fh:
        before = strip_volatile(json.load(fh))
    with REPORT.open(encoding="utf-8") as fh:
        after = strip_volatile(json.load(fh))

    if before == after:
        print("determinism: OK -- every field identical except wall-clock timings")
        return 0

    diffs: list[str] = []
    _walk(before, after, "", diffs)
    print(f"determinism: FAILED -- {len(diffs)} differing field(s)")
    for line in diffs[:40]:
        print(f"  {line}")
    return 1


def _walk(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in second run")
            elif k not in b:
                out.append(f"{path}.{k}: only in first run")
            else:
                _walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--snapshot" in args:
        snapshot()
        return 0
    return compare()


if __name__ == "__main__":
    raise SystemExit(main())

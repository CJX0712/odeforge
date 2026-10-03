"""Core layer, data family, eval metrics, budget alignment, pipeline and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cli
from core.config import ENV_PREFIX, Config
from core.errors import ERR_CODES, ConfigError, ODEForgeError
from core.seed import get_seed, rng, seed_state, set_all
from data.systems import SYSTEMS, build_system, default_probe_points, poincare_hits
from eval.budget import (
    budget_report,
    solve_dop853_to_budget,
    solve_dp54_to_budget,
    symplectic_budget,
)
from eval.invariants import check_invariants, check_mass_conservation
from eval.metrics import (
    evaluate,
    interpolation_error,
    max_deviation,
    mean_std,
    net_drift,
    relative_drift,
    sign_ratio,
    significant_win,
    trajectory_error,
)
from hpo.search import Candidate, CoordinateDescent, GridSearch, budget_steps_for
from integrators import dp54
from pipeline import ODEForgePipeline

ROOT = Path(__file__).resolve().parents[1]


# =============================================================== core: config
def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.seed == 20260101
    assert cfg.budget_n_fev > 0


def test_config_reads_env_overrides() -> None:
    cfg = Config.from_env(env={f"{ENV_PREFIX}SEED": "123", f"{ENV_PREFIX}RTOL": "1e-6"})
    assert cfg.seed == 123
    assert cfg.rtol == pytest.approx(1e-6)


def test_config_explicit_overrides_beat_env() -> None:
    cfg = Config.from_env(env={f"{ENV_PREFIX}SEED": "123"}, seed=456)
    assert cfg.seed == 456


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError):
        Config.build(nonexistent=1)


def test_config_rejects_bad_type() -> None:
    with pytest.raises(ConfigError):
        Config.build(seed="not-an-int")


def test_config_rejects_non_positive() -> None:
    with pytest.raises(ConfigError):
        Config.build(budget_n_fev=0)


def test_config_rejects_out_of_range_unit_interval() -> None:
    with pytest.raises(ConfigError):
        Config.build(hamiltonian_threshold=2.0)


def test_config_rejects_unparsable_env() -> None:
    with pytest.raises(ConfigError):
        Config.from_env(env={f"{ENV_PREFIX}BUDGET_N_FEV": "many"})


def test_config_replace_is_validated() -> None:
    cfg = Config()
    assert cfg.replace(seed=7).seed == 7
    with pytest.raises(ConfigError):
        cfg.replace(seed=-1)


def test_config_round_trips_to_dict() -> None:
    cfg = Config()
    assert Config.build(**cfg.as_dict()) == cfg


# ================================================================== core: seed
def test_set_all_is_the_single_entry_point() -> None:
    assert set_all(4242) == 4242
    assert get_seed() == 4242
    assert seed_state()["global_seed"] == 4242


def test_set_all_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_all(-1)


def test_rng_is_deterministic_and_offset_sensitive() -> None:
    set_all(99)
    a = rng(offset=3).standard_normal(5)
    b = rng(offset=3).standard_normal(5)
    c = rng(offset=4).standard_normal(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rng_seeds_itself_when_unset() -> None:
    import autodiff.tape as tape_mod  # noqa: PLC0415
    import core.seed as seed_mod  # noqa: PLC0415

    seed_mod._CURRENT_SEED = None
    tape_mod._GLOBAL_TAPE.clear()
    tape_mod._GLOBAL_TAPE.leaves.clear()
    rng()
    assert get_seed() is not None


# =============================================================== core: errors
def test_error_codes_are_unique_and_present() -> None:
    codes = list(ERR_CODES)
    assert len(codes) == len(set(codes))
    for expected in ("E100", "E200", "E300", "E301", "E400", "E401", "E402", "E500", "E501"):
        assert expected in codes


def test_error_string_includes_code_and_detail() -> None:
    err = ODEForgeError("boom", detail={"a": 1})
    text = str(err)
    assert "E000" in text
    assert "boom" in text
    assert "a=1" in text


# ==================================================================== data
def test_family_has_at_least_eight_systems() -> None:
    assert len(SYSTEMS) >= 8


def test_every_system_is_well_formed() -> None:
    for name, spec in SYSTEMS.items():
        assert spec.name == name
        assert spec.rhs(0.0, spec.y0).shape == spec.y0.shape
        assert spec.t_span[1] > spec.t_span[0]
        assert np.all(np.isfinite(spec.y0))
        if spec.is_separable:
            assert spec.grad_v is not None
            assert np.isfinite(float(spec.grad_v(np.asarray(spec.y0[0]))))


def test_build_system_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        build_system("nope")


def test_difficulty_knobs_change_the_system() -> None:
    a = build_system("harmonic")
    b = build_system("harmonic", omega=2.0)
    assert not np.allclose(a.rhs(0.0, a.y0), b.rhs(0.0, b.y0))


def test_harmonic_matches_its_analytic_solution() -> None:
    spec = build_system("harmonic")
    for t in (0.5, 1.0, 2.0):  # t=0 is skipped: dp54 rejects a zero-width span
        traj = dp54(spec.rhs, spec.y0, (0.0, t), rtol=1e-12, atol=1e-14)
        assert np.allclose(traj.y[-1], spec.analytic(t), atol=1e-10)


def test_exponential_matches_its_analytic_solution() -> None:
    spec = build_system("exponential")
    traj = dp54(spec.rhs, spec.y0, (0.0, 1.0), rtol=1e-12, atol=1e-14)
    assert np.allclose(traj.y[-1], spec.analytic(1.0), atol=1e-10)


@pytest.mark.parametrize(
    "name", ["harmonic", "double_well", "pendulum", "kepler", "lotka_volterra", "robertson"]
)
def test_declared_invariants_hold_along_a_short_prefix(name) -> None:
    spec = build_system(name)
    t0, t1 = spec.t_span
    traj = dp54(spec.rhs, spec.y0, (t0, t0 + 0.02 * (t1 - t0)), rtol=1e-9, atol=1e-11)
    quantity = spec.conserved[0]
    series = quantity.series(traj)
    denom = max(1e-12, float(np.max(np.abs(series))))
    assert float(np.max(np.abs(series - series[0]))) / denom < 1e-6


def test_robertson_conserves_mass() -> None:
    spec = build_system("robertson")
    traj = dp54(spec.rhs, spec.y0, (0.0, 1.0), rtol=1e-8, atol=1e-10)
    check = check_mass_conservation(spec, traj, tol=1e-8)
    assert check["passed"]


def test_robertson_span_is_documented_as_truncated() -> None:
    spec = build_system("robertson")
    assert spec.difficulty["textbook_t_end"] == 1e11
    assert spec.t_span[1] < 1e11


def test_check_invariants_returns_nothing_without_a_declared_quantity() -> None:
    assert (
        check_invariants(
            build_system("lorenz"),
            dp54(build_system("lorenz").rhs, build_system("lorenz").y0, (0.0, 1.0)),
        )
        == []
    )


def test_poincare_hits_detects_crossings() -> None:
    t = np.linspace(0, 10, 2001)
    y = np.column_stack([t, np.sin(t), np.zeros_like(t)])
    # Over [0, 10] the sine makes 1.6 periods, so it crosses zero going up once and
    # going down twice. Both directions must be counted, and an unreachable value none.
    assert len(poincare_hits(t, y, 0.0, axis=1, rising=True)) == 1
    assert len(poincare_hits(t, y, 0.0, axis=1, rising=False)) == 2
    assert poincare_hits(t, y, 5.0, axis=1).size == 0


def test_default_probe_points_are_reproducible() -> None:
    a = default_probe_points(6, np.array([1.0, 0.0]))
    b = default_probe_points(6, np.array([1.0, 0.0]))
    assert np.array_equal(a, b)


# ==================================================================== eval
def test_metric_families_on_a_known_trajectory() -> None:
    spec = build_system("harmonic")
    quantity = spec.conserved[0]
    traj = dp54(spec.rhs, spec.y0, spec.t_span, rtol=1e-12, atol=1e-14)
    assert abs(net_drift(traj, quantity)) < 1e-10
    assert max_deviation(traj, quantity) < 1e-10
    assert relative_drift(traj, quantity) < 1e-10
    series = quantity.series(traj)
    assert 0.0 <= sign_ratio(series) <= 1.0


def test_evaluate_bundles_cost_and_metrics() -> None:
    spec = build_system("double_well")
    traj = dp54(spec.rhs, spec.y0, (0.0, 1.0), rtol=1e-8, atol=1e-10)
    bundle = evaluate(traj, spec.conserved[0]).as_dict()
    assert "net_drift" in bundle
    assert "cost" in bundle
    assert bundle["cost"]["n_fev"] == traj.n_fev


def test_trajectory_error_rejects_bad_reference_shape() -> None:
    spec = build_system("harmonic")
    traj = dp54(spec.rhs, spec.y0, (0.0, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        trajectory_error(traj, traj.t, np.zeros((traj.n_points, 3)))


def test_interpolation_error_detects_a_coarse_grid() -> None:
    """A sparse adaptive grid must report a large interpolation error, not hide it."""
    spec = build_system("lorenz")
    coarse = dp54(spec.rhs, spec.y0, (0.0, 2.0), rtol=1e-3, atol=1e-5)
    # A grid too short to decimate is reported as 0.0 by design, so the self-test is
    # skipped below 4*stride+1 points. Measured here: 32 points at rtol=1e-3, and the
    # decimated resample misses the true curve by 3.3e+01 -- the self-test is doing its
    # job and flagging that this grid must not be used as a reference.
    assert interpolation_error(coarse, stride=4) > 1.0


def test_interpolation_error_scales_with_decimation() -> None:
    """The self-test must grow as the grid is decimated, or it detects nothing.

    Measured on the harmonic oscillator at ``rtol=1e-12``: 4.2e-09 at stride 2,
    1.4e-07 at stride 4, 5.6e-05 at stride 16. A self-test that returned a constant
    would be useless as a trust signal.
    """
    spec = build_system("harmonic")
    dense = dp54(spec.rhs, spec.y0, (0.0, 1.0), rtol=1e-12, atol=1e-14)
    e2 = interpolation_error(dense, stride=2)
    e4 = interpolation_error(dense, stride=4)
    e16 = interpolation_error(dense, stride=16)
    assert e2 < e4 < e16
    assert e4 < 1e-5


def test_mean_std_and_significant_win() -> None:
    stats = mean_std([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["n"] == 3
    win = significant_win([1.0, 1.1, 0.9], [5.0, 5.1, 4.9])
    assert win["win"] is True
    assert significant_win([1.0, 1.0], [1.0, 1.0])["win"] is False


# =================================================================== budget
def test_symplectic_budget_hits_the_target_exactly() -> None:
    spec = build_system("double_well")
    res = symplectic_budget("yoshida6", spec, 70_000)
    assert res.achieved_n_fev == pytest.approx(70_000, rel=0.01)
    assert res.knob["n_steps"] == (70_000 - 1) // 7


def test_dp54_budget_alignment() -> None:
    spec = build_system("van_der_pol")
    res = solve_dp54_to_budget(spec, 8000)
    assert res.relative_error < 0.25


def test_dop853_budget_alignment() -> None:
    spec = build_system("van_der_pol")
    res = solve_dop853_to_budget(spec, 8000)
    assert res.relative_error < 0.25


def test_budget_report_exposes_both_counts() -> None:
    spec = build_system("double_well")
    rep = budget_report(
        symplectic_budget("yoshida6", spec, 20_000), solve_dop853_to_budget(spec, 20_000)
    )
    assert rep["target_n_fev"] == 20_000
    assert "flagship_n_fev" in rep
    assert "baseline_n_fev" in rep
    assert "aligned" in rep


# ===================================================================== hpo
def test_grid_search_finds_the_best_candidate() -> None:
    candidates = [Candidate("a", 10), Candidate("b", 20), Candidate("c", 30)]
    result = GridSearch(candidates).search(lambda c: (float(c.n_steps), None))
    assert result.best.n_steps == 10
    assert result.best_score == pytest.approx(10.0)
    assert len(result.trials) == 3


def test_grid_search_supports_bigger_is_better() -> None:
    result = GridSearch([Candidate("a", 1), Candidate("b", 5)]).search(
        lambda c: (float(c.n_steps), None), bigger_is_better=True
    )
    assert result.best.n_steps == 5


def test_grid_search_rejects_an_empty_grid() -> None:
    with pytest.raises(ODEForgeError):
        GridSearch([]).search(lambda c: (0.0, None))


def test_coordinate_descent_runs() -> None:
    """Regression: this path once raised NameError on its first statement."""
    search = CoordinateDescent(["yoshida6", "yoshida4"], budget_n_fev=5000, max_rounds=1)
    result = search.search(lambda c: (float(c.n_steps or 0), None))
    assert result.best.method in {"yoshida6", "yoshida4"}
    assert result.trials


def test_coordinate_descent_rejects_no_methods() -> None:
    with pytest.raises(ODEForgeError):
        CoordinateDescent([])


def test_budget_steps_for_unknown_method() -> None:
    with pytest.raises(ODEForgeError):
        budget_steps_for("nonexistent", 1000)


def test_candidate_key_is_stable() -> None:
    assert Candidate("yoshida6", 100, 1e-8, 1e-10).key() == "yoshida6:n=100:rtol=1e-08:atol=1e-10"


# =============================================================== pipeline
def test_pipeline_runs_a_separable_hamiltonian() -> None:
    traj = ODEForgePipeline().run("double_well")
    assert traj.method == "yoshida6"
    assert np.all(np.isfinite(traj.y))


def test_pipeline_routes_non_hamiltonian_to_dp54() -> None:
    assert ODEForgePipeline().run("van_der_pol").method == "dp54"


def test_pipeline_is_deterministic() -> None:
    a = ODEForgePipeline().run("double_well")
    b = ODEForgePipeline().run("double_well")
    assert np.array_equal(a.y, b.y)
    assert a.n_fev == b.n_fev


def test_pipeline_evaluate_system_reports_metrics_and_invariants() -> None:
    row = ODEForgePipeline().evaluate_system("pendulum")
    assert row["method"] == "yoshida6"
    assert row["invariants"]
    assert row["metrics"]["net_drift"] is not None


def test_pipeline_meta_records_provenance() -> None:
    meta = ODEForgePipeline().meta()
    assert meta["author"].startswith("晨星")
    assert meta["email"] == "CJX0712@users.noreply.github.com"
    assert "numpy" in meta
    assert "scipy_available" in meta


def test_pipeline_search_returns_a_winner() -> None:
    # The budget must clear the double well's stability floor (h <~ 0.43 at T=500), so
    # yoshida6 needs at least ~8400 steps, i.e. a budget of 60000 evaluations.
    out = ODEForgePipeline().search("double_well", budget=60_000)
    assert out["best_method"].startswith("yoshida")
    assert out["n_trials"] == len(out["trials"])
    assert out["best_n_steps"] >= 1


def test_pipeline_search_refuses_a_non_separable_system() -> None:
    with pytest.raises(ValueError, match="not separable"):
        ODEForgePipeline().search("kepler")


def test_pipeline_ablation_report_has_multiple_variants() -> None:
    rep = ODEForgePipeline().ablation_report(system="double_well")
    assert rep["n_variants"] >= 3
    assert "full" in rep["degradation"]


def test_pipeline_invariants_report() -> None:
    rep = ODEForgePipeline().invariants_report()
    assert rep["rows"]


def test_pipeline_write_report_round_trips(tmp_path) -> None:
    pipe = ODEForgePipeline()
    report = {"gates": {"G1": {"passed": True}}, "value": 1.5}
    path = pipe.write_report(report, tmp_path / "r.json")
    with path.open(encoding="utf-8") as fh:
        back = json.load(fh)
    assert back["value"] == 1.5
    assert back["gates"]["G1"]["passed"] is True


def test_pipeline_write_report_rejects_a_broken_round_trip(tmp_path, monkeypatch) -> None:
    pipe = ODEForgePipeline()
    monkeypatch.setattr(json, "dump", lambda *a, **k: None)
    with pytest.raises((ValueError, OSError, TypeError)):
        pipe.write_report({"gates": {"G1": {"passed": True}}}, tmp_path / "x.json")


# ===================================================================== cli
def test_cli_parser_builds() -> None:
    parser = cli.build_parser()
    assert parser.prog == "odeforge"


@pytest.mark.parametrize("argv", [["info"], ["run", "double_well"], ["invariants"]])
def test_cli_subcommands_return_zero(argv) -> None:
    assert cli.main(argv) == 0


def test_cli_rejects_an_unknown_system() -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "not-a-system"])


def test_cli_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_info_payload_has_inventory() -> None:
    payload = cli._cmd_info(Config())
    assert "yoshida6" in payload["methods"]
    assert payload["scipy_available"] in (True, False)


def test_cli_run_payload_shape() -> None:
    import argparse  # noqa: PLC0415

    args = argparse.Namespace(system="double_well", method=None)
    payload = cli._cmd_run(args, Config())
    assert payload["system"] == "double_well"
    assert "route" in payload


def test_cli_module_runs_as_a_script() -> None:
    """Smoke test the documented entry point in a real subprocess."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), "info"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ODEForge" in proc.stdout


def test_secret_scan_finds_nothing() -> None:
    """Self-check: no credential-shaped string may exist in the shipped sources."""
    import re  # noqa: PLC0415

    pattern = re.compile(
        r"(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*[=:]\s*['\"][^'\"]+"
        r"|password\s*[=:]\s*['\"][^'\"]+|secret\s*[=:]\s*['\"][^'\"]+)"
    )
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".md", ".toml", ".yml", ".yaml", ".txt"}:
            continue
        if any(part in {".git", "results", "__pycache__"} for part in path.parts):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

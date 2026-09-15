"""Tests for the report generator, against a fabricated result set.

``build_report.py`` runs once, at the very end of a pipeline that takes hours.
A formatting bug there is discovered at the worst possible moment. These tests
feed it a synthetic ``report_data.json`` plus synthetic result tables — shaped
exactly like the real ones but generated in seconds — so the renderer is
exercised on every code path without waiting for a real run.

The fixture deliberately includes the awkward cases: a model that under-covers
badly, one that over-covers, a MAPE in the billions, NaN metrics, and a dataset
with no panel (so `lgbm_global` is absent). A renderer that only works on
well-behaved input is not much of a renderer.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

import config  # noqa: E402

MODELS = [
    ("seasonal_naive", "baseline"), ("sarima", "classical"),
    ("ets", "classical"), ("prophet", "gam"), ("sktime_theta", "gam"),
    ("lgbm_quantile", "ml"), ("lgbm_conformal", "ml"), ("lgbm_global", "ml"),
]


def _make_tables(table_dir: pathlib.Path, rng) -> None:
    pooled, per_fold, cov_h = [], [], []
    for key, spec in config.DATASETS.items():
        prim = "/".join(spec.primary_group) if spec.primary_group else key
        for wt in ("expanding", "rolling"):
            for i, (mname, fam) in enumerate(MODELS):
                if mname == "lgbm_global" and not spec.is_panel:
                    continue
                # Force the awkward cases rather than hoping rng produces them.
                cov = {0: 1.0, 1: 0.90, 2: 0.31}.get(i, float(rng.uniform(0.3, 1.0)))
                mase = np.nan if mname == "seasonal_naive" and key == "intermittent" \
                    else float(rng.uniform(0.2, 2.0))
                pooled.append(dict(
                    dataset=key, series=prim, model=mname, family=fam,
                    window_type=wt, n_folds_scored=spec.folds.n_folds,
                    n_folds_failed=0,
                    n_points=spec.folds.n_folds * spec.folds.horizon,
                    mae=rng.uniform(1, 50), rmse=rng.uniform(1, 60),
                    wape=rng.uniform(1, 200), mase=mase,
                    mape=rng.uniform(1, 1e9), smape=rng.uniform(1, 160),
                    coverage=cov, coverage_error=cov - 0.9,
                    calibrated=config.is_calibrated(cov),
                    interval_width=rng.uniform(1, 400),
                    width_pct_of_mean=rng.uniform(1, 40),
                    pinball_lower=rng.uniform(0, 5),
                    pinball_upper=rng.uniform(0, 5),
                    pinball_mean=rng.uniform(0, 5),
                    fit_seconds_total=rng.uniform(1, 60),
                    fit_seconds_per_fold=rng.uniform(0.01, 20)))
                for h in range(1, spec.folds.horizon + 1):
                    cov_h.append(dict(
                        dataset=key, series=prim, model=mname, family=fam,
                        window_type=wt, h=h, coverage=rng.uniform(0, 1),
                        interval_width=rng.uniform(1, 400),
                        mae=rng.uniform(1, 50), n=6))
                for f in range(spec.folds.n_folds):
                    per_fold.append(dict(
                        dataset=key, series=prim, window_type=wt, fold=f,
                        model=mname, family=fam,
                        train_size=spec.folds.min_train_size + f,
                        test_start_date="2025-01-01", test_end_date="2025-01-14",
                        horizon=spec.folds.horizon,
                        fit_seconds=rng.uniform(0, 5), error=None,
                        mae=rng.uniform(1, 50), rmse=rng.uniform(1, 60),
                        wape=rng.uniform(1, 200), mase=rng.uniform(0.2, 2),
                        mape=rng.uniform(1, 1e10), smape=rng.uniform(1, 160),
                        coverage=rng.uniform(0, 1),
                        interval_width=rng.uniform(1, 400),
                        all_zero_window=False, actual_sum=100.0,
                        actual_mean=10.0,
                        ljung_box_p=(rng.uniform(0, 1)
                                     if mname in ("sarima", "ets") else np.nan),
                        order="(1, 1, 1)" if mname == "sarima" else np.nan,
                        seasonal_order=("(0, 1, 1, 7)" if mname == "sarima"
                                        else np.nan)))
    pd.DataFrame(pooled).to_csv(table_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(per_fold).to_csv(table_dir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(cov_h).to_csv(table_dir / "coverage_by_horizon.csv", index=False)


def _make_report_data() -> dict:
    data = {
        "environment": {
            "generated_utc": "2026-09-15T20:00:00+00:00", "python": "3.13.0",
            "platform": "Windows", "packages": {
                k: "1.0" for k in ("pandas", "statsmodels", "lightgbm",
                                   "prophet", "sktime")},
            "seed": config.RNG_SEED, "nominal_coverage": 0.9,
            "coverage_tolerance": 0.05},
        "datasets": {}, "figures": {"calibration": "outputs/figures/c.png"},
        "tables": {"pooled_metrics.csv": "outputs/tables/pooled_metrics.csv"},
        "elapsed_seconds": 1234.5,
    }
    for key, spec in config.DATASETS.items():
        data["datasets"][key] = {
            "spec": {
                "key": key, "freq": spec.freq,
                "seasonal_period": spec.seasonal_period,
                "target": spec.target_col, "description": spec.description,
                "transform": spec.transform,
                "scale_free_metric": spec.scale_free_metric,
                "structural_break": spec.structural_break,
                "n_series_in_file": 3, "n": 1000, "start": "2023-01-01",
                "end": "2025-12-31",
                "folds": {
                    "n_folds": spec.folds.n_folds,
                    "horizon": spec.folds.horizon,
                    "min_train_size": spec.folds.min_train_size,
                    "rolling_train_size": spec.folds.rolling_train_size},
                "skip_models": spec.skip_models},
            "diagnostics": {
                "stationarity": [{"name": f"{key} (level)",
                                  "verdict": "unit root",
                                  "interpretation": "ADF/KPSS text"}],
                "decomposition": {"interpretation": "decomposition text"},
                "additive_vs_multiplicative": {"choice": "additive",
                                               "interpretation": "form text"},
                "acf_pacf": {"interpretation": "acf text"},
                "figures": {"overview": "outputs/figures/x.png"}},
            "audits": {
                "leakage": {"prediction_features_invariant": True,
                            "training_features_invariant": True,
                            "n_features": 31, "future_values_perturbed": 300,
                            "n_train_rows": 9000},
                "harness_parity": {
                    wt: {"max_abs_point_difference": 0.0, "identical": True,
                         "n_folds": 6} for wt in ("expanding", "rolling")},
                "break_index": 456, "break_fold": 0},
        }
    data["mape_failure_demo"] = {
        "dataset": "intermittent", "zero_rate": 0.952, "n_folds": 160,
        "mape_median": 2.8e9, "mape_max": 1.2e10, "smape_median": 157.1,
        "wape_median": 126.3, "mae_median": 0.7, "n_all_zero_windows": 38,
        "interpretation": "MAPE failure text"}
    return data


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    """Render the report into a temp dir and hand back its text."""
    table_dir = tmp_path / "outputs" / "tables"
    table_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "TABLE_DIR", table_dir)
    _make_tables(table_dir, np.random.default_rng(config.RNG_SEED))
    (tmp_path / "report_data.json").write_text(
        json.dumps(_make_report_data()), encoding="utf-8")

    import build_report as BR
    monkeypatch.setattr(BR, "ROOT", tmp_path)
    monkeypatch.setattr(BR, "REPORT_PATH", tmp_path / "CAPSTONE_REPORT.md")
    assert BR.main() == 0
    return (tmp_path / "CAPSTONE_REPORT.md").read_text(encoding="utf-8")


def test_report_renders_every_required_section(rendered):
    for heading in (
        "## The short version",
        "## Method",
        "## Does the verdict survive a second window type?",
        "## Calibration: did the 90% interval actually cover 90%?",
        "## Why the scale-free metric is WAPE, not MAPE, on sparse demand",
        "## Decision framework",
        "## Limitations",
        "## Reproducing this",
    ):
        assert heading in rendered, f"missing section: {heading}"
    for key in config.DATASETS:
        assert f"`{key}`" in rendered


def test_report_states_programme_and_cohort(rendered):
    """Scored by the brief: programme name and cohort dates must be present."""
    assert "SDAIA Academy" in rendered
    assert "https://github.com/SDAIAAcademy" in rendered
    assert "2026-09-12" in rendered
    assert "السلاسل الزمنية والتنبؤ" in rendered


def test_report_never_leaves_an_unrendered_placeholder(rendered):
    """No template placeholder or raw Python repr survives into the output.

    Checked against *table cells* rather than the whole document: words like
    "None" appear legitimately in prose ("**None** of this course's four tools
    is the right tool"), and a substring search over the full text would flag
    that sentence rather than a genuine rendering failure.
    """
    for bad in ("<!-- COHORT_DATES -->", "TODO", "FIXME", "{}"):
        assert bad not in rendered, f"unrendered placeholder in report: {bad}"

    cells = []
    for line in rendered.splitlines():
        if line.startswith("|") and line.endswith("|"):
            cells.extend(c.strip() for c in line.strip("|").split("|"))
    for bad in ("None", "nan", "NaN", "nan%", "None%", "inf", "-inf"):
        assert bad not in cells, (
            f"raw {bad!r} rendered into a table cell -- the formatter should "
            f"emit an em dash for a missing value"
        )


def test_report_pairs_coverage_with_width(rendered):
    """The rule the brief names as the most likely point-loser."""
    assert "Width % of mean" in rendered or "width" in rendered.lower()
    assert "Coverage" in rendered
    # The calibration section must name both, in the same section.
    cal = rendered.split("## Calibration")[1].split("## ")[0]
    assert "width" in cal.lower() and "coverage" in cal.lower()


def test_report_marks_over_and_under_coverage_distinctly(rendered):
    """A 100% interval is miscalibrated, not a win -- the report must say so."""
    assert "over-covers" in rendered
    assert "under-covers" in rendered


def test_report_handles_nan_metrics_without_printing_nan(rendered):
    """MASE is genuinely undefined on some series; it must render as a dash."""
    assert "nan" not in rendered.lower().replace("nan-", "")


def test_build_report_tolerates_an_empty_table_file(tmp_path, monkeypatch):
    """A stage that produced no rows must degrade, not crash the build.

    ``pd.read_csv`` raises EmptyDataError on a zero-byte file, which would fail
    the whole report over an optional table.
    """
    table_dir = tmp_path / "outputs" / "tables"
    table_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "TABLE_DIR", table_dir)
    _make_tables(table_dir, np.random.default_rng(0))
    (table_dir / "coverage_by_horizon.csv").write_text("", encoding="utf-8")
    (tmp_path / "report_data.json").write_text(
        json.dumps(_make_report_data()), encoding="utf-8")

    import build_report as BR
    monkeypatch.setattr(BR, "ROOT", tmp_path)
    monkeypatch.setattr(BR, "REPORT_PATH", tmp_path / "CAPSTONE_REPORT.md")
    assert BR.main() == 0
    assert (tmp_path / "CAPSTONE_REPORT.md").exists()


# ==========================================================================
# Notebook integrity
# ==========================================================================

def test_every_notebook_code_cell_parses():
    """The notebook is generated from strings, so a bad escape is invisible.

    `\n` written as `\\n` inside build_notebook's triple-quoted cell sources
    becomes a real newline in the generated cell, which silently produces an
    unterminated f-string. The notebook still writes, still validates against
    the nbformat schema, and only fails when someone runs it. Parsing every
    code cell catches it at build time.
    """
    import ast

    import nbformat

    nb_path = ROOT / "capstone_notebook.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)

    failures = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            failures.append(f"cell {i}: {exc}")
    assert not failures, failures


def test_notebook_covers_every_rubric_requirement():
    """Each graded capability must actually appear in the notebook's code."""
    import nbformat

    nb_path = ROOT / "capstone_notebook.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = nbformat.read(nb_path, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    prose = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    both = code + prose

    required_code = {
        "STL decomposition": "stl_decompose",
        "ACF/PACF": "acf_pacf_report",
        "ADF/KPSS": "stationarity_report",
        "differencing applied": "apply_differencing",
        "differencing motivated": "recommend_differencing",
        "SARIMA fit": "SarimaForecaster",
        "exponential smoothing fit": "EtsForecaster",
        "Ljung-Box": "ljung_box",
        "LightGBM fit": "Lgbm",
        "lag/rolling features": "DirectMultiStepBuilder",
        "leakage audit": "assert_no_leakage",
        "shared harness": "run_walk_forward",
        "both window types": "WINDOW_TYPES",
        "harness parity": "verify_against_course_harness",
        "seasonal-naive baseline": "SeasonalNaive",
        "pinball loss": "pinball",
        "coverage": "coverage",
        "interval width": "interval_width",
    }
    for label, token in required_code.items():
        assert token in code, f"notebook code is missing {label} ({token!r})"

    # All four named tools must be discussed, not just used.
    for tool in ("statsmodels", "Prophet", "sktime", "LightGBM"):
        assert tool in both, f"notebook never mentions {tool}"

    # Attribution items the rubric awards points for.
    assert config.PROGRAMME_PROVIDER in prose
    assert config.SDAIA_GITHUB in prose
    assert config.COURSE_MATERIALS_DATE in prose


def test_every_notebook_code_cell_captured_output():
    """The brief's checklist: *every* cell has real, captured output.

    A cell that renders through `plt.show()` produces nothing under the Agg
    backend `plots.py` pins, so it reads as executed and shows an empty result.
    Catching it here is the difference between "runs" and "demonstrates".
    """
    import nbformat

    nb_path = ROOT / "capstone_notebook.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = nbformat.read(nb_path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    if not any(c.get("outputs") for c in code):
        pytest.skip("notebook built but not yet executed")

    silent = [i for i, c in enumerate(code) if not c.get("outputs")]
    assert not silent, (
        f"code cells with no captured output: {silent} — "
        f"first is: {code[silent[0]].source.strip().splitlines()[0][:80]!r}"
        if silent else "")


def test_notebook_has_no_execution_errors():
    """An executed notebook must contain zero error outputs."""
    import nbformat

    nb_path = ROOT / "capstone_notebook.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = nbformat.read(nb_path, as_version=4)
    errors = [
        (i, o.get("ename"), str(o.get("evalue"))[:200])
        for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    assert not errors, errors

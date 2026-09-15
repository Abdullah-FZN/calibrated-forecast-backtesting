"""End-to-end capstone pipeline: diagnostics -> backtest -> metrics -> artefacts.

Run everything::

    python capstone_pipeline.py

Run one dataset, or skip the slow part while iterating on figures::

    python capstone_pipeline.py --datasets economic
    python capstone_pipeline.py --stages diagnostics,figures

Outputs, all under ``outputs/``:

``tables/per_fold_metrics.csv``
    One row per (dataset, series, window type, model, fold). The rawest useful
    artefact -- every summary in the report is an aggregation of this file, so
    a reader can recompute any headline number from it.
``tables/pooled_metrics.csv``
    One row per (dataset, series, window type, model), scored over every test
    point at once. Coverage is judged here, not on a mean of per-fold coverages.
``tables/*.csv``
    Diagnostics, leakage audit, harness parity, coverage by horizon step, the
    MAPE-failure demonstration, and the environment record.
``figures/*.png``
    The figures the report and notebook reference.
``report_data.json``
    Everything above in one machine-readable file. ``build_report.py`` renders
    CAPSTONE_REPORT.md from it, so the report's numbers cannot drift from the
    run that produced them.

**What this script is, and is not.** The capstone brief asks for one notebook.
This module is the *library* that notebook imports, not a second deliverable
competing with it: keeping the pipeline here means the notebook's cells stay
short enough to read, and means the identical code path produces the committed
tables and the notebook's inline output. Nothing here is a service, a package,
or a second entry point to the analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtesting as B
import config
import dataio
import diagnostics as D
import features as F
import models as M
import plots as P

ALL_STAGES = ("diagnostics", "backtest", "global", "figures", "report")


# ==========================================================================
# Logging
# ==========================================================================

class Tee:
    """Write the console log to a file as well, so a run leaves evidence."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def hr(title: str = "", char: str = "=") -> None:
    if title:
        print(f"\n{char * 78}\n{title}\n{char * 78}")
    else:
        print(char * 78)


# ==========================================================================
# Environment record
# ==========================================================================

def environment_record() -> dict:
    import importlib.metadata as md

    pkgs = ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels",
            "lightgbm", "prophet", "sktime", "matplotlib"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except Exception:
            versions[p] = None
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "seed": config.RNG_SEED,
        "nominal_coverage": config.NOMINAL_COVERAGE,
        "coverage_tolerance": config.COVERAGE_TOLERANCE,
    }


# ==========================================================================
# Stage 1 -- diagnostics
# ==========================================================================

def run_diagnostics(spec, bundle, make_figures: bool) -> dict:
    hr(f"DIAGNOSTICS — {spec.key} / {bundle.label}", "-")
    rep = D.full_report(bundle)

    for s in rep["stationarity"]:
        print(f"  [{s['verdict']:>16}] {s['name']}")
        print(f"      {s['interpretation']}")
    print(f"  DECOMPOSITION: {rep['decomposition']['interpretation']}")
    print(f"  FORM:          {rep['additive_vs_multiplicative']['interpretation']}")
    print(f"  ACF/PACF:      {rep['acf_pacf']['interpretation']}")

    figs = {}
    if make_figures:
        m = spec.seasonal_period
        stl = D.stl_decompose(bundle.values, m)
        ac = D.acf_pacf_report(bundle.values, 36 if spec.freq == "MS" else 60,
                               m, label=bundle.label)
        scored_from = spec.folds.first_test_index(bundle.n)
        figs["overview"] = P.plot_series_overview(
            bundle, f"{spec.key}_overview.png",
            break_date=spec.structural_break, scored_from=scored_from)
        figs["decomposition"] = P.plot_decomposition(
            bundle, stl, f"{spec.key}_decomposition.png")
        figs["acf_pacf"] = P.plot_acf_pacf(
            ac, m, f"{spec.key}_acf_pacf.png", bundle.label)
        if spec.transform == "boxcox":
            tr = bundle.values[: spec.folds.min_train_size]
            tf = dataio.make_transform("boxcox", period=m).fit(tr)
            figs["transform"] = P.plot_transform_effect(
                tr, tf.transform(tr), m, tf.lmbda,
                f"{spec.key}_transform.png", bundle.label)
            rep["transform_fitted_on_first_window"] = tf.params()
        splits_by_type = {
            wt: B.make_splits(bundle.n, spec.folds, wt)
            for wt in B.WINDOW_TYPES
        }
        figs["folds"] = P.plot_fold_layout(
            bundle, splits_by_type, f"{spec.key}_folds.png",
            break_date=spec.structural_break)
    rep["figures"] = figs
    return rep


# ==========================================================================
# Stage 2 -- audits
# ==========================================================================

def run_audits(spec, bundle) -> dict:
    hr(f"AUDITS — {spec.key} / {bundle.label}", "-")
    out: dict = {"series": bundle.label}

    builder = F.DirectMultiStepBuilder(
        lags=spec.lags, windows=spec.rolling_windows,
        horizon=spec.folds.horizon, freq=spec.freq,
        use_holidays=(spec.key == "retail"),
    )
    train_end = spec.folds.min_train_size
    out["leakage"] = F.assert_no_leakage(builder, bundle.values,
                                         bundle.dates, train_end)
    print(f"  leakage audit: perturbed {out['leakage']['future_values_perturbed']} "
          f"future values; prediction features invariant="
          f"{out['leakage']['prediction_features_invariant']}, training "
          f"features invariant={out['leakage']['training_features_invariant']}"
          f" ({out['leakage']['n_features']} features, "
          f"{out['leakage']['n_train_rows']} training rows)")

    out["harness_parity"] = {}
    for wt in B.WINDOW_TYPES:
        r = B.verify_against_course_harness(bundle, wt)
        out["harness_parity"][wt] = r
        print(f"  harness parity ({wt}): identical={r['identical']}, "
              f"max abs point difference={r['max_abs_point_difference']:.1e} "
              f"over {r['n_folds']} folds")

    out["splits"] = {}
    for wt in B.WINDOW_TYPES:
        splits = B.make_splits(bundle.n, spec.folds, wt)
        info = B.assert_splits_sound(splits, bundle.n)
        info["train_sizes"] = [int(x) for x in info["train_sizes"]]
        info["test_spans"] = [[int(a), int(b)] for a, b in info["test_spans"]]
        out["splits"][wt] = info
    if spec.structural_break:
        bi = bundle.index_of(spec.structural_break)
        first = spec.folds.first_test_index(bundle.n)
        fold_of_break = (bi - first) // spec.folds.horizon
        out["break_index"] = bi
        out["break_in_scored_region"] = bool(bi >= first)
        out["break_fold"] = int(fold_of_break) if bi >= first else None
        print(f"  structural break at index {bi} ({spec.structural_break}); "
              f"scored region starts at {first}; break falls in fold "
              f"{out['break_fold']}")
    return out


# ==========================================================================
# Stage 3 -- backtest
# ==========================================================================

#: Scoring helper, defined in backtesting.py where the rest of the
#: metric aggregation lives. Re-exported here only because this module's
#: own stage functions call it.
coverage_by_horizon = B.coverage_by_horizon


def run_backtests(spec, bundles, make_figures: bool) -> dict:
    per_fold_all, pooled_all, cov_h_all = [], [], []
    fan_cache: dict = {}
    importance_cache: dict = {}
    selection_cache: dict = {}

    for bundle in bundles:
        hr(f"BACKTEST — {spec.key} / {bundle.label}", "-")
        for wt in B.WINDOW_TYPES:
            outcomes_by_model = {}
            for fc in M.default_forecasters(spec):
                t0 = time.perf_counter()
                outcomes = B.run_walk_forward(bundle, fc, wt)
                pf = B.score_outcomes(outcomes, spec)
                pf.insert(1, "series", bundle.label)
                pooled = B.pool_outcomes(outcomes, spec)
                pooled["series"] = bundle.label
                per_fold_all.append(pf)
                pooled_all.append(pooled)

                ch = coverage_by_horizon(outcomes)
                if not ch.empty:
                    ch.insert(0, "dataset", spec.key)
                    ch.insert(1, "series", bundle.label)
                    cov_h_all.append(ch)

                n_fail = sum(1 for o in outcomes if o.error)
                mark = "" if not n_fail else f"  [{n_fail} fold(s) failed]"
                print(
                    f"  {wt:9s} {fc.name:16s} "
                    f"MAE={pooled.get('mae', float('nan')):8.3f} "
                    f"WAPE={pooled.get('wape', float('nan')):7.2f}% "
                    f"MASE={pooled.get('mase', float('nan')):6.3f} "
                    f"cov={pooled.get('coverage', float('nan')):5.1%} "
                    f"width={pooled.get('interval_width', float('nan')):8.2f} "
                    f"({time.perf_counter() - t0:5.1f}s){mark}"
                )
                if outcomes and outcomes[0].error:
                    print(f"      first error: {outcomes[0].error[:150]}")
                outcomes_by_model[fc.name] = outcomes

                if fc.name == "lgbm_quantile":
                    for o in outcomes:
                        imp = o.meta.get("feature_importance")
                        if imp:
                            importance_cache[f"{spec.key}|{bundle.label}"] = imp
                            break

                # Capture the model-selection scoreboard from the first fold of
                # the primary series. It is already computed during the search,
                # so recording it costs nothing -- and a reported AIC without
                # the candidates it beat is not evidence that a selection
                # happened.
                if (fc.name in ("sarima", "ets") and wt == "expanding"
                        and bundle.group == (spec.primary_group or ())):
                    for o in outcomes:
                        cands = o.meta.get("candidates")
                        if cands:
                            selection_cache.setdefault(spec.key, {})[fc.name] = {
                                "series": bundle.label,
                                "fold": o.fold,
                                "train_size": o.train_end - o.train_start,
                                "selected_order": o.meta.get("order"),
                                "selected_seasonal_order": o.meta.get(
                                    "seasonal_order"),
                                "selected_config": o.meta.get("ets_config"),
                                "selected_aic": o.meta.get("aic"),
                                "selected_bic": o.meta.get("bic"),
                                "d_from_test": o.meta.get(
                                    "d_from_stationarity_test"),
                                "candidates": cands,
                            }
                            break

            if (make_figures and bundle.group == (spec.primary_group or ())
                    and wt == "expanding"):
                fan_cache[spec.key] = (bundle, outcomes_by_model)

    return {
        "per_fold": per_fold_all,
        "pooled": pooled_all,
        "coverage_by_horizon": cov_h_all,
        "fan_cache": fan_cache,
        "importance": importance_cache,
        "selection": selection_cache,
    }


# ==========================================================================
# Stage 4 -- global (cross-series) LightGBM
# ==========================================================================

def run_global_lgbm(spec, bundles, window_type: str) -> tuple[pd.DataFrame, list]:
    """One LightGBM trained across every series in the panel, per fold.

    day3/06_model_comparison.qmd names this as the untested hypothesis behind
    its own retail result: a *per-series* LightGBM lost to Holt-Winters, and
    the lesson notes that the pooling argument "depends on pooling, which
    single-series LightGBM does not get to use". This runs the pooled version
    so the hypothesis is tested rather than repeated.

    Leakage is handled exactly as in the single-series case: each series'
    features are built from that series' own training slice only, the
    transform is fit per series per fold, and the series id enters as a plain
    categorical column. Nothing crosses the fold boundary; what crosses is the
    *series* boundary, which is the entire point.
    """
    import lightgbm as lgb

    hr(f"GLOBAL LIGHTGBM — {spec.key} ({window_type})", "-")
    n = bundles[0].n
    splits = B.make_splits(n, spec.folds, window_type)
    B.assert_splits_sound(splits, n)
    series_ids = [b.label for b in bundles]

    per_fold_rows, outcomes_per_series = [], {b.label: [] for b in bundles}

    for fold_idx, (tr, te) in enumerate(splits):
        frames, targets = [], []
        transforms, builders = {}, {}
        for b in bundles:
            tf = dataio.make_transform(spec.transform,
                                       period=spec.seasonal_period)
            tf.fit(b.values[tr])
            transforms[b.label] = tf
            builder = F.DirectMultiStepBuilder(
                lags=spec.lags, windows=spec.rolling_windows,
                horizon=spec.folds.horizon, freq=spec.freq,
                use_holidays=(spec.key == "retail"),
            )
            builders[b.label] = builder
            dm = builder.build_train(tf.transform(b.values[tr]), b.dates[tr])
            X = dm.X.copy()
            X["series_id"] = series_ids.index(b.label)
            frames.append(X)
            targets.append(dm.y)

        X_all = pd.concat(frames, ignore_index=True)
        X_all["series_id"] = X_all["series_id"].astype("category")
        y_all = np.concatenate(targets)

        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        model.fit(X_all, y_all, categorical_feature=["series_id"])

        for b in bundles:
            tf, builder = transforms[b.label], builders[b.label]
            pm = builder.build_predict(tf.transform(b.values[tr]),
                                       b.dates[tr], b.dates[te])
            Xp = pm.X.copy()
            Xp["series_id"] = pd.Categorical(
                [series_ids.index(b.label)] * len(Xp),
                categories=range(len(series_ids)))
            point_z = model.predict(Xp) + pm.anchor
            point = np.maximum(tf.inverse_quantile(point_z), 0.0)
            # No native interval from a point objective. Rather than silently
            # skip the probabilistic half for this variant, reuse the same
            # in-fold conformal construction the single-series model uses, on
            # the pooled residuals of this fold's own training window.
            resid = np.abs(model.predict(X_all) - y_all)
            k = int(np.ceil((len(resid) + 1) * config.NOMINAL_COVERAGE))
            margin = float(np.sort(resid)[min(k, len(resid)) - 1])
            lower = np.maximum(tf.inverse_quantile(point_z - margin), 0.0)
            upper = np.maximum(tf.inverse_quantile(point_z + margin), 0.0)

            o = B.FoldOutcome(
                fold=fold_idx, window_type=window_type,
                train_start=tr.start, train_end=tr.stop,
                test_start=te.start, test_end=te.stop,
                dates_test=b.dates[te], y_train=b.values[tr],
                y_true=b.values[te], point=point, lower=lower, upper=upper,
                meta={"model": "lgbm_global", "family": "ml",
                      "fit_seconds": np.nan,
                      "interval": "split conformal on pooled train residuals",
                      "n_series_pooled": len(bundles),
                      "n_train_rows": int(len(y_all))},
            )
            outcomes_per_series[b.label].append(o)
            row = B.score_fold(o, spec)
            row["series"] = b.label
            per_fold_rows.append(row)

    pooled_rows = []
    for label, outs in outcomes_per_series.items():
        p = B.pool_outcomes(outs, spec)
        p["series"] = label
        pooled_rows.append(p)
        print(f"  {window_type:9s} lgbm_global    {label:16s} "
              f"MAE={p.get('mae', float('nan')):8.3f} "
              f"WAPE={p.get('wape', float('nan')):7.2f}% "
              f"MASE={p.get('mase', float('nan')):6.3f} "
              f"cov={p.get('coverage', float('nan')):5.1%}")
    return pd.DataFrame(per_fold_rows), pooled_rows


# ==========================================================================
# Stage 5 -- the MAPE failure demonstration
# ==========================================================================

def mape_failure_demo(spec, bundles, per_fold: pd.DataFrame) -> dict:
    """Show, on real folds, why MAPE is not reportable on a zero-heavy series.

    The metrics cheat sheet makes this argument with a hand-built 10-day
    window. Doing it on this project's own backtest folds turns a quoted
    warning into a measured result.
    """
    sub = per_fold[(per_fold["dataset"] == spec.key)
                   & per_fold["mape"].notna()]
    if sub.empty:
        return {}
    zero_rate = float(np.mean([np.mean(b.values == 0) for b in bundles]))
    return {
        "dataset": spec.key,
        "zero_rate": zero_rate,
        "n_folds": int(len(sub)),
        "mape_median": float(sub["mape"].median()),
        "mape_max": float(sub["mape"].max()),
        "smape_median": float(sub["smape"].median()),
        "wape_median": float(sub["wape"].median()),
        "mae_median": float(sub["mae"].median()),
        "n_all_zero_windows": int(sub["all_zero_window"].sum()),
        "interpretation": (
            f"Across {len(sub)} scored folds on a series that is "
            f"{zero_rate:.1%} zeros, the median fold MAPE is "
            f"{sub['mape'].median():,.0f}% and the worst is "
            f"{sub['mape'].max():,.0f}% -- numbers with no usable "
            f"interpretation, produced by dividing by actuals that are exactly "
            f"zero. sMAPE is bounded at {sub['smape'].median():.1f}% median, "
            f"which fails more politely without being more informative. WAPE's "
            f"median of {sub['wape'].median():.1f}% is the only one of the "
            f"three a planner could act on, because it divides total error by "
            f"total volume once instead of row by row. "
            f"{int(sub['all_zero_window'].sum())} fold(s) had an all-zero test "
            f"window, where even WAPE is only vacuously defined."
        ),
    }


# ==========================================================================
# Orchestration
# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="all",
                    help="comma-separated dataset keys, or 'all'")
    ap.add_argument("--stages", default="all",
                    help=f"comma-separated subset of {ALL_STAGES}, or 'all'")
    ap.add_argument("--all-series", action="store_true",
                    help="backtest every series in a panel dataset, not just "
                         "the primary one (slower; the global-model stage "
                         "always uses all series regardless)")
    args = ap.parse_args(argv)

    keys = (list(config.DATASETS) if args.datasets == "all"
            else [k.strip() for k in args.datasets.split(",")])
    stages = (set(ALL_STAGES) if args.stages == "all"
              else {s.strip() for s in args.stages.split(",")})

    log_path = config.LOG_DIR / "pipeline_run.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    t_start = time.perf_counter()

    report: dict = {"environment": environment_record(), "datasets": {}}
    hr("CAPSTONE FORECASTING PIPELINE")
    print(json.dumps(report["environment"], indent=2))
    print(f"\ndatasets: {keys}\nstages:   {sorted(stages)}")

    per_fold_frames, pooled_rows, cov_h_frames = [], [], []
    fan_caches, importances = {}, {}

    for key in keys:
        spec = config.DATASETS[key]
        all_bundles = dataio.iter_series(spec)
        primary = dataio.get_series(spec)
        bundles = all_bundles if (args.all_series and spec.is_panel) else [primary]

        hr(f"DATASET: {key}")
        print(spec.description)
        print(f"  series in file: {len(all_bundles)}  |  backtested here: "
              f"{[b.label for b in bundles]}")
        print(f"  n={primary.n}  {primary.dates[0].date()} -> "
              f"{primary.dates[-1].date()}  freq={spec.freq}  "
              f"seasonal_period={spec.seasonal_period}")
        print(f"  folds: {spec.folds.n_folds} x horizon {spec.folds.horizon}; "
              f"expanding min_train={spec.folds.min_train_size}, "
              f"rolling train={spec.folds.rolling_train_size}")
        print(f"  transform: {spec.transform}  |  headline scale-free metric: "
              f"{spec.scale_free_metric.upper()}")

        ds_report: dict = {
            "spec": {
                "key": spec.key, "freq": spec.freq,
                "seasonal_period": spec.seasonal_period,
                "target": spec.target_col,
                "description": spec.description,
                "transform": spec.transform,
                "scale_free_metric": spec.scale_free_metric,
                "structural_break": spec.structural_break,
                "n_series_in_file": len(all_bundles),
                "n": primary.n,
                "start": str(primary.dates[0].date()),
                "end": str(primary.dates[-1].date()),
                "folds": {
                    "n_folds": spec.folds.n_folds,
                    "horizon": spec.folds.horizon,
                    "min_train_size": spec.folds.min_train_size,
                    "rolling_train_size": spec.folds.rolling_train_size,
                },
                "skip_models": spec.skip_models,
            },
            "series_summary": [b.describe() for b in all_bundles],
        }

        if "diagnostics" in stages:
            ds_report["diagnostics"] = run_diagnostics(
                spec, primary, "figures" in stages)
            ds_report["audits"] = run_audits(spec, primary)

        if "backtest" in stages:
            res = run_backtests(spec, bundles, "figures" in stages)
            per_fold_frames.extend(res["per_fold"])
            pooled_rows.extend(res["pooled"])
            cov_h_frames.extend(res["coverage_by_horizon"])
            fan_caches.update(res["fan_cache"])
            importances.update(res["importance"])
            ds_report["model_selection"] = res["selection"].get(key, {})

        if "global" in stages and spec.is_panel:
            for wt in B.WINDOW_TYPES:
                gpf, gpool = run_global_lgbm(spec, all_bundles, wt)
                per_fold_frames.append(gpf)
                pooled_rows.extend(gpool)

        report["datasets"][key] = ds_report

    # ---------------------------------------------------------------- tables
    per_fold = (pd.concat(per_fold_frames, ignore_index=True)
                if per_fold_frames else pd.DataFrame())
    pooled = pd.DataFrame(pooled_rows) if pooled_rows else pd.DataFrame()
    cov_h = (pd.concat(cov_h_frames, ignore_index=True)
             if cov_h_frames else pd.DataFrame())

    if not per_fold.empty:
        per_fold.to_csv(config.TABLE_DIR / "per_fold_metrics.csv", index=False)
        B.summarise_per_fold(per_fold).to_csv(
            config.TABLE_DIR / "per_fold_summary.csv", index=False)
    if not pooled.empty:
        pooled.to_csv(config.TABLE_DIR / "pooled_metrics.csv", index=False)
    if not cov_h.empty:
        cov_h.to_csv(config.TABLE_DIR / "coverage_by_horizon.csv", index=False)

    if "intermittent" in keys and not per_fold.empty:
        demo = mape_failure_demo(config.INTERMITTENT,
                                 dataio.iter_series(config.INTERMITTENT),
                                 per_fold)
        if demo:
            report["mape_failure_demo"] = demo
            hr("MAPE FAILURE DEMONSTRATION", "-")
            print("  " + demo["interpretation"])

    # --------------------------------------------------------------- figures
    figures: dict = {}
    if "figures" in stages and not pooled.empty:
        hr("FIGURES", "-")
        primary_pooled = pooled[
            pooled.apply(
                lambda r: r["series"] == (
                    "/".join(config.DATASETS[r["dataset"]].primary_group)
                    if config.DATASETS[r["dataset"]].primary_group
                    else r["dataset"]), axis=1)
        ] if "series" in pooled.columns else pooled
        exp = primary_pooled[primary_pooled["window_type"] == "expanding"]

        if not exp.empty:
            figures["calibration"] = P.plot_calibration(
                exp, "calibration_coverage_vs_width.png",
                f"Calibration at a {config.NOMINAL_COVERAGE:.0%} nominal level "
                f"— expanding window, every dataset")
            figures["comparison_wape"] = P.plot_model_comparison(
                exp, "wape", "comparison_wape.png",
                "Pooled WAPE by model family (lower is better)")
            figures["comparison_mase"] = P.plot_model_comparison(
                exp[exp["mase"].notna()], "mase", "comparison_mase.png",
                "Pooled MASE — below 1.0 beats seasonal-naive", 1.0)
        figures["window_comparison"] = P.plot_window_comparison(
            primary_pooled, "wape", "window_expanding_vs_rolling.png",
            "Expanding vs rolling training window, identical test windows")

        for key, (bundle, obm) in fan_caches.items():
            sel = {k: v for k, v in obm.items()
                   if k in ("seasonal_naive", "sarima", "ets", "prophet",
                            "lgbm_quantile", "lgbm_conformal")}
            fold_idx = 0
            outcomes_fold = {k: v[fold_idx] for k, v in sel.items()
                             if v and v[fold_idx].error is None}
            if outcomes_fold:
                figures[f"fan_{key}"] = P.plot_forecast_fan(
                    outcomes_fold, bundle, fold_idx, f"{key}_fan_fold0.png")

        if not cov_h.empty:
            for key in cov_h["dataset"].unique():
                sub = cov_h[(cov_h["dataset"] == key)
                            & (cov_h["window_type"] == "expanding")]
                if not sub.empty:
                    figures[f"coverage_h_{key}"] = P.plot_coverage_by_horizon(
                        sub, f"{key}_coverage_by_horizon.png",
                        f"Coverage by horizon step — {key} (expanding)")

        if not per_fold.empty:
            for key in per_fold["dataset"].unique():
                spec = config.DATASETS[key]
                sub = per_fold[per_fold["dataset"] == key]
                bf = (report["datasets"].get(key, {})
                      .get("audits", {}).get("break_fold"))
                figures[f"per_fold_{key}"] = P.plot_per_fold_metric(
                    sub, "wape", f"{key}_per_fold_wape.png",
                    f"Per-fold WAPE — {key}", break_fold=bf)

        for tag, imp in importances.items():
            key = tag.split("|")[0]
            figures[f"importance_{key}"] = P.plot_feature_importance(
                imp, f"{key}_feature_importance.png",
                f"LightGBM feature importance — {key}")
        for k, v in figures.items():
            print(f"  {k:28s} -> {v}")
    report["figures"] = figures

    # ----------------------------------------------------------------- write
    pd.DataFrame([report["environment"]["packages"]]).to_csv(
        config.TABLE_DIR / "environment.csv", index=False)
    report["elapsed_seconds"] = round(time.perf_counter() - t_start, 1)
    report["tables"] = {
        p.name: str(p.relative_to(config.ROOT)).replace("\\", "/")
        for p in sorted(config.TABLE_DIR.glob("*.csv"))
    }
    with open(config.ROOT / "report_data.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    hr("DONE")
    print(f"elapsed: {report['elapsed_seconds']}s")
    print(f"tables:  {config.TABLE_DIR}")
    print(f"figures: {config.FIGURE_DIR}")
    print(f"log:     {log_path}")
    sys.stdout = tee.stdout
    tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

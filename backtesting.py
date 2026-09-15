"""Walk-forward backtesting and scoring.

Fold geometry comes from the course's own ``common/backtest.py`` --
``expanding_window_splits`` and ``rolling_window_splits`` are called directly,
unmodified, so the splits this project scores are the splits the course
defines. What this module adds is the part ``run_backtest`` deliberately does
not carry:

* the **dates** for each fold's test window (Prophet and every calendar feature
  need them, and a bare ``np.ndarray`` of values has no index);
* **prediction intervals** alongside the point forecast, since
  ``fit_predict_fn`` is typed to return ``horizon`` point predictions only;
* **per-fold metadata** -- the SARIMA order AIC chose, the Box-Cox lambda, the
  Ljung-Box p-value, fit seconds -- which is most of what the report is
  actually about.

``verify_against_course_harness`` closes the loop: it runs the same forecaster
through the course's ``run_backtest`` and asserts the point forecasts are
identical to this module's. That is what makes "or an equivalent you can
justify" a checked claim rather than an assertion.

**The leakage contract.** For fold *i*, ``_context_for`` builds a
``FoldContext`` from ``y[train_slice]`` and ``dates[train_slice]`` only. Future
*dates* are passed (they are knowable in advance); future *values* are not
passed, and are not reachable from any object the forecaster receives. Each
fold constructs a fresh forecaster state, so nothing fitted on a later fold can
reach an earlier one.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import dataio
import models as M

sys.path.insert(0, str(config.COMMON_DIR))

from backtest import (  # noqa: E402  -- vendored course utility
    expanding_window_splits,
    rolling_window_splits,
    run_backtest,
)
from metrics import (  # noqa: E402
    coverage,
    interval_width,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    smape,
    wape,
)

WINDOW_TYPES = ("expanding", "rolling")


# ==========================================================================
# Splits
# ==========================================================================

def make_splits(n: int, plan: "config.FoldPlan",
                window_type: str) -> list[tuple[slice, slice]]:
    """Fold slices from the course harness, by window type."""
    if window_type == "expanding":
        return expanding_window_splits(
            n=n, n_folds=plan.n_folds, horizon=plan.horizon,
            min_train_size=plan.min_train_size,
        )
    if window_type == "rolling":
        return rolling_window_splits(
            n=n, n_folds=plan.n_folds, horizon=plan.horizon,
            train_size=plan.rolling_train_size,
        )
    raise ValueError(f"unknown window_type {window_type!r}")


def assert_splits_sound(splits: list[tuple[slice, slice]], n: int) -> dict:
    """Check the three properties a walk-forward split must have.

    Cheap, and it turns "no fold's training data overlaps its own test window"
    from a claim in a README into something the pipeline refuses to run
    without.
    """
    for i, (tr, te) in enumerate(splits):
        if tr.stop > te.start:
            raise AssertionError(
                f"fold {i}: train ends at {tr.stop}, test starts at "
                f"{te.start} -- train window overlaps its own test window"
            )
        if te.stop > n:
            raise AssertionError(f"fold {i}: test window runs past the series")
        if tr.start < 0 or tr.stop - tr.start <= 0:
            raise AssertionError(f"fold {i}: empty or negative train window")
    starts = [te.start for _, te in splits]
    if starts != sorted(starts):
        raise AssertionError("test windows are not in chronological order")
    return {
        "n_folds": len(splits),
        "train_sizes": [tr.stop - tr.start for tr, _ in splits],
        "test_spans": [(te.start, te.stop) for _, te in splits],
        "first_scored_index": splits[0][1].start,
        "last_scored_index": splits[-1][1].stop,
    }


# ==========================================================================
# Running
# ==========================================================================

@dataclass
class FoldOutcome:
    """One model, one fold."""

    fold: int
    window_type: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    dates_test: pd.DatetimeIndex
    y_train: np.ndarray
    y_true: np.ndarray
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    meta: dict = field(default_factory=dict)
    error: str | None = None


def _context_for(bundle: "dataio.SeriesBundle", tr: slice, te: slice,
                 level: float) -> "M.FoldContext":
    return M.FoldContext(
        y_train=bundle.values[tr],
        dates_train=bundle.dates[tr],
        dates_future=bundle.dates[te],
        spec=bundle.spec,
        level=level,
    )


def run_walk_forward(bundle: "dataio.SeriesBundle", forecaster: "M.Forecaster",
                     window_type: str, level: float = config.NOMINAL_COVERAGE,
                     verbose: bool = False) -> list[FoldOutcome]:
    """Run one forecaster across every fold of one series.

    A fold that raises is recorded with ``error`` set rather than aborting the
    run -- on the intermittent series some classical fits genuinely fail to
    converge, and "this family could not be fit here" is a result worth
    reporting, not a crash worth hiding.
    """
    plan = bundle.spec.folds
    splits = make_splits(bundle.n, plan, window_type)
    assert_splits_sound(splits, bundle.n)

    outcomes: list[FoldOutcome] = []
    for i, (tr, te) in enumerate(splits):
        ctx = _context_for(bundle, tr, te, level)
        base = dict(
            fold=i, window_type=window_type,
            train_start=tr.start, train_end=tr.stop,
            test_start=te.start, test_end=te.stop,
            dates_test=bundle.dates[te],
            y_train=bundle.values[tr], y_true=bundle.values[te],
        )
        try:
            res = forecaster.fit_predict(ctx)
            outcomes.append(FoldOutcome(
                point=res.point, lower=res.lower, upper=res.upper,
                meta=res.meta, **base,
            ))
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            nan = np.full(te.stop - te.start, np.nan)
            outcomes.append(FoldOutcome(
                point=nan, lower=nan, upper=nan,
                meta={"model": forecaster.name, "family": forecaster.family},
                error=f"{type(exc).__name__}: {exc}", **base,
            ))
            if verbose:
                print(f"  ! {forecaster.name} fold {i} failed: {exc}")
                traceback.print_exc(limit=2)
    return outcomes


def verify_against_course_harness(bundle: "dataio.SeriesBundle",
                                  window_type: str = "expanding") -> dict:
    """Prove this module's fold loop matches ``common/backtest.run_backtest``.

    Runs the seasonal-naive baseline both ways -- through the course's own
    harness and through ``run_walk_forward`` -- and compares point forecasts
    element by element. The baseline is used because it is the one model whose
    forecast depends on nothing but ``y_train``, so any difference would be a
    difference in the *harness*, not in the model.
    """
    plan = bundle.spec.folds
    splits = make_splits(bundle.n, plan, window_type)
    period = bundle.spec.seasonal_period

    def fit_predict(y_train, horizon):
        # Fits from scratch on its argument and closes over nothing mutable --
        # the property day2/04_backtesting.qmd says makes the harness guarantee
        # real rather than nominal.
        from backtest import seasonal_naive_forecast
        return seasonal_naive_forecast(y_train, horizon, period)

    course = run_backtest(bundle.values, splits, fit_predict)
    ours = run_walk_forward(bundle, M.SeasonalNaive(), window_type)

    if len(course) != len(ours):
        raise AssertionError("fold count differs between harnesses")
    max_diff = 0.0
    for c, o in zip(course, ours):
        if not np.array_equal(c["y_true"], o.y_true):
            raise AssertionError(f"fold {c['fold']}: test windows differ")
        if not np.array_equal(c["y_train"], o.y_train):
            raise AssertionError(f"fold {c['fold']}: train windows differ")
        max_diff = max(max_diff,
                       float(np.max(np.abs(c["y_pred"] - o.point))))
    return {
        "series": bundle.label,
        "window_type": window_type,
        "n_folds": len(course),
        "max_abs_point_difference": max_diff,
        "identical": max_diff == 0.0,
    }


# ==========================================================================
# Scoring
# ==========================================================================

def _safe_mase(y_true, y_pred, y_train, sp) -> tuple[float, str | None]:
    """MASE, or NaN plus the documented reason it is undefined.

    ``metrics.mase`` raises on two cases by design -- a training window shorter
    than the seasonal period, and a perfectly repeating training window whose
    naive baseline has zero error. Both are reachable here: the economic
    series' rolling folds are short, and a 97.9%-zero SKU can produce a
    training window whose seasonal differences are all zero. Catching them and
    recording *why* keeps the cell in the results table honest instead of
    either crashing the run or quietly printing a number that is not a MASE.
    """
    try:
        return float(mase(y_true, y_pred, y_train, seasonal_period=sp)), None
    except ValueError as exc:
        return float("nan"), str(exc).split(" -- ")[0].replace("\n", " ")


def score_fold(o: FoldOutcome, spec: "config.DatasetSpec") -> dict:
    """All point and probabilistic metrics for one fold."""
    row: dict = {
        "dataset": spec.key,
        "window_type": o.window_type,
        "fold": o.fold,
        "model": o.meta.get("model"),
        "family": o.meta.get("family"),
        "train_size": o.train_end - o.train_start,
        "test_start_date": o.dates_test[0],
        "test_end_date": o.dates_test[-1],
        "horizon": len(o.y_true),
        "fit_seconds": o.meta.get("fit_seconds", np.nan),
        "error": o.error,
    }
    if o.error is not None or not np.all(np.isfinite(o.point)):
        return row

    yt, yp = o.y_true, o.point
    row["mae"] = mae(yt, yp)
    row["rmse"] = rmse(yt, yp)
    row["wape"] = wape(yt, yp)
    m, why = _safe_mase(yt, yp, o.y_train, spec.seasonal_period)
    row["mase"] = m
    row["mase_undefined_reason"] = why
    # MAPE and sMAPE are computed everywhere but only *reported* where the
    # series has no zeros -- the intermittent dataset uses them as the worked
    # demonstration of why they fail, not as a score.
    row["mape"] = mape(yt, yp)
    row["smape"] = smape(yt, yp)
    row["actual_sum"] = float(np.sum(np.abs(yt)))
    row["actual_mean"] = float(np.mean(yt))
    # A WAPE of 0.0 on an all-zero window is "vacuously right", not accuracy --
    # flagged so the report never quotes one as a win.
    row["all_zero_window"] = bool(np.all(yt == 0))

    row["coverage"] = coverage(yt, o.lower, o.upper)
    row["interval_width"] = interval_width(o.lower, o.upper)
    # Width as a share of the window's own mean level, so a width is comparable
    # between a series averaging 670 units and one averaging 0.4.
    denom = float(np.mean(np.abs(yt)))
    row["width_pct_of_mean"] = (row["interval_width"] / denom * 100
                                if denom > 0 else np.nan)
    row["pinball_lower"] = pinball_loss(yt, o.lower, config.LOWER_Q)
    row["pinball_upper"] = pinball_loss(yt, o.upper, config.UPPER_Q)
    row["pinball_mean"] = (row["pinball_lower"] + row["pinball_upper"]) / 2.0
    row["crossed_bounds_repaired"] = o.meta.get("crossed_bounds_repaired", 0)

    for k in ("order", "seasonal_order", "ets_config", "aic", "bic",
              "ljung_box_p", "residuals_white_noise", "lambda", "interval",
              "n_train_rows", "sp", "margin_h1", "margin_hmax"):
        if k in o.meta:
            row[k] = o.meta[k]
    return row


def score_outcomes(outcomes: list[FoldOutcome],
                   spec: "config.DatasetSpec") -> pd.DataFrame:
    return pd.DataFrame([score_fold(o, spec) for o in outcomes])


def pool_outcomes(outcomes: list[FoldOutcome],
                  spec: "config.DatasetSpec") -> dict:
    """Score every fold's test points **together**, as one sample.

    Two different questions, two different aggregations, and conflating them is
    a real reporting error:

    * the *mean of per-fold metrics* answers "how does this model do on a
      typical fold", and its spread across folds is the variance walk-forward
      validation exists to expose;
    * the *pooled* metric answers "across every point this model was asked to
      forecast, how did it do" -- which is the right basis for **coverage**,
      because a 90% interval is a claim about long-run frequency, and a mean of
      six per-fold coverages each computed over 14 points is not that.

    Both are reported. Coverage is judged on the pooled number.
    """
    ok = [o for o in outcomes if o.error is None and np.all(np.isfinite(o.point))]
    if not ok:
        return {"dataset": spec.key, "n_folds_scored": 0,
                "model": outcomes[0].meta.get("model") if outcomes else None,
                "error": "no fold produced a usable forecast"}

    yt = np.concatenate([o.y_true for o in ok])
    yp = np.concatenate([o.point for o in ok])
    lo = np.concatenate([o.lower for o in ok])
    hi = np.concatenate([o.upper for o in ok])
    # MASE is scaled against a training window, and there are several. Use the
    # longest (the final fold's), which is the most stable estimate of the
    # series' own naive-error scale.
    y_train_ref = max((o.y_train for o in ok), key=len)
    m, why = _safe_mase(yt, yp, y_train_ref, spec.seasonal_period)

    cov = coverage(yt, lo, hi)
    width = interval_width(lo, hi)
    denom = float(np.mean(np.abs(yt)))
    return {
        "dataset": spec.key,
        "model": ok[0].meta.get("model"),
        "family": ok[0].meta.get("family"),
        "window_type": ok[0].window_type,
        "n_folds_scored": len(ok),
        "n_folds_failed": len(outcomes) - len(ok),
        "n_points": int(len(yt)),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "wape": wape(yt, yp),
        "mase": m,
        "mase_undefined_reason": why,
        "mape": mape(yt, yp),
        "smape": smape(yt, yp),
        "coverage": cov,
        "coverage_error": cov - config.NOMINAL_COVERAGE,
        "calibrated": bool(abs(cov - config.NOMINAL_COVERAGE)
                           <= config.COVERAGE_TOLERANCE),
        "interval_width": width,
        "width_pct_of_mean": width / denom * 100 if denom > 0 else np.nan,
        "pinball_lower": pinball_loss(yt, lo, config.LOWER_Q),
        "pinball_upper": pinball_loss(yt, hi, config.UPPER_Q),
        "pinball_mean": (pinball_loss(yt, lo, config.LOWER_Q)
                         + pinball_loss(yt, hi, config.UPPER_Q)) / 2.0,
        "fit_seconds_total": float(np.nansum(
            [o.meta.get("fit_seconds", np.nan) for o in ok])),
        "fit_seconds_per_fold": float(np.nanmean(
            [o.meta.get("fit_seconds", np.nan) for o in ok])),
    }


def summarise_per_fold(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each metric across folds.

    day2/04_backtesting.qmd is explicit that a single mean across folds hides
    exactly what walk-forward validation exists to surface, so the spread is
    reported next to the mean, never instead of it.
    """
    if per_fold.empty:
        return per_fold
    metric_cols = [c for c in ("mae", "rmse", "wape", "mase", "coverage",
                               "interval_width", "pinball_mean", "fit_seconds")
                   if c in per_fold.columns]
    grp = per_fold.groupby(["dataset", "window_type", "model", "family"],
                           dropna=False)
    agg = grp[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_folds"] = grp.size()
    agg["n_failed"] = grp["error"].apply(lambda s: int(s.notna().sum()))
    return agg.reset_index()

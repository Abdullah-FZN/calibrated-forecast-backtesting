"""Forecast accuracy and uncertainty metrics used throughout this course.

One small, tested file rather than a metric reimplemented slightly
differently in every notebook. Every lab notebook fetches this file
verbatim (see `setup.qmd`) instead of copy-pasting it — the same call
signature, the same edge-case handling, every time it's used to grade a
backtest fold.

All functions take numpy arrays or anything array-like and return a float
(or an array of per-row values for `pinball_loss`, which reduces to a mean
too via `.mean()`).
"""

from __future__ import annotations

import numpy as np


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the series' own units."""
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error — penalises large misses harder than MAE."""
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Mean absolute percentage error, as a percentage (0-100+).

    Undefined near y_true == 0 — `epsilon` avoids a division blow-up but
    does not fix the underlying problem. Prefer WAPE or MASE for a series
    with zeros or near-zeros (see the intermittent-demand lab).
    """
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE, as a percentage (0-200 by this common definition).

    Bounded, and defined at y_true == 0 as long as y_pred isn't also 0 —
    still distorts on near-zero actuals, so treat it the same way as MAPE
    on sparse/intermittent series.
    """
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error — total absolute error over total
    actual volume, as a percentage. The metric of choice for demand series
    with zeros: it never divides row-by-row, so a single near-zero actual
    can't dominate the score the way it does in MAPE/sMAPE.
    """
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float(np.sum(np.abs(y_pred)) > 0) * 100.0
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def mase(y_true, y_pred, y_train, seasonal_period: int = 1) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    Scales MAE by the in-sample MAE of a seasonal-naive forecast on the
    TRAINING history (`y_train`), so a value below 1.0 means "beats naive
    seasonal repetition" — comparable across series of very different
    scale, which raw MAE/RMSE are not. `seasonal_period=1` is the ordinary
    (non-seasonal) naive baseline; use the series' actual period (7 for
    daily-with-weekly-seasonality, 12 for monthly-with-yearly-seasonality)
    when scoring a seasonal series.
    """
    y_true, y_pred = _as_float_array(y_true), _as_float_array(y_pred)
    y_train = _as_float_array(y_train)
    if len(y_train) <= seasonal_period:
        raise ValueError(
            "y_train must be longer than seasonal_period to scale against "
            "a naive forecast"
        )
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        raise ValueError(
            "in-sample naive-seasonal MAE is 0 (a perfectly repeating "
            "training series) — MASE is undefined; report MAE instead"
        )
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball_loss(y_true, y_pred_quantile, quantile: float) -> float:
    """Pinball (quantile) loss for one predicted quantile, averaged over
    all points. Lower is better; 0 only for a perfect quantile forecast.
    This is the loss a quantile regressor is trained to minimise, and the
    right accuracy metric for a single prediction-interval bound rather
    than a point forecast.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    y_true, y_pred_quantile = _as_float_array(y_true), _as_float_array(y_pred_quantile)
    diff = y_true - y_pred_quantile
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage: the fraction of actuals that fell inside
    [lower, upper]. Compare against the interval's nominal level (e.g. a
    well-calibrated 80% interval should cover close to 80% of held-out
    points — see reference/metrics_cheatsheet.qmd for how far off is
    "still fine" vs "the interval is miscalibrated").
    """
    y_true, lower, upper = _as_float_array(y_true), _as_float_array(lower), _as_float_array(upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower, upper) -> float:
    """Mean width of a prediction interval — the other half of calibration:
    a wide-enough interval can hit any coverage target while saying
    nothing useful. Report width alongside coverage, never coverage alone.
    """
    lower, upper = _as_float_array(lower), _as_float_array(upper)
    return float(np.mean(upper - lower))

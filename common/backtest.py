"""A small, tested walk-forward backtesting harness — the one every lab from
Day 2 onward plugs a model into, instead of each notebook growing its own
slightly-different fold logic. See day2/04_backtesting.qmd for the theory
this implements.

The core guarantee this module exists to protect: for fold i, every value
in `y_pred` was produced using only `y_train` (which never includes a
single row from that fold's own test window, or any later fold's). No
function here looks past the boundary it's given — that's what makes a
walk-forward split "sound" in the sense objective 4 means it.
"""

from __future__ import annotations

import numpy as np


def expanding_window_splits(
    n: int, n_folds: int, horizon: int, min_train_size: int
) -> list[tuple[slice, slice]]:
    """`n_folds` (train_slice, test_slice) pairs over a length-`n` sequence,
    indexed 0..n-1 in time order.

    Expanding window: each fold's train slice starts at 0 and grows to
    include every earlier fold's test window — train only ever gets more
    history, never less, and the LAST fold's test window ends exactly at
    `n` (today, backtested against the most recent real data available).
    """
    required = min_train_size + n_folds * horizon
    if n < required:
        raise ValueError(
            f"not enough data: need >= {required} rows for {n_folds} folds of "
            f"horizon {horizon} after a {min_train_size}-row minimum train "
            f"window, got {n}"
        )
    splits = []
    for i in range(n_folds):
        train_end = n - (n_folds - i) * horizon
        test_end = train_end + horizon
        splits.append((slice(0, train_end), slice(train_end, test_end)))
    return splits


def rolling_window_splits(
    n: int, n_folds: int, horizon: int, train_size: int
) -> list[tuple[slice, slice]]:
    """Like `expanding_window_splits`, but each fold's train window is a
    FIXED size that slides forward instead of growing — useful when older
    history is believed to be stale (a regime change, a discontinued
    product) rather than merely smaller.
    """
    required = train_size + n_folds * horizon
    if n < required:
        raise ValueError(
            f"not enough data: need >= {required} rows for {n_folds} folds of "
            f"horizon {horizon} with a fixed {train_size}-row train window, "
            f"got {n}"
        )
    splits = []
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * horizon
        train_end = test_end - horizon
        train_start = train_end - train_size
        splits.append((slice(train_start, train_end), slice(train_end, test_end)))
    return splits


def run_backtest(y, splits, fit_predict_fn) -> list[dict]:
    """Run one model across every fold in `splits`.

    `fit_predict_fn(y_train: np.ndarray, horizon: int) -> array-like of
    length `horizon`` is called fresh for every fold — it must fit (or
    otherwise derive) its forecast using ONLY `y_train`, and return exactly
    `horizon` point predictions for the steps immediately following it.

    Returns one dict per fold: `fold`, `y_train`, `y_true`, `y_pred` — hand
    these to `metrics.py`'s functions to score each fold, or concatenate
    `y_true`/`y_pred` across folds for an overall score.
    """
    y = np.asarray(y, dtype=float)
    results = []
    for fold_idx, (train_slice, test_slice) in enumerate(splits):
        y_train = y[train_slice]
        y_true = y[test_slice]
        horizon = test_slice.stop - test_slice.start
        y_pred = np.asarray(fit_predict_fn(y_train, horizon), dtype=float)
        if y_pred.shape[0] != horizon:
            raise ValueError(
                f"fold {fold_idx}: fit_predict_fn returned {y_pred.shape[0]} "
                f"predictions, expected {horizon}"
            )
        results.append(
            {"fold": fold_idx, "y_train": y_train, "y_true": y_true, "y_pred": y_pred}
        )
    return results


def seasonal_naive_forecast(y_train, horizon: int, period: int) -> np.ndarray:
    """Forecast by repeating the last full seasonal cycle of `y_train`. With
    `period=1` this is the ordinary (non-seasonal) naive forecast: repeat
    the last observed value. The standard baseline every model in this
    course is expected to beat — see `metrics.mase`, which scales against
    exactly this forecast.
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < period:
        raise ValueError("y_train shorter than one seasonal period")
    last_cycle = y_train[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last_cycle, reps)[:horizon]

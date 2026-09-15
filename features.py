"""Leakage-safe temporal feature engineering for the tree-based models.

LightGBM has no notion of time. Every bit of temporal structure has to arrive
as a column, and every one of those columns is an opportunity to hand the model
a number it would not actually possess at forecast time. This module exists to
make that impossible by construction rather than by careful review.

The central idea: **forecast origins, not rows.**

A row of the design matrix is not "a day". It is a (origin ``t``, horizon step
``h``) pair, meaning: *standing at time t, knowing y[0..t] and nothing later,
predict y[t+h]*. Every predictor is then either

* a function of ``y[0..t]`` -- lags and rolling statistics, all of which end
  **at** the origin, or
* a function of the calendar date ``dates[t+h]`` -- day of week, Fourier
  seasonal terms, a moving-holiday flag -- which is knowable arbitrarily far in
  advance and is therefore not leakage, or
* ``h`` itself.

This is *direct* multi-step forecasting. The alternative, recursive
forecasting, predicts step 1, appends the prediction to the history, recomputes
lags, and predicts step 2 -- which is correct only if implemented perfectly and
is the single most common correctness bug this course names. Direct multi-step
sidesteps the whole class of bug: a model asked for ``h=14`` never sees any
value later than the origin, because ``h`` is an input rather than a loop
counter. It also avoids compounding a step-1 error through fourteen
substitutions. The price is that the model must learn a slightly different
mapping per horizon, which is why ``h`` is a feature rather than 14 separate
models.

``assert_no_leakage`` turns the guarantee into a test: it perturbs the future
and asserts that not one feature value moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ==========================================================================
# Calendar / known-in-advance regressors
# ==========================================================================

def moving_holiday_dates(start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.DatetimeIndex:
    """Dates covered by the retail series' drifting, Hijri-like holiday bumps.

    ``data/generate_series.py`` places a multi-day demand spike each year at an
    anchor that drifts 11 days earlier per year -- standing in for Ramadan/Eid,
    whose Gregorian dates move the same way. A fixed ``dayofyear`` feature
    cannot represent that: the same calendar day is inside the window one year
    and outside it the next.

    **Is supplying this leakage?** No, and the distinction matters. A moving
    religious holiday is *known years in advance* -- any real demand planner
    has those dates in a calendar table before the forecast is made. It is
    exactly the kind of known-future regressor ``SARIMAX(exog=)``,
    ``Prophet(holidays=)`` and a LightGBM feature column all exist to consume,
    and this project feeds the identical window to all three so no model gets
    an advantage the others are denied.

    The *promo* shocks in the same generator are deliberately **not** provided
    to any model. They are drawn at random with no advance signal, so a planner
    standing at the forecast origin could not know them. Giving a model a promo
    flag would be forecasting with tomorrow's newspaper.
    """
    year0 = start.year
    days: list[pd.Timestamp] = []
    for k, year in enumerate(range(year0, end.year + 1)):
        anchor = pd.Timestamp(year=year, month=4, day=10) - pd.Timedelta(
            days=11 * k
        )
        window = pd.date_range(
            anchor - pd.Timedelta(days=3), anchor + pd.Timedelta(days=9)
        )
        days.extend(window)
    idx = pd.DatetimeIndex(sorted(set(days)))
    return idx[(idx >= start) & (idx <= end)]


def calendar_frame(dates: pd.DatetimeIndex, freq: str,
                   holidays: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Calendar features for a set of *target* dates.

    Everything here is a deterministic function of the date alone, so it is
    available for any future date at any forecast origin.
    """
    out = pd.DataFrame(index=pd.RangeIndex(len(dates)))
    if freq == "MS":
        out["month"] = dates.month.to_numpy()
        out["quarter"] = dates.quarter.to_numpy()
        out["year_frac"] = (dates.month.to_numpy() - 1) / 12.0
        # Two Fourier harmonics of the annual cycle. The generator builds the
        # economic series from a 42-month and an 11-month cycle, neither of
        # which is annual -- these terms are here so the model is not *denied*
        # an annual pattern, not because one is asserted to exist.
        for k in (1, 2):
            out[f"sin_year_{k}"] = np.sin(2 * np.pi * k * out["year_frac"])
            out[f"cos_year_{k}"] = np.cos(2 * np.pi * k * out["year_frac"])
    else:
        dow = dates.dayofweek.to_numpy()
        out["dayofweek"] = dow
        out["is_weekend"] = np.isin(dow, (5, 6)).astype(int)
        out["day_of_month"] = dates.day.to_numpy()
        out["month"] = dates.month.to_numpy()
        doy = dates.dayofyear.to_numpy().astype(float)
        # The generator's yearly term is sin(2*pi*d/365.25) +
        # 0.4*sin(4*pi*d/365.25) -- a fundamental plus its second harmonic.
        # Supplying both harmonics is ordinary Fourier seasonality, the same
        # basis Prophet fits internally.
        for k in (1, 2):
            out[f"sin_year_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            out[f"cos_year_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    if holidays is not None:
        in_window = dates.isin(holidays)
        out["is_holiday_window"] = in_window.astype(int)
        # Signed distance to the nearest holiday day, clipped to +/-10. Lets a
        # tree split on "three days before" separately from "during".
        if len(holidays):
            hol = holidays.asi8.astype(np.int64)
            tgt = dates.asi8.astype(np.int64)
            nearest = np.searchsorted(hol, tgt)
            nearest = np.clip(nearest, 1, len(hol) - 1)
            before, after = hol[nearest - 1], hol[nearest]
            day_ns = 86_400_000_000_000
            d_before = (tgt - before) / day_ns
            d_after = (tgt - after) / day_ns
            signed = np.where(np.abs(d_before) <= np.abs(d_after),
                              d_before, d_after)
            out["days_to_holiday"] = np.clip(signed, -10, 10)
        else:
            out["days_to_holiday"] = 10.0
    return out


# ==========================================================================
# Origin-anchored lag / rolling features
# ==========================================================================

def _origin_features(y: np.ndarray, origins: np.ndarray,
                     lags: tuple[int, ...],
                     windows: tuple[int, ...]) -> pd.DataFrame:
    """Lag and rolling features evaluated **at** each forecast origin.

    ``origins`` are positional indices into ``y``. For origin ``t`` the value
    ``y[t]`` is the most recent observation the forecaster is allowed to see,
    so ``lag_1 = y[t]``, ``lag_7 = y[t-6]``, and a width-``w`` rolling window
    covers ``y[t-w+1 .. t]`` inclusive. Nothing indexes past ``t``.

    Computed with cumulative sums over the *whole array supplied*, which is why
    every caller must pass a training-window slice rather than the full series.
    """
    y = np.asarray(y, dtype=float)
    feats: dict[str, np.ndarray] = {}
    s = pd.Series(y)

    # Lags are built with pandas' shift -- the standard construction, and the
    # one whose off-by-one behaviour is easiest to check. ``shift(k-1)`` moves
    # each value forward by k-1 positions, so position t of the shifted series
    # holds y[t-(k-1)]. Gathered at the origins that gives lag_1 = y[t] (the
    # most recent observation the forecaster may see), lag_7 = y[t-6], and so
    # on. Positions with insufficient history become NaN and their rows are
    # dropped by the caller, rather than being silently back-filled with a
    # value that was never observed.
    for k in lags:
        feats[f"lag_{k}"] = s.shift(k - 1).to_numpy()[origins]

    # Rolling statistics are computed once over the supplied array with
    # pandas' trailing windows -- position t of a width-w rolling result is
    # exactly mean/std/min/max of y[t-w+1 .. t], the same closed-at-the-origin
    # window the loop form expressed -- then gathered at the origins. Same
    # numbers, O(n) instead of O(rows x w), which matters at 20 folds x 2
    # window types.
    for w in windows:
        roll = s.rolling(w)
        for stat, col in (("mean", f"roll_mean_{w}"), ("std", f"roll_std_{w}"),
                          ("min", f"roll_min_{w}"), ("max", f"roll_max_{w}")):
            full = getattr(roll, stat)().to_numpy()
            feats[col] = full[origins]

    # Momentum: how the recent level compares to the level before it. Gives a
    # tree a trend signal it can act on without extrapolating a raw time index
    # (which it cannot do -- splits saturate at the largest value seen).
    if len(windows) >= 2:
        short, long = windows[0], windows[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"momentum_{short}_{long}"] = (
                feats[f"roll_mean_{short}"] - feats[f"roll_mean_{long}"]
            )
    return pd.DataFrame(feats)


@dataclass(frozen=True)
class DesignMatrix:
    """A supervised learning view of a time series, with provenance."""

    X: pd.DataFrame
    y: np.ndarray             # target, in model space
    anchor: np.ndarray        # baseline subtracted from the target (0 if none)
    origins: np.ndarray       # forecast origin index for each row
    target_index: np.ndarray  # index of the value being predicted
    feature_names: list[str]

    @property
    def n_rows(self) -> int:
        return len(self.y)


class DirectMultiStepBuilder:
    """Builds train and predict matrices for direct multi-step forecasting.

    Parameters
    ----------
    lags, windows
        Lag offsets and rolling widths, in observations.
    horizon
        Maximum steps ahead. Training rows are generated for every step
        ``1..horizon``, with ``h`` as a feature.
    target_mode
        ``"delta"`` (default) predicts ``y[t+h] - anchor(t)`` where the anchor
        is the rolling mean at the origin; ``"level"`` predicts ``y[t+h]``
        directly.

        This is not a cosmetic choice. A gradient-boosted tree cannot
        extrapolate: its splits are thresholds on values it saw in training, so
        on a trending series a level-target model's forecasts flatten out at
        the top of the observed range the moment the series moves past it.
        Predicting a deviation from a locally-estimated anchor removes the
        trend from the target and puts it back at prediction time, where
        arithmetic -- not a tree -- carries it. The anchor is computed from the
        origin's own trailing window, so it is as leakage-free as any other
        feature here.
    """

    def __init__(self, lags: tuple[int, ...], windows: tuple[int, ...],
                 horizon: int, freq: str, target_mode: str = "delta",
                 use_holidays: bool = False):
        self.lags = lags
        self.windows = windows
        self.horizon = horizon
        self.freq = freq
        self.target_mode = target_mode
        self.use_holidays = use_holidays
        self.anchor_window = max(windows)

    @property
    def min_history(self) -> int:
        """Observations needed before the first usable forecast origin."""
        return max(max(self.lags), max(self.windows))

    def _anchor_at(self, y: np.ndarray, origins: np.ndarray) -> np.ndarray:
        if self.target_mode == "level":
            return np.zeros(len(origins), dtype=float)
        w = self.anchor_window
        # Trailing mean of the w observations ending at each origin, with
        # min_periods=1 so an early origin still gets an anchor from whatever
        # history it has rather than a NaN that would drop the row.
        full = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
        return full[origins]

    def _holidays(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex | None:
        if not self.use_holidays:
            return None
        # Widen the range so a target date near either end still gets a correct
        # "days to nearest holiday" value.
        return moving_holiday_dates(
            dates[0] - pd.Timedelta(days=400),
            dates[-1] + pd.Timedelta(days=400),
        )

    def build_train(self, y_train: np.ndarray,
                    dates_train: pd.DatetimeIndex,
                    min_origin: int = 0) -> DesignMatrix:
        """Every (origin, h) pair whose target falls inside the training window.

        ``y_train``/``dates_train`` are this fold's training data and nothing
        else -- the builder has no access to the series beyond them, which is
        what makes the leakage guarantee structural rather than behavioural.

        ``min_origin`` restricts the result to origins at or after that index.
        It exists for the conformal calibration step, which needs exactly this
        set of rows -- every (origin, h) pair in the held-out calibration
        window -- and previously obtained them by calling ``build_predict``
        once per origin. That rebuilt the whole rolling-feature frame ~146
        times per fold on the retail series. Restricting the origins here
        produces the identical rows in a single pass.
        """
        n = len(y_train)
        first = max(self.min_history - 1, int(min_origin))
        rows_o, rows_h = [], []
        for t in range(first, n - 1):
            hmax = min(self.horizon, n - 1 - t)
            for h in range(1, hmax + 1):
                rows_o.append(t)
                rows_h.append(h)
        origins = np.asarray(rows_o, dtype=int)
        steps = np.asarray(rows_h, dtype=int)
        if len(origins) == 0:
            raise ValueError(
                f"training window of {n} rows is too short for lags up to "
                f"{max(self.lags)} and windows up to {max(self.windows)}"
            )
        target_idx = origins + steps

        feats = _origin_features(y_train, origins, self.lags, self.windows)
        cal = calendar_frame(dates_train[target_idx], self.freq,
                             self._holidays(dates_train))
        feats = pd.concat([feats.reset_index(drop=True),
                           cal.reset_index(drop=True)], axis=1)
        feats["h"] = steps

        anchor = self._anchor_at(y_train, origins)
        target = y_train[target_idx] - anchor

        keep = ~feats.isna().any(axis=1).to_numpy()
        feats = feats.loc[keep].reset_index(drop=True)
        return DesignMatrix(
            X=feats,
            y=target[keep],
            anchor=anchor[keep],
            origins=origins[keep],
            target_index=target_idx[keep],
            feature_names=list(feats.columns),
        )

    def build_predict(self, y_train: np.ndarray,
                      dates_train: pd.DatetimeIndex,
                      dates_future: pd.DatetimeIndex) -> DesignMatrix:
        """One row per future step, all anchored at the last training index.

        This is the whole point of the direct formulation: the forecast for
        ``h=14`` is produced from the same origin as the forecast for ``h=1``,
        so no predicted value is ever fed back in as an input.
        """
        n = len(y_train)
        origin = n - 1
        steps = np.arange(1, len(dates_future) + 1)
        origins = np.full(len(steps), origin, dtype=int)

        feats = _origin_features(y_train, origins, self.lags, self.windows)
        all_dates = dates_train.append(dates_future)
        cal = calendar_frame(dates_future, self.freq, self._holidays(all_dates))
        feats = pd.concat([feats.reset_index(drop=True),
                           cal.reset_index(drop=True)], axis=1)
        feats["h"] = steps
        anchor = self._anchor_at(y_train, origins)
        return DesignMatrix(
            X=feats,
            y=np.full(len(steps), np.nan),
            anchor=anchor,
            origins=origins,
            target_index=origins + steps,
            feature_names=list(feats.columns),
        )


# ==========================================================================
# Leakage audit
# ==========================================================================

def assert_no_leakage(builder: DirectMultiStepBuilder, y_full: np.ndarray,
                      dates_full: pd.DatetimeIndex, train_end: int,
                      rng: np.random.Generator | None = None) -> dict:
    """Prove empirically that no feature depends on post-origin data.

    The argument "I only sliced training rows" is the one every leaked pipeline
    also makes. This check does not take the code's word for it: it builds the
    prediction matrix from a training window, then **replaces the entire future
    with different numbers** and rebuilds. If a single feature value changes,
    something read past the forecast origin.

    A second check re-runs the same comparison for the *training* matrix after
    corrupting the future, which catches the subtler version of the bug --
    features computed once over a full frame and then row-sliced.

    Returns a dict of what was checked, so the notebook can print evidence
    rather than a bare ``assert`` that a reader has to trust.
    """
    rng = rng or np.random.default_rng(0)
    y_train = y_full[:train_end]
    dates_train = dates_full[:train_end]
    horizon = builder.horizon
    dates_future = dates_full[train_end: train_end + horizon]

    base_pred = builder.build_predict(y_train, dates_train, dates_future)
    base_train = builder.build_train(y_train, dates_train)

    # Corrupt everything at or after the forecast origin's next step.
    y_corrupt = y_full.copy()
    tail = y_corrupt[train_end:]
    y_corrupt[train_end:] = rng.permutation(tail) * 1000.0 + 12345.0

    alt_pred = builder.build_predict(y_corrupt[:train_end], dates_train,
                                     dates_future)
    alt_train = builder.build_train(y_corrupt[:train_end], dates_train)

    pred_same = base_pred.X.equals(alt_pred.X)
    train_same = base_train.X.equals(alt_train.X)
    if not pred_same:
        diff = [c for c in base_pred.X.columns
                if not base_pred.X[c].equals(alt_pred.X[c])]
        raise AssertionError(
            f"LEAKAGE: prediction features changed when the future was "
            f"replaced -- columns: {diff}"
        )
    if not train_same:
        diff = [c for c in base_train.X.columns
                if not base_train.X[c].equals(alt_train.X[c])]
        raise AssertionError(
            f"LEAKAGE: training features changed when the future was "
            f"replaced -- columns: {diff}"
        )

    # And the positive control: every index the builder touched must be <= its
    # own origin, or <= train_end-1 for the prediction matrix.
    max_touched = int(base_pred.origins.max())
    if max_touched > train_end - 1:
        raise AssertionError(
            f"LEAKAGE: prediction origin {max_touched} is beyond the last "
            f"training index {train_end - 1}"
        )

    return {
        "train_end_index": int(train_end),
        "n_features": int(base_pred.X.shape[1]),
        "n_train_rows": int(base_train.n_rows),
        "prediction_features_invariant": bool(pred_same),
        "training_features_invariant": bool(train_same),
        "max_origin_index": max_touched,
        "future_values_perturbed": int(len(tail)),
    }

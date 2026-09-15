"""Forecaster implementations across the three model families.

Every forecaster in this module implements the same two-method contract:

    fit_predict(ctx: FoldContext) -> ForecastResult

``ctx`` carries **only** that fold's training window plus the future *dates*
(which are knowable in advance). There is no path from a forecaster to the test
values, and no forecaster holds state between folds -- each ``fit_predict``
call constructs its own estimator from scratch. That is the property
day2/04_backtesting.qmd says ``run_backtest`` can guarantee structurally only
if the fit function refuses to close over anything, so these classes carry
configuration but never a fitted model.

Every forecaster returns a *point forecast and an interval at the same nominal
level*, so coverage and width are comparable across families without a
footnote. Where a family has no native interval (LightGBM), the interval is
constructed explicitly and the construction is named in the result metadata.

Transform handling is uniform and deliberate:

* models fit in transformed space;
* interval bounds come back through ``inverse_quantile`` (exact, because the
  transform is monotone);
* point forecasts come back through ``inverse_mean`` with the model's own
  in-sample residual variance, correcting the median-vs-mean bias that
  back-transforming otherwise introduces.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import dataio
import features as feat

warnings.filterwarnings("ignore")


# ==========================================================================
# Contract
# ==========================================================================

@dataclass(frozen=True)
class FoldContext:
    """Everything a forecaster is allowed to see for one fold."""

    y_train: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_future: pd.DatetimeIndex
    spec: "config.DatasetSpec"
    level: float = config.NOMINAL_COVERAGE

    @property
    def horizon(self) -> int:
        return len(self.dates_future)

    @property
    def alpha(self) -> float:
        """Two-sided tail mass, e.g. 0.10 for a 90% interval."""
        return 1.0 - self.level


@dataclass
class ForecastResult:
    """Point forecast plus interval, in the series' original units."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    meta: dict = field(default_factory=dict)

    def validate(self, horizon: int, name: str) -> "ForecastResult":
        for arr, label in ((self.point, "point"), (self.lower, "lower"),
                           (self.upper, "upper")):
            if len(arr) != horizon:
                raise ValueError(
                    f"{name}: {label} has {len(arr)} values, expected "
                    f"{horizon}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: {label} contains non-finite values")
        # Quantile crossing: three independently-fit quantile models can emit a
        # 95th percentile below the 5th. Repair it by sorting rather than
        # silently reporting a negative-width interval, and record that the
        # repair fired -- a model that needs it often is telling you something.
        crossed = int(np.sum(self.upper < self.lower))
        if crossed:
            lo = np.minimum(self.lower, self.upper)
            hi = np.maximum(self.lower, self.upper)
            self.lower, self.upper = lo, hi
            self.meta["crossed_bounds_repaired"] = crossed
        self.point = np.clip(self.point, self.lower, self.upper)
        return self


class Forecaster:
    """Base class. Subclasses set ``name``/``family`` and implement
    ``_forecast``; ``fit_predict`` adds timing, validation and non-negativity
    handling so every family is treated identically."""

    name = "base"
    family = "base"
    #: Whether negative predictions should be floored at zero. True for every
    #: series here -- units sold, headcount and order counts are all
    #: non-negative -- but stated per class rather than assumed globally.
    non_negative = True

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        raise NotImplementedError

    def fit_predict(self, ctx: FoldContext) -> ForecastResult:
        t0 = time.perf_counter()
        res = self._forecast(ctx)
        res.meta["fit_seconds"] = time.perf_counter() - t0
        res.meta.setdefault("model", self.name)
        res.meta.setdefault("family", self.family)
        if self.non_negative:
            res.point = np.maximum(res.point, 0.0)
            res.lower = np.maximum(res.lower, 0.0)
            res.upper = np.maximum(res.upper, 0.0)
        return res.validate(ctx.horizon, self.name)

    # -- shared helpers ---------------------------------------------------

    def _make_transform(self, ctx: FoldContext) -> dataio.TargetTransform:
        """A fresh transform, fit on this fold's training window only."""
        return dataio.make_transform(
            ctx.spec.transform, period=ctx.spec.seasonal_period
        ).fit(ctx.y_train)


# ==========================================================================
# Baseline
# ==========================================================================

class SeasonalNaive(Forecaster):
    """Repeat the last full seasonal cycle, with a conformal interval.

    The bar every other model has to clear -- and the same forecast
    ``metrics.mase`` scales against, so a MASE below 1.0 means "beat this".
    Given an interval here too, so the baseline is judged on calibration on the
    same terms as the models rather than being exempted from half the report.
    """

    name = "seasonal_naive"
    family = "baseline"

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        from backtest import seasonal_naive_forecast

        m = ctx.spec.seasonal_period
        y = ctx.y_train
        point = seasonal_naive_forecast(y, ctx.horizon, m)

        # Conformal margin from in-sample seasonal-naive errors, recomputed on
        # this fold's window. Not a theoretical guarantee (the residuals are
        # not exchangeable with future ones) but an honest empirical one.
        resid = np.abs(y[m:] - y[:-m])
        q = float(np.quantile(resid, ctx.level)) if resid.size else 0.0
        return ForecastResult(
            point=point,
            lower=point - q,
            upper=point + q,
            meta={"interval": "conformal(in-sample seasonal-naive residuals)",
                  "margin": q, "period": m},
        )


# ==========================================================================
# Classical: SARIMA and exponential smoothing (statsmodels)
# ==========================================================================

class SarimaForecaster(Forecaster):
    """SARIMA with a per-fold, two-stage AIC search and a Ljung-Box check.

    **Why a search rather than one fixed order.** statsmodels ships no
    ``auto_arima`` (reference/tooling_guide.qmd is explicit that this is a
    pmdarima/sktime feature, outside this course's four tools), so the order
    has to be chosen deliberately. Fixing it by eye from one ACF/PACF plot of
    the whole series would choose it using data from every fold's test window;
    re-running the search inside each fold keeps the choice honest at the cost
    of a few seconds.

    **Why two stages rather than a full grid.** A full 8x4 grid is 32 fits per
    fold, and at roughly a second each that is over half an hour for the
    workforce backtest alone -- spent almost entirely on combinations AIC
    discards immediately. Stage 1 fixes a plain (1,1,1) core and picks the
    seasonal order; stage 2 fixes that seasonal order and picks (p,d,q). This
    is the same stepwise logic auto_arima uses, at 12 fits per fold. It is a
    greedy search and can in principle miss a joint optimum -- stated here
    rather than glossed, since the alternative was not running the search at
    all.
    """

    name = "sarima"
    family = "classical"

    def __init__(self, orders=None, seasonal_orders=None, maxiter: int = 50):
        self.orders = orders or config.SARIMA_ORDERS
        self.seasonal_orders = seasonal_orders or config.SARIMA_SEASONAL_ORDERS
        self.maxiter = maxiter

    @staticmethod
    def _fit_one(z, order, seasonal_order, maxiter):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            z, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=maxiter)

    def _resolve_seasonal(self, so, m):
        P, D, Q, s = so
        return (P, D, Q, m if s is None else s)

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        m = ctx.spec.seasonal_period
        tf = self._make_transform(ctx)
        z = tf.transform(ctx.y_train)

        candidates = [self._resolve_seasonal(so, m)
                      for so in self.seasonal_orders]
        # Stage 1 -- seasonal order, with the core order held at (1,1,1).
        best_so, best_aic = candidates[0], np.inf
        for so in candidates:
            try:
                r = self._fit_one(z, (1, 1, 1), so, self.maxiter)
                if np.isfinite(r.aic) and r.aic < best_aic:
                    best_so, best_aic = so, float(r.aic)
            except Exception:
                continue

        # Stage 2 -- (p,d,q), with the seasonal order held at stage 1's winner.
        best_fit, best_order, best_aic = None, None, np.inf
        for order in self.orders:
            try:
                r = self._fit_one(z, order, best_so, self.maxiter)
                if np.isfinite(r.aic) and r.aic < best_aic:
                    best_fit, best_order, best_aic = r, order, float(r.aic)
            except Exception:
                continue
        if best_fit is None:
            raise RuntimeError(
                f"SARIMA: every candidate failed to fit on a "
                f"{len(z)}-observation window"
            )

        fc = best_fit.get_forecast(steps=ctx.horizon)
        mean_z = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=ctx.alpha), dtype=float)
        lo_z, hi_z = ci[:, 0], ci[:, 1]

        resid = np.asarray(best_fit.resid, dtype=float)
        # Drop the burn-in the differencing leaves behind before measuring
        # residual variance or testing for autocorrelation.
        burn = max(best_order[1] + best_so[1] * m, 1)
        resid_clean = resid[burn:]
        resid_var = float(np.var(resid_clean, ddof=1))

        lb_lags = min(ctx.spec.ljung_box_lags, max(1, len(resid_clean) // 5))
        try:
            lb = acorr_ljungbox(resid_clean, lags=[lb_lags], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat, lb_p = float("nan"), float("nan")

        return ForecastResult(
            point=tf.inverse_mean(mean_z, resid_var),
            lower=tf.inverse_quantile(lo_z),
            upper=tf.inverse_quantile(hi_z),
            meta={
                "order": best_order,
                "seasonal_order": best_so,
                "aic": float(best_fit.aic),
                "bic": float(best_fit.bic),
                "ljung_box_lags": lb_lags,
                "ljung_box_stat": lb_stat,
                "ljung_box_p": lb_p,
                # H0 is "residuals are white noise". p > alpha means we fail to
                # reject it, i.e. no autocorrelation left that the test can see.
                "residuals_white_noise": bool(lb_p > config.ALPHA)
                if np.isfinite(lb_p) else None,
                "converged": bool(best_fit.mle_retvals.get("converged", False))
                if hasattr(best_fit, "mle_retvals") else None,
                "interval": "analytic (state-space)",
                "resid_var": resid_var,
                **tf.params(),
            },
        )


class EtsForecaster(Forecaster):
    """Exponential smoothing (SES / Holt / Holt-Winters), AIC-selected.

    Uses statsmodels' ``ETSModel`` rather than the older ``ExponentialSmoothing``
    class for one reason that matters to this project: ETSModel is a proper
    state-space formulation and therefore exposes ``get_prediction``, i.e.
    *analytic prediction intervals*. ``ExponentialSmoothing`` gives point
    forecasts only, and an interval would have to be simulated -- which is
    fine, but makes this family's interval incomparable to SARIMA's.

    Configurations are scored by AIC on the same training window, so the
    choice between "no trend", "damped trend" and "trend + seasonality" is made
    by the same rule that picks the SARIMA order, not by taste.
    """

    name = "ets"
    family = "classical"

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        from statsmodels.stats.diagnostic import acorr_ljungbox

        m = ctx.spec.seasonal_period
        tf = self._make_transform(ctx)
        z = tf.transform(ctx.y_train)

        best, best_cfg, best_aic = None, None, np.inf
        for cfg in config.ETS_CONFIGS:
            needs_season = cfg["seasonal"] is not None
            # Holt-Winters needs at least two full cycles to estimate a
            # seasonal profile; skipping rather than letting it fail keeps the
            # per-fold log readable.
            if needs_season and len(z) < 2 * m + 1:
                continue
            try:
                model = ETSModel(
                    pd.Series(z), error="add", trend=cfg["trend"],
                    seasonal=cfg["seasonal"], damped_trend=cfg["damped_trend"],
                    seasonal_periods=m if needs_season else None,
                )
                r = model.fit(disp=False)
                if np.isfinite(r.aic) and r.aic < best_aic:
                    best, best_cfg, best_aic = r, cfg, float(r.aic)
            except Exception:
                continue
        if best is None:
            raise RuntimeError("ETS: no configuration fit successfully")

        pred = best.get_prediction(
            start=len(z), end=len(z) + ctx.horizon - 1
        )
        summary = pred.summary_frame(alpha=ctx.alpha)
        mean_z = summary["mean"].to_numpy(dtype=float)
        lo_z = summary["pi_lower"].to_numpy(dtype=float)
        hi_z = summary["pi_upper"].to_numpy(dtype=float)

        resid = np.asarray(best.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        resid_var = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
        lb_lags = min(ctx.spec.ljung_box_lags, max(1, len(resid) // 5))
        try:
            lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat, lb_p = float("nan"), float("nan")

        return ForecastResult(
            point=tf.inverse_mean(mean_z, resid_var),
            lower=tf.inverse_quantile(lo_z),
            upper=tf.inverse_quantile(hi_z),
            meta={
                "ets_config": best_cfg["name"],
                "trend": best_cfg["trend"],
                "seasonal": best_cfg["seasonal"],
                "damped": best_cfg["damped_trend"],
                "aic": float(best.aic),
                "bic": float(best.bic),
                "ljung_box_lags": lb_lags,
                "ljung_box_stat": lb_stat,
                "ljung_box_p": lb_p,
                "residuals_white_noise": bool(lb_p > config.ALPHA)
                if np.isfinite(lb_p) else None,
                "interval": "analytic (state-space)",
                "resid_var": resid_var,
                **tf.params(),
            },
        )


# ==========================================================================
# GAM / framework family: Prophet and sktime
# ==========================================================================

class ProphetForecaster(Forecaster):
    """Prophet, with the moving-holiday window supplied as a holidays frame.

    Prophet is fit on the **untransformed** series. Its additive decomposition
    plus multiplicative seasonality mode already handles level-dependent
    seasonal amplitude directly, so wrapping it in a Box-Cox as well would be
    correcting the same thing twice, and its native interval would then need a
    back-transform whose bias correction Prophet's own uncertainty model does
    not account for.
    """

    name = "prophet"
    family = "gam"

    def __init__(self, seasonality_mode: str | None = None,
                 use_holidays: bool = False):
        self.seasonality_mode = seasonality_mode
        self.use_holidays = use_holidays

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        import logging

        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
        logging.getLogger("prophet").setLevel(logging.CRITICAL)

        daily = ctx.spec.freq != "MS"
        mode = self.seasonality_mode or (
            "multiplicative" if ctx.spec.transform == "boxcox" else "additive"
        )

        holidays = None
        if self.use_holidays and daily:
            span = feat.moving_holiday_dates(
                ctx.dates_train[0] - pd.Timedelta(days=400),
                ctx.dates_future[-1] + pd.Timedelta(days=400),
            )
            holidays = pd.DataFrame(
                {"holiday": "moving_holiday", "ds": span,
                 "lower_window": 0, "upper_window": 0}
            )

        m = Prophet(
            interval_width=ctx.level,
            weekly_seasonality=daily,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=mode,
            holidays=holidays,
        )
        df = pd.DataFrame({"ds": ctx.dates_train, "y": ctx.y_train})
        m.fit(df)
        future = pd.DataFrame({"ds": ctx.dates_future})
        fc = m.predict(future)

        return ForecastResult(
            point=fc["yhat"].to_numpy(dtype=float),
            lower=fc["yhat_lower"].to_numpy(dtype=float),
            upper=fc["yhat_upper"].to_numpy(dtype=float),
            meta={
                "seasonality_mode": mode,
                "holidays_supplied": holidays is not None,
                "n_changepoints": int(len(m.changepoints)),
                "interval": "native (Prophet MAP trend + observation noise)",
                "transform": "none",
            },
        )


class SktimeThetaForecaster(Forecaster):
    """sktime's Theta forecaster, via the uniform ``predict_interval`` API.

    Included to exercise the point reference/tooling_guide.qmd makes about
    sktime: the interval call is identical regardless of what estimator is
    underneath, so the scoring code below never learns which family it is
    scoring. Theta itself is a strong, cheap benchmark (it won the M3
    competition) and is deseasonalised internally, which makes it a fair
    classical-adjacent comparison rather than a fourth ARIMA.
    """

    name = "sktime_theta"
    family = "gam"

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        from sktime.forecasting.theta import ThetaForecaster

        m = ctx.spec.seasonal_period
        tf = self._make_transform(ctx)
        z = tf.transform(ctx.y_train)
        # A plain integer index rather than a PeriodIndex: sktime accepts both,
        # but building the PeriodIndex costs ~0.85s per fold here and buys
        # nothing, since `sp` is passed explicitly and the forecast horizon is
        # relative. At 40 folds x 4 datasets that is most of an hour of pure
        # index construction.
        y = pd.Series(z, index=pd.RangeIndex(len(z)))
        fh = np.arange(1, ctx.horizon + 1)
        # Theta's seasonality test needs at least two cycles.
        sp = m if len(z) >= 2 * m else 1
        f = ThetaForecaster(sp=sp)
        f.fit(y)
        point_z = np.asarray(f.predict(fh), dtype=float)
        pi = f.predict_interval(fh, coverage=ctx.level)
        lo_z = np.asarray(pi.iloc[:, 0], dtype=float)
        hi_z = np.asarray(pi.iloc[:, 1], dtype=float)

        resid_var = float(np.var(np.diff(z), ddof=1)) if len(z) > 2 else 0.0
        return ForecastResult(
            point=tf.inverse_mean(point_z, None),
            lower=tf.inverse_quantile(lo_z),
            upper=tf.inverse_quantile(hi_z),
            meta={"sp": sp, "interval": "sktime predict_interval",
                  "resid_var": resid_var, **tf.params()},
        )


# ==========================================================================
# Tree-based ML: LightGBM
# ==========================================================================

class _LgbmBase(Forecaster):
    """Shared feature plumbing for the LightGBM variants."""

    family = "ml"

    def __init__(self, use_holidays: bool = False,
                 target_mode: str = "delta", params: dict | None = None):
        self.use_holidays = use_holidays
        self.target_mode = target_mode
        self.params = {**config.LGBM_PARAMS, **(params or {})}

    def _builder(self, ctx: FoldContext) -> feat.DirectMultiStepBuilder:
        return feat.DirectMultiStepBuilder(
            lags=ctx.spec.lags,
            windows=ctx.spec.rolling_windows,
            horizon=ctx.horizon,
            freq=ctx.spec.freq,
            target_mode=self.target_mode,
            use_holidays=self.use_holidays and ctx.spec.key == "retail",
        )


class LgbmQuantileForecaster(_LgbmBase):
    """LightGBM with a quantile objective -- three models, one interval.

    ``objective="quantile", alpha=q`` trains against pinball loss at ``q``, so
    the 0.05 and 0.95 models *are* the interval bounds by construction and the
    0.50 model is the median point forecast. Three independent fits means
    nothing enforces ``lower <= upper``; ``ForecastResult.validate`` checks for
    the crossing and records it when it fires.

    Note what is *not* here: no scaler. Trees split on thresholds, so feature
    scaling is a no-op for them -- which conveniently removes the most common
    leakage vector (a scaler fit outside the fold loop) from this family
    entirely.
    """

    name = "lgbm_quantile"

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        import lightgbm as lgb

        tf = self._make_transform(ctx)
        z = tf.transform(ctx.y_train)
        builder = self._builder(ctx)
        train = builder.build_train(z, ctx.dates_train)
        pred = builder.build_predict(z, ctx.dates_train, ctx.dates_future)

        out = {}
        for tag, q in (("lower", config.LOWER_Q), ("median", 0.5),
                       ("upper", config.UPPER_Q)):
            model = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                      **self.params)
            model.fit(train.X, train.y)
            out[tag] = model.predict(pred.X) + pred.anchor

        importance = None
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(train.feature_names,
                                  [int(v) for v in model.feature_importances_]))

        return ForecastResult(
            point=tf.inverse_quantile(out["median"]),
            lower=tf.inverse_quantile(out["lower"]),
            upper=tf.inverse_quantile(out["upper"]),
            meta={
                "interval": f"quantile objective (alpha="
                            f"{config.LOWER_Q}/{config.UPPER_Q})",
                "n_train_rows": train.n_rows,
                "n_features": len(train.feature_names),
                "target_mode": self.target_mode,
                "holidays_supplied": builder.use_holidays,
                # The median model's forecast is a quantile, so it needs
                # inverse_quantile, not inverse_mean -- no bias correction is
                # appropriate for a median.
                "point_is": "median (q=0.50)",
                "feature_importance": importance,
                **tf.params(),
            },
        )


class LgbmConformalForecaster(_LgbmBase):
    """LightGBM point model wrapped in a split-conformal interval.

    The calibration window is the **last** ``CONFORMAL_CALIB_FRACTION`` of this
    fold's training data, held out from fitting. The margin is the ``level``
    quantile of absolute calibration residuals, recomputed inside every fold --
    which day3/05_probabilistic_forecasting.qmd argues for explicitly: a margin
    computed once over the whole series would be calibrated on a regime that
    may no longer hold by the time it is used, and would be at its narrowest
    exactly when the workforce series' break makes it least trustworthy.

    The margin is computed **per horizon step**, not pooled. Error at h=14 is
    genuinely larger than at h=1, and a single pooled margin produces an
    interval that is too wide early and too narrow late while looking correctly
    calibrated on average.
    """

    name = "lgbm_conformal"

    def _forecast(self, ctx: FoldContext) -> ForecastResult:
        import lightgbm as lgb

        tf = self._make_transform(ctx)
        z_full = tf.transform(ctx.y_train)
        n = len(z_full)
        n_calib = max(ctx.horizon * 2,
                      int(round(n * config.CONFORMAL_CALIB_FRACTION)))
        n_calib = min(n_calib, n // 3)
        split = n - n_calib

        builder = self._builder(ctx)
        # Proper-training portion only -- the calibration window must be data
        # the point model has never seen, or its residuals understate real error.
        inner_train = builder.build_train(z_full[:split],
                                          ctx.dates_train[:split])
        model = lgb.LGBMRegressor(**self.params)
        model.fit(inner_train.X, inner_train.y)

        # Score the calibration window the same way the test window will be
        # scored: one forecast of `horizon` steps from each origin in it.
        resid_by_step: list[list[float]] = [[] for _ in range(ctx.horizon)]
        origins = range(split, n - 1)
        for t in origins:
            steps = min(ctx.horizon, n - 1 - t)
            if steps <= 0:
                continue
            pm = builder.build_predict(
                z_full[: t + 1], ctx.dates_train[: t + 1],
                ctx.dates_train[t + 1: t + 1 + steps],
            )
            yhat = model.predict(pm.X) + pm.anchor
            actual = z_full[t + 1: t + 1 + steps]
            for hstep, (a, p) in enumerate(zip(actual, yhat)):
                resid_by_step[hstep].append(abs(float(a - p)))

        margins = np.empty(ctx.horizon, dtype=float)
        for hstep in range(ctx.horizon):
            r = resid_by_step[hstep]
            if r:
                # Finite-sample conformal quantile: the ceil((n+1)(1-a))/n-th
                # order statistic, not the plain empirical quantile. On a
                # calibration window of a few dozen points the difference is
                # the interval hitting nominal coverage rather than sitting
                # just under it.
                k = int(np.ceil((len(r) + 1) * ctx.level))
                k = min(k, len(r))
                margins[hstep] = float(np.sort(r)[k - 1])
            else:
                margins[hstep] = float(np.max(margins[:hstep])
                                       if hstep else 0.0)

        # Refit on the full training window now the margin is fixed -- the
        # margin came from a model that never saw the calibration data, and the
        # deployed model should use everything available.
        full_train = builder.build_train(z_full, ctx.dates_train)
        final = lgb.LGBMRegressor(**self.params)
        final.fit(full_train.X, full_train.y)
        pm = builder.build_predict(z_full, ctx.dates_train, ctx.dates_future)
        point_z = final.predict(pm.X) + pm.anchor

        return ForecastResult(
            point=tf.inverse_quantile(point_z),
            lower=tf.inverse_quantile(point_z - margins),
            upper=tf.inverse_quantile(point_z + margins),
            meta={
                "interval": "split conformal (per-horizon, refit each fold)",
                "n_calib_obs": n_calib,
                "n_calib_residuals": int(sum(len(r) for r in resid_by_step)),
                "margin_h1": float(margins[0]),
                "margin_hmax": float(margins[-1]),
                "n_train_rows": full_train.n_rows,
                "target_mode": self.target_mode,
                "holidays_supplied": builder.use_holidays,
                **tf.params(),
            },
        )


# ==========================================================================
# Registry
# ==========================================================================

def default_forecasters(spec: "config.DatasetSpec") -> list[Forecaster]:
    """The model line-up for one dataset.

    Same families everywhere so the cross-dataset table is comparable; the
    per-dataset variation is confined to whether the moving-holiday regressor
    is supplied, which only the retail generator actually contains.
    """
    holidays = spec.key == "retail"
    return [
        SeasonalNaive(),
        SarimaForecaster(),
        EtsForecaster(),
        ProphetForecaster(use_holidays=holidays),
        SktimeThetaForecaster(),
        LgbmQuantileForecaster(use_holidays=holidays),
        LgbmConformalForecaster(use_holidays=holidays),
    ]

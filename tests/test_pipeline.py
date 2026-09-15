"""Correctness tests for the capstone pipeline.

These test the properties that would silently produce *better-looking* results
if they broke -- which is the only kind of bug a forecasting backtest cannot
catch by eye. A leaked feature or an overlapping fold does not raise; it just
quietly reports a lower error than the model will ever earn in production.

Run with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

import backtesting as B  # noqa: E402
import config  # noqa: E402
import dataio  # noqa: E402
import diagnostics as D  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402


ALL_SPECS = list(config.DATASETS.values())
SPEC_IDS = [s.key for s in ALL_SPECS]


# ==========================================================================
# Ingestion
# ==========================================================================

@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_every_series_is_regular_and_complete(spec):
    """No gaps, no duplicates, no NaN, at the declared frequency."""
    for bundle in dataio.iter_series(spec):
        bundle.validate()  # raises on any violation
        assert bundle.n > 0
        assert bundle.dates.is_monotonic_increasing


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_fold_plan_fits_the_data(spec):
    n = dataio.get_series(spec).n
    spec.folds.validate(n, spec.key)
    assert spec.folds.n_folds >= 3, "the brief requires at least 3 folds"


def test_workforce_break_lands_inside_the_scored_region():
    """The whole point of the workforce fold geometry.

    If this fails, the backtest still runs and still reports plausible numbers
    -- it just never tests the structural break, which is the one thing that
    dataset exists to test.
    """
    spec = config.WORKFORCE
    b = dataio.get_series(spec)
    break_idx = b.index_of(spec.structural_break)
    first_scored = spec.folds.first_test_index(b.n)
    assert break_idx >= first_scored, (
        f"break at index {break_idx} is before the first scored index "
        f"{first_scored} -- the backtest would never see it"
    )


# ==========================================================================
# Fold soundness
# ==========================================================================

@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("window_type", B.WINDOW_TYPES)
def test_no_fold_trains_on_its_own_test_window(spec, window_type):
    n = dataio.get_series(spec).n
    splits = B.make_splits(n, spec.folds, window_type)
    B.assert_splits_sound(splits, n)
    for i, (tr, te) in enumerate(splits):
        train_idx = set(range(tr.start, tr.stop))
        test_idx = set(range(te.start, te.stop))
        assert not (train_idx & test_idx), f"fold {i} overlaps"
        assert max(train_idx) < min(test_idx), f"fold {i} trains on the future"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_both_window_types_score_identical_test_windows(spec):
    """Expanding vs rolling must be a controlled experiment.

    If the two window types scored different test windows, any difference
    between them would be confounded with which days they happened to be
    graded on.
    """
    n = dataio.get_series(spec).n
    exp = B.make_splits(n, spec.folds, "expanding")
    roll = B.make_splits(n, spec.folds, "rolling")
    assert [(t.start, t.stop) for _, t in exp] == \
           [(t.start, t.stop) for _, t in roll]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_rolling_train_windows_are_fixed_size(spec):
    n = dataio.get_series(spec).n
    sizes = {tr.stop - tr.start
             for tr, _ in B.make_splits(n, spec.folds, "rolling")}
    assert sizes == {spec.folds.rolling_train_size}


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_expanding_train_windows_only_grow(spec):
    n = dataio.get_series(spec).n
    sizes = [tr.stop - tr.start
             for tr, _ in B.make_splits(n, spec.folds, "expanding")]
    assert sizes == sorted(sizes)
    assert all(tr.start == 0 for tr, _ in B.make_splits(n, spec.folds,
                                                        "expanding"))


# ==========================================================================
# Leakage
# ==========================================================================

@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_features_never_read_past_the_forecast_origin(spec):
    """The headline correctness guarantee, tested rather than asserted."""
    b = dataio.get_series(spec)
    builder = F.DirectMultiStepBuilder(
        lags=spec.lags, windows=spec.rolling_windows,
        horizon=spec.folds.horizon, freq=spec.freq,
        use_holidays=(spec.key == "retail"),
    )
    audit = F.assert_no_leakage(builder, b.values, b.dates,
                                spec.folds.min_train_size)
    assert audit["prediction_features_invariant"]
    assert audit["training_features_invariant"]
    assert audit["max_origin_index"] <= spec.folds.min_train_size - 1


def test_lag_and_rolling_features_match_hand_computed_slices():
    """Guards the vectorised rewrite against an off-by-one.

    The rolling statistics were moved from an explicit per-row loop to pandas
    trailing windows for speed. That is exactly the kind of change that can
    shift a window by one position and leak a single future observation, so
    the values are checked against slices computed by hand.
    """
    spec = config.RETAIL
    b = dataio.get_series(spec)
    y = b.values[:400]
    builder = F.DirectMultiStepBuilder(
        lags=(1, 7, 28), windows=(7, 28), horizon=5, freq="D")
    dm = builder.build_train(y, b.dates[:400])
    for row in (0, 17, len(dm.y) - 1):
        t = int(dm.origins[row])
        assert dm.X["lag_1"].iloc[row] == pytest.approx(y[t])
        assert dm.X["lag_7"].iloc[row] == pytest.approx(y[t - 6])
        assert dm.X["lag_28"].iloc[row] == pytest.approx(y[t - 27])
        assert dm.X["roll_mean_7"].iloc[row] == pytest.approx(y[t - 6:t + 1].mean())
        assert dm.X["roll_max_28"].iloc[row] == pytest.approx(y[t - 27:t + 1].max())
        # And the target really is h steps after the origin.
        h = int(dm.X["h"].iloc[row])
        assert dm.target_index[row] == t + h


def test_prediction_matrix_is_anchored_at_the_last_training_index():
    """Direct multi-step: every horizon step forecast from the same origin."""
    spec = config.WORKFORCE
    b = dataio.get_series(spec)
    builder = F.DirectMultiStepBuilder(
        lags=spec.lags, windows=spec.rolling_windows, horizon=14, freq="D")
    n_train = 400
    pm = builder.build_predict(b.values[:n_train], b.dates[:n_train],
                               b.dates[n_train:n_train + 14])
    assert set(pm.origins.tolist()) == {n_train - 1}
    assert pm.X["h"].tolist() == list(range(1, 15))
    # Every row's lag features are identical -- they all read the same origin.
    assert pm.X["lag_1"].nunique() == 1
    assert pm.X["lag_1"].iloc[0] == pytest.approx(b.values[n_train - 1])


def test_moving_holiday_window_drifts_backwards_each_year():
    """A fixed day-of-year feature could not represent this."""
    days = F.moving_holiday_dates(pd.Timestamp("2023-01-01"),
                                  pd.Timestamp("2025-12-31"))
    firsts = {}
    for d in days:
        firsts.setdefault(d.year, d)
    years = sorted(firsts)
    assert len(years) >= 3
    for a, b_ in zip(years, years[1:]):
        delta = (firsts[b_].replace(year=firsts[a].year) - firsts[a]).days
        assert delta < 0, "holiday window should move earlier each year"


# ==========================================================================
# Transforms
# ==========================================================================

def test_transform_refuses_to_run_before_fit():
    tf = dataio.make_transform("boxcox", period=7)
    with pytest.raises(RuntimeError, match="before fit"):
        tf.transform(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("kind", ["none", "log1p", "boxcox"])
def test_transform_round_trips(kind):
    rng = np.random.default_rng(config.RNG_SEED)
    y = rng.gamma(shape=9.0, scale=40.0, size=400)
    tf = dataio.make_transform(kind, period=7).fit(y)
    back = tf.inverse_quantile(tf.transform(y))
    assert np.allclose(back, y, rtol=1e-6, atol=1e-6)


def test_guerrero_lambda_stabilises_level_spread_dependence():
    """The claim the transform exists to make, measured."""
    spec = config.RETAIL
    b = dataio.get_series(spec)
    y = b.values[:spec.folds.min_train_size]
    tf = dataio.make_transform("boxcox", period=7).fit(y)
    z = tf.transform(y)

    def level_spread_corr(v):
        w = v[len(v) % 7:].reshape(-1, 7)
        return abs(float(np.corrcoef(w.mean(1), w.std(1, ddof=1))[0, 1]))

    assert level_spread_corr(y) > 0.5, "raw series should show the dependence"
    assert level_spread_corr(z) < 0.2, "transform should remove most of it"


def test_boxcox_inverse_mean_is_at_or_above_inverse_median():
    """The retransformation bias correction must push up, not down.

    Back-transforming a conditional mean gives a conditional median; the
    correction restores the mean, which on a right-skewed series is higher.
    """
    rng = np.random.default_rng(config.RNG_SEED)
    y = rng.gamma(shape=6.0, scale=50.0, size=500)
    tf = dataio.make_transform("boxcox", period=7).fit(y)
    z = tf.transform(y)[:20]
    med = tf.inverse_quantile(z)
    mean = tf.inverse_mean(z, resid_var=float(np.var(tf.transform(y), ddof=1)))
    assert np.all(mean >= med - 1e-9)


# ==========================================================================
# Harness parity
# ==========================================================================

@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("window_type", B.WINDOW_TYPES)
def test_runner_matches_the_course_harness(spec, window_type):
    b = dataio.get_series(spec)
    r = B.verify_against_course_harness(b, window_type)
    assert r["identical"], (
        f"{spec.key}/{window_type}: point forecasts differ from "
        f"common/backtest.run_backtest by up to "
        f"{r['max_abs_point_difference']}"
    )


# ==========================================================================
# Forecaster contract
# ==========================================================================

@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_forecasters_return_well_formed_intervals(spec):
    b = dataio.get_series(spec)
    plan = spec.folds
    tr = slice(0, plan.min_train_size)
    te = slice(plan.min_train_size, plan.min_train_size + plan.horizon)
    ctx = M.FoldContext(y_train=b.values[tr], dates_train=b.dates[tr],
                        dates_future=b.dates[te], spec=spec)
    for fc in M.default_forecasters(spec):
        res = fc.fit_predict(ctx)
        assert len(res.point) == plan.horizon
        assert np.all(np.isfinite(res.point))
        assert np.all(res.lower <= res.upper), f"{fc.name}: crossed bounds"
        assert np.all(res.point >= res.lower - 1e-9)
        assert np.all(res.point <= res.upper + 1e-9)
        if fc.non_negative:
            assert np.all(res.lower >= -1e-9), f"{fc.name}: negative lower bound"


def test_forecasters_cannot_see_the_test_window():
    """A forecaster handed a corrupted future must produce the same forecast.

    This is the model-level counterpart to the feature-level leakage audit: it
    catches a forecaster that reached around its FoldContext to the original
    series, which the feature test cannot see.
    """
    spec = config.WORKFORCE
    b = dataio.get_series(spec)
    plan = spec.folds
    tr = slice(0, plan.min_train_size)
    te = slice(plan.min_train_size, plan.min_train_size + plan.horizon)

    corrupted = b.values.copy()
    corrupted[plan.min_train_size:] = 99999.0
    ctx_a = M.FoldContext(y_train=b.values[tr], dates_train=b.dates[tr],
                          dates_future=b.dates[te], spec=spec)
    ctx_b = M.FoldContext(y_train=corrupted[tr], dates_train=b.dates[tr],
                          dates_future=b.dates[te], spec=spec)
    for fc in (M.SeasonalNaive(), M.LgbmQuantileForecaster(),
               M.LgbmConformalForecaster()):
        a = fc.fit_predict(ctx_a)
        c = fc.fit_predict(ctx_b)
        assert np.allclose(a.point, c.point), (
            f"{fc.name} changed its forecast when only the (unseen) future "
            f"changed -- it is reading past its training window"
        )


# ==========================================================================
# Metric policy
# ==========================================================================

def test_mase_failure_modes_are_caught_not_raised():
    """metrics.mase raises by design in two cases; scoring must not crash."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    flat = np.full(30, 5.0)
    val, why = B._safe_mase(y_true, y_pred, flat, 7)
    assert np.isnan(val) and why, "flat training window should yield NaN + reason"

    short = np.array([1.0, 2.0, 3.0])
    val2, why2 = B._safe_mase(y_true, y_pred, short, 7)
    assert np.isnan(val2) and why2


def test_wape_all_zero_window_is_flagged_not_celebrated():
    """A 0.0 WAPE on an all-zero window is vacuous, and must be labelled."""
    o = B.FoldOutcome(
        fold=0, window_type="expanding", train_start=0, train_end=10,
        test_start=10, test_end=13,
        dates_test=pd.date_range("2025-01-01", periods=3),
        y_train=np.array([0.0, 1.0] * 5), y_true=np.zeros(3),
        point=np.zeros(3), lower=np.zeros(3), upper=np.zeros(3),
        meta={"model": "t", "family": "baseline"},
    )
    row = B.score_fold(o, config.INTERMITTENT)
    assert row["all_zero_window"] is True
    assert row["wape"] == 0.0  # vacuously


def test_intermittent_uses_wape_not_mape_as_headline():
    assert config.INTERMITTENT.scale_free_metric == "wape"
    for spec in (config.RETAIL, config.WORKFORCE, config.ECONOMIC):
        assert spec.scale_free_metric == "mase"


def test_coverage_tolerance_is_applied_symmetrically():
    rows = [
        {"coverage": config.NOMINAL_COVERAGE, "expect": True},
        {"coverage": config.NOMINAL_COVERAGE + config.COVERAGE_TOLERANCE,
         "expect": True},
        {"coverage": 1.0, "expect": False},   # over-covering is miscalibrated
        {"coverage": 0.5, "expect": False},
    ]
    for r in rows:
        assert config.is_calibrated(r["coverage"]) is r["expect"], r


# ==========================================================================
# Diagnostics
# ==========================================================================

def test_ljung_box_reading_is_the_right_way_round():
    """A large p-value must read as 'white noise', not the reverse."""
    rng = np.random.default_rng(config.RNG_SEED)
    white = rng.normal(size=500)
    r = D.ljung_box_report(white, lags=10)
    assert r["passes_white_noise"] is True
    assert "fail to reject" in r["interpretation"]

    # A strongly autocorrelated series must fail.
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    r2 = D.ljung_box_report(ar, lags=10)
    assert r2["passes_white_noise"] is False
    assert "rejects the white-noise null" in r2["interpretation"]


def test_adf_kpss_agree_on_a_constructed_random_walk_and_white_noise():
    """The verdict logic, tested across draws rather than on one lucky sample.

    Both tests have a real type-I error rate, and on a single 400-point white
    noise draw KPSS does sometimes reject stationarity -- one draw from this
    project's own seed produces a KPSS statistic of 0.82, which rejects at 1%
    and lands the verdict on "trend-stationary". That is the test behaving as
    designed, not the code misbehaving, so asserting on one draw would make
    this a coin flip. Twenty draws and a large-majority threshold test the
    decision table without being hostage to sampling noise.
    """
    n_draws = 20
    rw_verdicts, wn_verdicts = [], []
    for seed in range(n_draws):
        rng = np.random.default_rng(config.RNG_SEED + seed)
        rw_verdicts.append(
            D.stationarity_report(np.cumsum(rng.normal(size=400)), "rw").verdict)
        wn_verdicts.append(
            D.stationarity_report(rng.normal(size=400), "wn").verdict)

    # Measured on this seed range: 18/20 and 17/20 respectively. The misses are
    # the tests' own type-I errors (ADF rejects a unit root on ~1 random walk
    # in 20 at alpha=0.05, which is exactly its advertised rate), so the
    # threshold is a large majority rather than unanimity -- demanding
    # unanimity would be demanding that two statistical tests never err.
    assert sum(v == "unit root" for v in rw_verdicts) >= 0.8 * n_draws, \
        f"random walks should read as unit root: {rw_verdicts}"
    assert sum(v == "stationary" for v in wn_verdicts) >= 0.8 * n_draws, \
        f"white noise should read as stationary: {wn_verdicts}"


def test_decomposition_form_choice_handles_zeros():
    """A multiplicative decomposition of a zero-containing series is undefined."""
    b = dataio.get_series(config.INTERMITTENT)
    res = D.additive_vs_multiplicative(b.values, 7)
    assert res["choice"] == "additive"
    assert res["multiplicative"]["usable"] is False


def test_conformal_calibration_rows_match_per_origin_construction():
    """The batched calibration set must equal the per-origin one, exactly.

    `LgbmConformalForecaster` originally scored its calibration window by
    calling `build_predict` once per origin. That is O(origins) rebuilds of the
    whole rolling-feature frame -- ~145 per fold on the retail series. The
    batched form asks `build_train` for the same (origin, h) pairs in one pass.

    It is a pure speedup only if the rows are identical, so that is asserted
    here rather than assumed: same count per horizon step, and the same
    residuals to floating-point equality.
    """
    import lightgbm as lgb

    spec = config.RETAIL
    b = dataio.get_series(spec)
    plan = spec.folds
    ctx = M.FoldContext(
        y_train=b.values[:plan.min_train_size],
        dates_train=b.dates[:plan.min_train_size],
        dates_future=b.dates[plan.min_train_size:plan.min_train_size + plan.horizon],
        spec=spec,
    )
    fc = M.LgbmConformalForecaster(use_holidays=True)
    tf = fc._make_transform(ctx)
    z = tf.transform(ctx.y_train)
    n = len(z)
    n_calib = min(max(ctx.horizon * 2,
                      int(round(n * config.CONFORMAL_CALIB_FRACTION))), n // 3)
    split = n - n_calib

    builder = fc._builder(ctx)
    inner = builder.build_train(z[:split], ctx.dates_train[:split])
    model = lgb.LGBMRegressor(**fc.params)
    model.fit(inner.X, inner.y)

    per_origin = [[] for _ in range(ctx.horizon)]
    for t in range(split, n - 1):
        steps = min(ctx.horizon, n - 1 - t)
        if steps <= 0:
            continue
        pm = builder.build_predict(z[:t + 1], ctx.dates_train[:t + 1],
                                   ctx.dates_train[t + 1:t + 1 + steps])
        yhat = model.predict(pm.X) + pm.anchor
        for hstep, (a, p) in enumerate(zip(z[t + 1:t + 1 + steps], yhat)):
            per_origin[hstep].append(abs(float(a - p)))

    calib = builder.build_train(z, ctx.dates_train, min_origin=split)
    yhat = model.predict(calib.X) + calib.anchor
    resid = np.abs(z[calib.target_index] - yhat)
    steps_arr = calib.X["h"].to_numpy()
    batched = [resid[steps_arr == h] for h in range(1, ctx.horizon + 1)]

    for h, (a, c) in enumerate(zip(per_origin, batched), start=1):
        assert len(a) == len(c), f"h={h}: {len(a)} vs {len(c)} residuals"
        assert np.allclose(np.sort(a), np.sort(c), atol=0, rtol=0), \
            f"h={h}: residuals differ between constructions"


def test_min_origin_is_a_pure_restriction_of_build_train():
    """build_train(min_origin=k) must be the k-suffix of build_train(), not a
    differently-computed frame."""
    spec = config.WORKFORCE
    b = dataio.get_series(spec)
    y, d = b.values[:400], b.dates[:400]
    builder = F.DirectMultiStepBuilder(
        lags=spec.lags, windows=spec.rolling_windows, horizon=7, freq="D")
    full = builder.build_train(y, d)
    cut = 300
    part = builder.build_train(y, d, min_origin=cut)

    expected = full.X[full.origins >= cut].reset_index(drop=True)
    assert len(part.X) == len(expected)
    pd.testing.assert_frame_equal(part.X, expected)
    assert np.array_equal(part.origins, full.origins[full.origins >= cut])
    assert np.allclose(part.y, full.y[full.origins >= cut])


# ==========================================================================
# Rubric-critical guarantees
# ==========================================================================

def test_the_course_harness_actually_drives_the_fold_loop():
    """`run_backtest` must run the loop, not merely be available.

    The distinction the brief draws is between *using* the shared harness and
    reimplementing it beside the shared splits. Asserted by instrumenting the
    course's own `run_backtest` and checking it was entered, and that the model
    was called once per fold through it.
    """
    import backtest as course_backtest

    spec = config.ECONOMIC
    b = dataio.get_series(spec)
    calls = {"n": 0}
    real = course_backtest.run_backtest

    def counting(y, splits, fit_predict_fn):
        calls["n"] += 1
        return real(y, splits, fit_predict_fn)

    B.run_backtest = counting
    try:
        outcomes = B.run_walk_forward(b, M.SeasonalNaive(), "expanding")
    finally:
        B.run_backtest = real

    assert calls["n"] == 1, "run_backtest was not the thing driving the loop"
    assert len(outcomes) == spec.folds.n_folds


def test_each_fold_gets_a_freshly_fitted_model():
    """No fitted state may survive from one fold to the next."""
    spec = config.ECONOMIC
    b = dataio.get_series(spec)
    fits = []

    class Counting(M.SeasonalNaive):
        def _forecast(self, ctx):
            fits.append(len(ctx.y_train))
            return super()._forecast(ctx)

    fc = Counting()
    outcomes = B.run_walk_forward(b, fc, "expanding")
    # One fit per fold, and each on a strictly larger expanding window.
    assert len(fits) == len(outcomes) == spec.folds.n_folds
    assert fits == sorted(fits) and len(set(fits)) == len(fits)
    # The forecaster object must carry no fitted model between folds.
    assert not [a for a in vars(fc) if "model" in a.lower() or "fit" in a.lower()]


def test_fold_loop_rejects_a_mismatched_training_window():
    """The integrity assertion inside fit_predict_fn must actually fire.

    If it never fires, it is decoration. The guard exists to catch a harness
    that hands over a window other than this fold's own training slice, so the
    test substitutes exactly that: a stand-in `run_backtest` that calls the
    model with the wrong slice. The run must abort rather than quietly score a
    forecast built from the wrong history.

    (Mutating the bundle in place would *not* exercise this: `run_backtest`
    holds the same array object, so both sides of the comparison move together
    and the guard correctly stays silent.)
    """
    spec = config.ECONOMIC
    b = dataio.get_series(spec)
    real = B.run_backtest

    def wrong_window(y, splits, fit_predict_fn):
        tr, te = splits[0]
        fit_predict_fn(y[: max(1, (tr.stop - tr.start) // 2)],
                       te.stop - te.start)
        return []

    B.run_backtest = wrong_window
    try:
        with pytest.raises(AssertionError, match="not this fold's slice"):
            B.run_walk_forward(b, M.SeasonalNaive(), "expanding")
    finally:
        B.run_backtest = real


def test_fold_loop_rejects_a_mismatched_horizon():
    """The second half of the same guard."""
    spec = config.ECONOMIC
    b = dataio.get_series(spec)
    real = B.run_backtest

    def wrong_horizon(y, splits, fit_predict_fn):
        tr, te = splits[0]
        fit_predict_fn(y[tr], (te.stop - te.start) + 1)
        return []

    B.run_backtest = wrong_horizon
    try:
        with pytest.raises(AssertionError, match="horizon"):
            B.run_walk_forward(b, M.SeasonalNaive(), "expanding")
    finally:
        B.run_backtest = real


def test_lag_features_are_shift_based():
    """The brief asks specifically for shift-based lag construction."""
    import inspect
    src = inspect.getsource(F._origin_features)
    assert ".shift(" in src, "lag features must be built with pandas shift"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_differencing_is_driven_by_the_stationarity_test(spec):
    """`d` must follow from ADF/KPSS, not be an unmotivated default."""
    b = dataio.get_series(spec)
    plan = D.recommend_differencing(b.values, spec.seasonal_period, name=spec.key)

    assert 0 <= plan.d <= 2
    assert plan.seasonal_D in (0, 1)
    assert plan.rationale and plan.evidence
    level = plan.evidence[0]
    if level.verdict == "stationary":
        assert plan.d == 0, "a stationary series must not be differenced"
        assert "already stationary" in plan.rationale
    else:
        assert plan.d >= 1, "a non-stationary series must be differenced"
        assert f"so d={plan.d}" in plan.rationale
    # The rationale must cite the actual p-value it acted on.
    assert f"{level.adf_p:.4f}" in plan.rationale or plan.d == 0


def test_sarima_takes_its_d_from_the_test_not_from_aic():
    """AIC picks p and q; the stationarity test picks d. Never the reverse."""
    spec = config.WORKFORCE
    b = dataio.get_series(spec)
    plan_fold = spec.folds
    tr = slice(0, plan_fold.min_train_size)
    te = slice(plan_fold.min_train_size,
               plan_fold.min_train_size + plan_fold.horizon)
    ctx = M.FoldContext(y_train=b.values[tr], dates_train=b.dates[tr],
                        dates_future=b.dates[te], spec=spec)
    res = M.SarimaForecaster().fit_predict(ctx)

    d_used = res.meta["order"][1]
    assert d_used == res.meta["d_from_stationarity_test"]
    assert res.meta["seasonal_order"][1] == res.meta["D_from_seasonal_strength"]
    assert res.meta["differencing_rationale"]


def test_sktime_interval_and_quantile_apis_agree():
    """Both APIs are called, and an 80/90% interval IS its quantile pair."""
    spec = config.ECONOMIC
    b = dataio.get_series(spec)
    plan = spec.folds
    tr = slice(0, plan.min_train_size)
    te = slice(plan.min_train_size, plan.min_train_size + plan.horizon)
    ctx = M.FoldContext(y_train=b.values[tr], dates_train=b.dates[tr],
                        dates_future=b.dates[te], spec=spec)
    res = M.SktimeThetaForecaster().fit_predict(ctx)
    assert "predict_interval" in res.meta["interval"]
    assert "predict_quantiles" in res.meta["interval"]
    assert res.meta["interval_vs_quantile_max_gap"] == pytest.approx(0.0, abs=1e-8)


def test_no_metric_is_reimplemented_outside_the_shared_module():
    """MAE/RMSE/coverage must come from common/metrics.py everywhere."""
    import re
    offenders = []
    for name in ("backtesting.py", "plots.py", "capstone_pipeline.py",
                 "build_report.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        for pattern, metric in (
            (r"np\.sqrt\(\s*np\.mean\(\(", "rmse"),
            (r"np\.mean\(\(\s*\w+\s*>=.*&.*<=", "coverage"),
        ):
            for m in re.finditer(pattern, src):
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{name}:{line} reimplements {metric}")
    assert not offenders, offenders


def test_attribution_is_defined_once_in_config():
    """Programme/cohort strings must not be retyped per file."""
    for name in ("build_report.py", "build_notebook.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert config.COURSE_MATERIALS_DATE not in src, (
            f"{name} hardcodes the cohort date instead of using config")
        assert "SDAIAAcademy" not in src, (
            f"{name} hardcodes the SDAIA link instead of using config")
    assert config.PROGRAMME and config.PROGRAMME_AR
    assert config.SDAIA_GITHUB.endswith("SDAIAAcademy")
    assert config.COURSE_MATERIALS_DATE in config.COHORT_STATEMENT
    assert str(config.RNG_SEED) in config.COHORT_STATEMENT


def test_boxcox_inverse_cannot_explode_for_negative_lambda():
    """A predicted value past the transform's range must not become 1e17.

    For lambda < 0 the Box-Cox transform maps (0, inf) onto a *bounded*
    interval: as y -> inf, z -> -1/lambda. A model predicting at or past that
    limit has no finite pre-image, and raising a clamped epsilon to a negative
    power returns something astronomical instead of failing.

    This is a regression test for a real defect: on the retail series
    (lambda ~ -0.52, limit z = 1.905) one global-LightGBM fold predicted past
    the limit, produced a fold MAE of 6.4e8 on a series averaging 560 units,
    and propagated a +1.6-million-percent mean WAPE change into the
    expanding-vs-rolling comparison table. It read as catastrophic model
    failure; it was arithmetic.
    """
    b = dataio.get_series(config.RETAIL)
    train = b.values[:config.RETAIL.folds.min_train_size]
    tf = dataio.make_transform("boxcox", period=7).fit(train)
    assert tf.lmbda < 0, "this test needs the negative-lambda branch"

    limit = -1.0 / tf.lmbda
    cap = tf.train_max * tf.inverse_clip_multiple
    extreme = np.array([limit - 1e-6, limit, limit + 1.0, 1e3, 1e9])
    out = tf.inverse_quantile(extreme)

    assert np.all(np.isfinite(out)), "inverse produced non-finite values"
    assert np.all(out <= cap + 1e-6), (
        f"inverse returned {out.max():.3e}, above the {cap:.1f} cap")
    assert tf.n_inverse_clipped > 0, "clipping happened but was not recorded"


def test_boxcox_round_trip_is_unaffected_by_the_guard():
    """The guard must not perturb any value the model could legitimately emit."""
    for key in ("retail", "workforce"):
        spec = config.DATASETS[key]
        b = dataio.get_series(spec)
        train = b.values[:spec.folds.min_train_size]
        tf = dataio.make_transform("boxcox", period=spec.seasonal_period).fit(train)
        back = tf.inverse_quantile(tf.transform(train))
        assert np.allclose(back, train, rtol=1e-9, atol=1e-6)
        assert tf.n_inverse_clipped == 0, (
            f"{key}: guard fired on ordinary in-sample data")


def test_no_pooled_result_is_physically_impossible():
    """Guards the committed results against a repeat of the blow-up.

    A forecast error orders of magnitude larger than the series itself is never
    a model result worth reporting -- it is a numerical fault. Checked against
    the committed table so a bad run cannot be published quietly.
    """
    table = config.TABLE_DIR / "pooled_metrics.csv"
    if not table.exists():
        pytest.skip("pipeline has not been run")
    pooled = pd.read_csv(table)
    if "wape" not in pooled.columns:
        pytest.skip("no wape column")
    # WAPE is a percentage of total actual volume. Even a hopeless forecast on
    # a 95%-zero series lands in the hundreds; five digits means arithmetic.
    insane = pooled[pooled["wape"] > 10_000]
    assert insane.empty, (
        "physically impossible WAPE in committed results:\n"
        + insane[["dataset", "series", "model", "window_type", "wape"]].to_string())

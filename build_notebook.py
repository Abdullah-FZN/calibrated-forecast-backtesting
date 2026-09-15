"""Construct capstone_notebook.ipynb, then execute it so its outputs are real.

    python build_notebook.py            # build and execute
    python build_notebook.py --no-exec  # build only

**Why the notebook is generated.** The capstone brief requires a notebook whose
every cell carries real captured output, and that runs top to bottom in a fresh
Colab runtime. Hand-maintaining that is how a notebook ends up with cells whose
output predates the code above them. Generating the source here and executing it
with ``nbclient`` means the committed outputs cannot be stale: they are produced
by running the committed source, start to finish, in one pass.

The notebook is the graded deliverable. This file is scaffolding for producing
it, and the analysis modules it imports are the same ones
``capstone_pipeline.py`` uses -- so the notebook's inline numbers and the
committed tables come from one code path, not two.
"""

from __future__ import annotations

import argparse
import sys
import time

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import config

NB_PATH = config.ROOT / "capstone_notebook.ipynb"

MODULES = [
    "common/metrics.py", "common/backtest.py",
    "config.py", "dataio.py", "features.py", "diagnostics.py",
    "models.py", "backtesting.py", "plots.py",
]
DATA = [f"data/{s.filename}" for s in config.DATASETS.values()]


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = []

    # ------------------------------------------------------------------ 0
    cells.append(md(f"""
# Calibrated Forecast Backtesting — Capstone

**A multi-model time series benchmark with walk-forward validation and calibrated prediction intervals.**

| | |
|---|---|
| **Programme** | {config.PROGRAMME} ({config.PROGRAMME_AR}) — {config.PROGRAMME_PROVIDER}, {config.PROGRAMME_FORMAT} |
| **Cohort / session dates** | {config.COHORT_STATEMENT} |
| **Repository** | {config.GITHUB_URL} |
| **SDAIA Academy** | {config.SDAIA_GITHUB} |
| **Course repository** | {config.COURSE_REPO} |

---

## What this notebook does

Most forecasting write-ups answer *which model had the lowest error?* — from a
single holdout. This one answers a harder question: **for series of genuinely
different shapes, which model family should actually be deployed, and how much
should anyone trust the uncertainty it reports?**

The seven sections below follow the capstone brief in order:

1. **Decomposition and diagnostics** — STL, ACF/PACF, ADF *and* KPSS, with the reading spelled out
2. **A classical model** — SARIMA with a per-fold AIC search and a Ljung-Box residual check
3. **An ML/GBM model** — LightGBM on engineered lag/rolling/calendar features, with the leakage guarantee *tested*
4. **A walk-forward backtest** — 6+ folds, expanding **and** rolling, via `common/backtest.py`
5. **Accuracy metrics** — MAE and RMSE alongside MASE/WAPE, with the scale-free choice justified per series
6. **Probabilistic forecasts** — 90% intervals from every family, scored on `coverage()` **and** `interval_width()`
7. **Model comparison and recommendation** — reasoned from history length, interpretability, interval support and compute

**Primary series:** `retail_demand.csv`, Riyadh/Grocery — the deep walkthrough.
Sections 4–7 then widen to all four datasets, because the interesting result is
that *the winner changes*, and the reason it changes is legible.

> **All four datasets are synthetic**, from the course's seeded
> `data/generate_series.py`. Every number here describes those synthetic
> series — not any real retailer, employer, or government indicator.
"""))

    # ------------------------------------------------------------------ setup
    cells.append(md("""
## Environment setup

One cell, the same shape as every lab in this course: install what is missing,
then `fetch()` the shared utilities and data — looking for a local checkout
first and falling back to `raw.githubusercontent.com`, so the identical
notebook runs from a clone or cold on Colab with nothing cloned.
"""))
    cells.append(code(f"""
import subprocess, sys, pathlib, urllib.request, warnings

def _pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

try:
    import statsmodels, lightgbm, prophet, sktime          # noqa: F401
except ImportError:
    # sktime requires pandas<3 and will silently downgrade it otherwise --
    # pinning here makes that constraint visible rather than surprising.
    _pip_install("pandas<3", "statsmodels", "lightgbm", "prophet", "sktime<2",
                 "matplotlib", "scipy", "scikit-learn")

REPO   = "{config.GITHUB_REPO}"
BRANCH = "{config.GITHUB_BRANCH}"
RAW    = f"https://raw.githubusercontent.com/{{REPO}}/{{BRANCH}}"

def fetch(rel_path: str) -> str:
    \"\"\"Local checkout first; download from the repository otherwise.\"\"\"
    local = pathlib.Path(rel_path)
    if local.exists():
        return str(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{{RAW}}/{{rel_path}}", local)
    return str(local)

for _f in {MODULES + DATA!r}:
    fetch(_f)

for _p in (".", "common"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import config, dataio, features as F, diagnostics as D, models as M
import backtesting as B, plots as P

RNG_SEED = config.RNG_SEED          # 20260912 -- the generator's own seed
np.random.seed(RNG_SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

import importlib.metadata as _md
print("environment")
for _pkg in ("pandas", "numpy", "statsmodels", "lightgbm", "prophet", "sktime"):
    try:
        print(f"  {{_pkg:<12}} {{_md.version(_pkg)}}")
    except Exception:
        print(f"  {{_pkg:<12}} (not installed)")
print(f"  seed         {{RNG_SEED}}")
print(f"  nominal coverage {{config.NOMINAL_COVERAGE:.0%}} "
      f"(+/-{{config.COVERAGE_TOLERANCE:.0%}} tolerance)")
"""))

    # ------------------------------------------------------------------ data
    cells.append(md("""
### The four series

Loaded through a pipeline that *proves* each series is regularly spaced before
anything relies on it. A silently missing date would turn every `lag_7` feature
downstream into a lag of the wrong length — and would never raise.
"""))
    cells.append(code("""
rows = []
for key, spec in config.DATASETS.items():
    for b in dataio.iter_series(spec):     # .validate() runs inside; raises on any gap
        d = b.describe()
        d["seasonal_period"] = spec.seasonal_period
        rows.append(d)

summary = pd.DataFrame(rows)[
    ["dataset", "series", "n", "start", "end", "freq", "seasonal_period",
     "mean", "std", "min", "max", "zero_rate"]
]
print(f"{len(summary)} series across {summary['dataset'].nunique()} datasets, "
      f"all validated gap-free\\n")
summary.round(2)
"""))

    cells.append(code("""
SPEC = config.RETAIL
SERIES = dataio.get_series(SPEC)          # Riyadh/Grocery
print(f"Primary series: {SERIES.label}  |  n={SERIES.n}  "
      f"{SERIES.dates[0].date()} -> {SERIES.dates[-1].date()}  freq={SPEC.freq}")
print(f"Seasonal period {SPEC.seasonal_period} (weekly); "
      f"backtest = {SPEC.folds.n_folds} folds x {SPEC.folds.horizon}-day horizon")

P.apply_style()
_ = P.plot_series_overview(
    SERIES, "nb_retail_overview.png",
    scored_from=SPEC.folds.first_test_index(SERIES.n))
from IPython.display import Image, display
display(Image(filename="outputs/figures/nb_retail_overview.png"))
"""))

    # ------------------------------------------------------------------ 1
    cells.append(md("""
---

# 1. Decomposition and diagnostics

Three questions, each answered with a statistic **and** a reading of it:

1. How much of this series is trend, and how much is seasonality?
2. Is it stationary — and if not, what differencing does it need?
3. What do the ACF and PACF suggest about model order?
"""))
    cells.append(code("""
stl = D.stl_decompose(SERIES.values, SPEC.seasonal_period)
print(stl.interpretation)
_ = P.plot_decomposition(SERIES, stl, "nb_retail_decomposition.png")
display(Image(filename="outputs/figures/nb_retail_decomposition.png"))
"""))
    cells.append(md("""
### Additive or multiplicative? Decided from the data, not asserted

A multiplicative series has spread that grows with level. Rather than eyeball
it, fit both decompositions and check which leaves *less* level-dependence in
the residual.
"""))
    cells.append(code("""
form = D.additive_vs_multiplicative(SERIES.values, SPEC.seasonal_period)
print(f"CHOICE: {form['choice']}\\n")
print(form["interpretation"])
"""))
    cells.append(md("""
### Stationarity — ADF **and** KPSS

ADF's null is "has a unit root"; KPSS's null is "is stationary". They are
opposites, and a lone ADF p-value cannot tell *"there is a unit root"* apart
from *"this sample is too small to say"*. Running both makes that distinction
visible.
"""))
    cells.append(code("""
for r in D.differencing_ladder(SERIES.values, SERIES.label, SPEC.seasonal_period):
    print(f"[{r.verdict:^17}] {r.name}")
    print(f"    ADF  stat={r.adf_stat:8.4f}  p={r.adf_p:.4f}")
    print(f"    KPSS stat={r.kpss_stat:8.4f}  p{r.kpss_p_bound}{r.kpss_p:.3f}")
    print(f"    -> {r.interpretation}\\n")
"""))
    cells.append(md("""
### Applying the differencing the tests call for

A test result that changes nothing is decoration. `recommend_differencing`
turns the ladder above into an actual `(d, D)` — and the SARIMA fit in section 2
takes `d` from it rather than letting AIC treat the order of integration as one
more free parameter.

Two rules, both visible in the rationale it returns:

- **`d` comes from ADF/KPSS.** If the level is already stationary, `d=0`.
  Differencing a stationary series is not a harmless default: it inflates the
  residual variance and manufactures an MA term the data never had.
- **`D` comes from seasonal *strength*, not the unit-root test.** A seasonal
  unit root and a strong deterministic seasonal pattern both make the level
  look non-stationary, and only the first calls for a seasonal difference. STL's
  F_S measures the quantity `D=1` actually removes; the 0.64 threshold is
  Hyndman & Athanasopoulos'.
"""))
    cells.append(code("""
plan = D.recommend_differencing(SERIES.values, SPEC.seasonal_period,
                                name=SERIES.label)
print(f"APPLIED:  d={plan.d}, D={plan.seasonal_D} "
      f"(seasonal period {plan.seasonal_period})\\n")
print(plan.rationale)
"""))
    cells.append(code("""
# Apply it, and show the test result before and after on the same series.
differenced = D.apply_differencing(SERIES.values, d=plan.d, D=plan.seasonal_D,
                                   period=SPEC.seasonal_period)
before = D.stationarity_report(SERIES.values, "level")
after = D.stationarity_report(differenced, f"after d={plan.d}, D={plan.seasonal_D}")

print(f"{'':22} {'ADF p':>9} {'KPSS p':>10}  verdict")
print(f"{'level':22} {before.adf_p:9.4f} "
      f"{before.kpss_p_bound + format(before.kpss_p, '.3f'):>10}  {before.verdict}")
print(f"{'after differencing':22} {after.adf_p:9.4f} "
      f"{after.kpss_p_bound + format(after.kpss_p, '.3f'):>10}  {after.verdict}")
print(f"\\nobservations: {len(SERIES.values)} -> {len(differenced)} "
      f"({len(SERIES.values) - len(differenced)} consumed by differencing)")
print(f"variance:     {SERIES.values.var(ddof=1):,.1f} -> {differenced.var(ddof=1):,.1f}")
"""))
    cells.append(code("""
# The differenced series is what a SARIMA with this (d, D) actually fits.
fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=False)
axes[0].plot(SERIES.dates, SERIES.values, color=P.FAMILY_COLOR["classical"],
             linewidth=0.9)
axes[0].set_title(f"{SERIES.label} — level (ADF p={before.adf_p:.4f}, {before.verdict})",
                  loc="left", fontsize=9)
axes[1].plot(SERIES.dates[len(SERIES.dates) - len(differenced):], differenced,
             color=P.FAMILY_COLOR["ml"], linewidth=0.7)
axes[1].axhline(0, color=P.AXIS, linewidth=0.8)
axes[1].set_title(f"after d={plan.d}, D={plan.seasonal_D} "
                  f"(ADF p={after.adf_p:.4f}, {after.verdict})",
                  loc="left", fontsize=9)
for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
# plots.py pins the Agg backend so figures render identically headless, which
# means plt.show() is a no-op here: save and display, as every other figure
# cell in this notebook does, so the output is actually captured in the file.
fig.savefig("outputs/figures/nb_retail_differencing.png",
            dpi=130, bbox_inches="tight", facecolor=P.SURFACE)
plt.close(fig)
display(Image(filename="outputs/figures/nb_retail_differencing.png"))
"""))
    cells.append(md("""
> **Note on the KPSS p-value.** statsmodels interpolates it from a lookup table
> spanning 0.01–0.10 and returns the nearest endpoint outside that range. A
> printed `0.100` therefore usually means *"at least 0.10"*, which is why the
> bound (`>=` / `<=`) is shown rather than a bare equals sign.
"""))
    cells.append(code("""
ac = D.acf_pacf_report(SERIES.values, nlags=60,
                       seasonal_period=SPEC.seasonal_period, label=SERIES.label)
print(ac.interpretation)
_ = P.plot_acf_pacf(ac, SPEC.seasonal_period, "nb_retail_acf.png", SERIES.label)
display(Image(filename="outputs/figures/nb_retail_acf.png"))
"""))
    cells.append(md("""
### Variance stabilisation — and why not maximum likelihood

The series is multiplicative, so an additive model should be fit on a
transformed scale. The Box-Cox λ is chosen by **Guerrero's criterion**
(minimise the coefficient of variation of subseries spread) rather than by
maximum likelihood.

ML picks the λ that makes the *marginal* distribution most normal — but on a
strongly seasonal series that marginal is a mixture across weekdays, so ML is
answering a question nobody asked. On this series it returns λ below −1, a
reciprocal transform that flattens the right tail beautifully and destroys the
additive seasonal structure SARIMA is about to fit.

**λ is fitted on the training window only** — it is a fitted statistic, and
estimating it once over the whole series is the same leak as a global
`StandardScaler`, wearing a different hat.
"""))
    cells.append(code("""
train_window = SERIES.values[:SPEC.folds.min_train_size]

tf_g = dataio.make_transform("boxcox", period=SPEC.seasonal_period).fit(train_window)
tf_m = dataio.make_transform("boxcox", period=SPEC.seasonal_period,
                             method="mle").fit(train_window)
print(f"Guerrero lambda = {tf_g.lmbda:+.3f}     (variance stabilisation)")
print(f"MLE      lambda = {tf_m.lmbda:+.3f}     (marginal normality)\\n")

def level_spread_corr(v, period=7):
    w = v[len(v) % period:].reshape(-1, period)
    return float(np.corrcoef(w.mean(1), w.std(1, ddof=1))[0, 1])

print("Correlation between weekly level and weekly spread")
print(f"  raw scale        {level_spread_corr(train_window):+.3f}")
print(f"  Box-Cox (Guerrero) {level_spread_corr(tf_g.transform(train_window)):+.3f}")
print("\\nThe transform's entire job is to drive that second number to zero.")

_ = P.plot_transform_effect(train_window, tf_g.transform(train_window),
                            SPEC.seasonal_period, tf_g.lmbda,
                            "nb_retail_transform.png", SERIES.label)
display(Image(filename="outputs/figures/nb_retail_transform.png"))
"""))

    # ------------------------------------------------------------------ 2
    cells.append(md("""
---

# 2. Classical model — SARIMA, AIC-selected, Ljung-Box checked

statsmodels ships no `auto_arima` (that is a pmdarima/sktime feature, outside
this course's four tools), so the order is chosen deliberately: a **two-stage
AIC search run inside each fold**.

- *Why a search:* fixing the order by eye from one ACF/PACF plot of the whole
  series would choose it using data from every fold's test window.
- *Why two stages:* a full 8×4 grid is 32 fits per fold — over half an hour on
  the workforce backtest alone, spent mostly on combinations AIC discards
  immediately. Stage 1 picks the seasonal order with a plain (1,1,1) core;
  stage 2 picks (p,d,q) given that. Twelve fits per fold. It is greedy and can
  miss a joint optimum, which is stated rather than glossed.
"""))
    cells.append(code("""
plan = SPEC.folds
tr = slice(0, plan.min_train_size)
te = slice(plan.min_train_size, plan.min_train_size + plan.horizon)
ctx = M.FoldContext(y_train=SERIES.values[tr], dates_train=SERIES.dates[tr],
                    dates_future=SERIES.dates[te], spec=SPEC)

sarima = M.SarimaForecaster()
res_sarima = sarima.fit_predict(ctx)
meta = res_sarima.meta
print(f"AIC-selected order         : {meta['order']}  seasonal {meta['seasonal_order']}")
print(f"AIC / BIC                  : {meta['aic']:.2f} / {meta['bic']:.2f}")
print(f"Box-Cox lambda (train only): {meta['lambda']:+.3f}")
print(f"Converged                  : {meta['converged']}")
print(f"Fit time                   : {meta['fit_seconds']:.2f}s")
"""))
    cells.append(md("""
### Residual diagnostic — Ljung-Box

The null is **"the residuals are white noise"**. A *large* p-value is the good
outcome: it means we fail to reject that null, so no autocorrelation is left
that the test can detect. Reading this backwards is a common error, so the
interpretation says it explicitly.
"""))
    cells.append(code("""
print(f"Ljung-Box(lag={meta['ljung_box_lags']}): "
      f"Q={meta['ljung_box_stat']:.2f}, p={meta['ljung_box_p']:.4f}")
print(f"Residuals are white noise: {meta['residuals_white_noise']}\\n")
print("PASS -- the model extracted the linear structure available to it."
      if meta["residuals_white_noise"] else
      "FAIL -- structure remains; the order should be reconsidered.")
"""))
    cells.append(md("""
### The exponential-smoothing family

SES, Holt, damped Holt and Holt-Winters, scored by AIC on the same window — the
same rule that picked the SARIMA order, so the two classical families are
selected comparably.

`ETSModel` is used rather than `ExponentialSmoothing` for one reason that
matters here: only the state-space formulation exposes **analytic prediction
intervals**. Without it this family's interval would have to be simulated, and
would not be comparable to SARIMA's in section 6.
"""))
    cells.append(code("""
res_ets = M.EtsForecaster().fit_predict(ctx)
m2 = res_ets.meta
print(f"AIC-selected configuration : {m2['ets_config']}")
print(f"  trend={m2['trend']}  seasonal={m2['seasonal']}  damped={m2['damped']}")
print(f"AIC / BIC                  : {m2['aic']:.2f} / {m2['bic']:.2f}")
print(f"Ljung-Box p                : {m2['ljung_box_p']:.4f}  "
      f"(white noise: {m2['residuals_white_noise']})")
"""))

    # ------------------------------------------------------------------ 3
    cells.append(md("""
---

# 3. ML model — LightGBM, and proving the features do not leak

LightGBM has no notion of time. Every temporal structure must arrive as a
column, and every column is a chance to hand the model a number it would not
possess at forecast time.

**The design that makes leakage structurally impossible.** A training row is
not *a day*. It is an `(origin t, horizon step h)` pair meaning *standing at t,
knowing `y[0..t]` and nothing later, predict `y[t+h]`*. Every predictor is:

- a function of `y[0..t]` — lags and rolling statistics, all ending **at** the origin; or
- a function of the target **date** `dates[t+h]` — day of week, Fourier terms, a moving-holiday flag, all knowable arbitrarily far ahead; or
- `h` itself.

This is **direct** multi-step forecasting. The alternative — recursive
forecasting, predicting step 1, appending it, recomputing lags — is correct
only if implemented perfectly, is the single most common correctness bug this
course names, and compounds a step-1 error through fourteen substitutions.
Direct forecasting sidesteps the entire class of bug because `h` is an *input*,
not a loop counter.
"""))
    cells.append(code("""
builder = F.DirectMultiStepBuilder(
    lags=SPEC.lags, windows=SPEC.rolling_windows,
    horizon=plan.horizon, freq=SPEC.freq,
    target_mode="delta", use_holidays=True,
)
design = builder.build_train(SERIES.values[tr], SERIES.dates[tr])
print(f"{design.n_rows:,} training rows x {len(design.feature_names)} features "
      f"from a {plan.min_train_size}-day window\\n")
print("features:", ", ".join(design.feature_names))
design.X.head(3)
"""))
    cells.append(md("""
### The leakage audit — tested, not asserted

"I only sliced training rows" is what every leaked pipeline also says. This
check replaces the **entire future** with different numbers, rebuilds both
design matrices, and asserts that not a single feature value moves. If any
column had reached past the forecast origin, it would move.
"""))
    cells.append(code("""
audit = F.assert_no_leakage(builder, SERIES.values, SERIES.dates,
                            train_end=plan.min_train_size)
for k, v in audit.items():
    print(f"  {k:<34} {v}")
print("\\nPASSED: no feature value changed when the future was replaced.")
"""))
    cells.append(md("""
**What is deliberately withheld.** The generator creates random *promo shocks*
with no advance signal. A planner at the forecast origin could not know them,
so no model here is given a promo flag — the error they cause is irreducible
and is part of every number in this notebook.

The **moving-holiday window is** supplied, to SARIMA (`exog`), Prophet
(`holidays=`) and LightGBM (a column) alike. A Hijri-calendar date is known
years ahead; this is the known-future regressor those APIs exist for. A fixed
`dayofyear` feature could not represent it, because the window drifts ~11 days
earlier each year.
"""))
    cells.append(code("""
hol = F.moving_holiday_dates(SERIES.dates[0], SERIES.dates[-1])
first_by_year = {}
for d in hol:
    first_by_year.setdefault(d.year, d)
print("First day of the moving-holiday window, by year:")
for y, d in sorted(first_by_year.items()):
    print(f"  {y}: {d.date()}")
print("\\nIt drifts earlier each year -- which is exactly why a fixed "
      "day-of-year feature cannot capture it.")
"""))
    cells.append(code("""
res_lgbm = M.LgbmQuantileForecaster(use_holidays=True).fit_predict(ctx)
imp = res_lgbm.meta["feature_importance"]
print(f"trained on {res_lgbm.meta['n_train_rows']:,} rows, "
      f"{res_lgbm.meta['n_features']} features, "
      f"{res_lgbm.meta['fit_seconds']:.2f}s for 3 quantile models\\n")
_ = P.plot_feature_importance(imp, "nb_retail_importance.png",
                              "LightGBM feature importance — Riyadh/Grocery")
display(Image(filename="outputs/figures/nb_retail_importance.png"))
"""))

    # ------------------------------------------------------------------ 4
    cells.append(md("""
---

# 4. Walk-forward backtest

Fold geometry comes from the course's own `common/backtest.py`. Both window
types are run over the **identical** test windows, which is what makes
expanding-vs-rolling a controlled experiment rather than two unrelated
backtests.

- **Expanding** — training window grows; every fold trains on everything before it.
- **Rolling** — training window is a fixed size that slides forward, ageing old history out.
"""))
    cells.append(code("""
splits = {wt: B.make_splits(SERIES.n, plan, wt) for wt in B.WINDOW_TYPES}
for wt, sp in splits.items():
    info = B.assert_splits_sound(sp, SERIES.n)   # raises on any overlap
    print(f"{wt:>9}: train sizes {info['train_sizes']}, "
          f"scored index {info['first_scored_index']}–{info['last_scored_index']}")

rows = []
for i, (t, e) in enumerate(splits["expanding"]):
    rows.append({
        "fold": i,
        "train rows": t.stop - t.start,
        "train span": f"{SERIES.dates[t.start].date()} – {SERIES.dates[t.stop-1].date()}",
        "test span": f"{SERIES.dates[e.start].date()} – {SERIES.dates[e.stop-1].date()}",
    })
pd.DataFrame(rows)
"""))
    cells.append(code("""
_ = P.plot_fold_layout(SERIES, splits, "nb_retail_folds.png")
display(Image(filename="outputs/figures/nb_retail_folds.png"))
"""))
    cells.append(md("""
### Is this really the course's harness?

The brief allows "`common/backtest.py`'s harness **or an equivalent you can
justify**". This project uses the course's split functions directly and adds a
runner that also carries dates, intervals and per-fold metadata — things
`run_backtest` is not typed to return.

Rather than assert the equivalence, it is **checked**: the seasonal-naive
baseline is run through the course's own `run_backtest` and through this
project's runner, and the point forecasts are compared element by element.
"""))
    cells.append(code("""
for wt in B.WINDOW_TYPES:
    r = B.verify_against_course_harness(SERIES, wt)
    print(f"{wt:>9}: {r['n_folds']} folds, identical={r['identical']}, "
          f"max abs difference={r['max_abs_point_difference']:.1e}")
"""))
    cells.append(md("""
### Running every model across every fold

Seven forecasters, four families, both window types. Each `fit_predict` call
receives only that fold's training window and the future *dates* — never a
future value, and never a model object left over from another fold.
"""))
    cells.append(code("""
import time
t0 = time.time()
results, per_fold_frames, pooled_rows = {}, [], []

for wt in B.WINDOW_TYPES:
    for fc in M.default_forecasters(SPEC):
        outs = B.run_walk_forward(SERIES, fc, wt)
        results[(wt, fc.name)] = outs
        pf = B.score_outcomes(outs, SPEC)
        per_fold_frames.append(pf)
        pooled_rows.append(B.pool_outcomes(outs, SPEC))

per_fold = pd.concat(per_fold_frames, ignore_index=True)
pooled = pd.DataFrame(pooled_rows)
print(f"{len(pooled)} model x window-type runs, "
      f"{len(per_fold)} scored folds, in {time.time() - t0:.0f}s")
"""))

    # ------------------------------------------------------------------ 5
    cells.append(md("""
---

# 5. Accuracy metrics

MAE and RMSE in the series' own units, plus a **scale-free** metric so results
are comparable across series and against a baseline.

**Which scale-free metric, and why.** This series has no zeros and no
near-zeros, so MAPE would be defined — but **MASE** is reported instead,
because it is scaled against a *seasonal-naive* forecast on the training
window. That makes "below 1.0" mean something concrete: *beats repeating last
week*. On `intermittent_demand` (section 7) MASE's denominator becomes unstable
and WAPE is the headline instead. The choice follows the series' shape, not
habit.
"""))
    cells.append(code("""
cols = ["model", "family", "mae", "rmse", "wape", "mase", "coverage",
        "interval_width", "width_pct_of_mean", "pinball_mean",
        "fit_seconds_per_fold"]
exp = pooled[pooled["window_type"] == "expanding"][cols].sort_values("mase")
exp.round(3)
"""))
    cells.append(md("""
### The baseline every model has to beat

MASE is a ratio, and the denominator is a **seasonal-naive** forecast — repeat
the last full seasonal cycle — computed on each fold's *training* window. So
`MASE < 1.0` is not a vague compliment: it means the model beat "just repeat
last week", scored on the same folds.

That baseline is run through the identical harness as every other model, with
its own conformal interval, so it is held to the same accuracy *and*
calibration bar rather than being exempted from half the report. A model that
cannot clear it is not a model worth deploying, however sophisticated.
"""))
    cells.append(code("""
from backtest import seasonal_naive_forecast

# The baseline, run as a first-class competitor through the same harness.
baseline_outcomes = B.run_walk_forward(SERIES, M.SeasonalNaive(), "expanding")
baseline = B.pool_outcomes(baseline_outcomes, SPEC)

print(f"Seasonal-naive baseline (period={SPEC.seasonal_period}), "
      f"{baseline['n_folds_scored']} folds, {baseline['n_points']} points")
print(f"  MAE  {baseline['mae']:8.2f}")
print(f"  RMSE {baseline['rmse']:8.2f}")
print(f"  WAPE {baseline['wape']:8.2f}%")
print(f"  MASE {baseline['mase']:8.3f}   <- 1.0 by construction on its own scale")

# What the seasonal-naive forecast literally is, on the last fold:
last = baseline_outcomes[-1]
naive_by_hand = seasonal_naive_forecast(last.y_train, len(last.y_true),
                                        SPEC.seasonal_period)
print()
print(f"last fold: seasonal_naive_forecast reproduces the harness output "
      f"exactly: {np.allclose(naive_by_hand, last.point)}")
"""))
    cells.append(code("""
beaten = exp[exp["family"] != "baseline"].copy()
beaten["beats_naive_mase"] = beaten["mase"] < 1.0
beaten["vs_naive_mae_%"] = (beaten["mae"] / baseline["mae"] - 1) * 100
print(f"{int(beaten['beats_naive_mase'].sum())} of {len(beaten)} models beat "
      f"seasonal-naive on MASE")
beaten[["model", "family", "mae", "mase", "beats_naive_mase",
        "vs_naive_mae_%"]].sort_values("mase").round(3)
"""))
    cells.append(md("""
### A mean across folds hides the fold that blew up

Walk-forward validation exists to expose variance. Reporting only the mean
throws that away, so the spread is reported beside it — and the per-fold trace
is plotted, because two models with an identical mean can differ enormously in
consistency.
"""))
    cells.append(code("""
summary = B.summarise_per_fold(per_fold)
show = [c for c in summary.columns
        if c in ("model", "family", "window_type", "n_folds")
        or c.startswith(("wape_", "mase_"))]
summary[show].round(3)
"""))
    cells.append(code("""
_ = P.plot_per_fold_metric(per_fold, "wape", "nb_retail_per_fold.png",
                           "Per-fold WAPE — Riyadh/Grocery")
display(Image(filename="outputs/figures/nb_retail_per_fold.png"))
"""))

    # ------------------------------------------------------------------ 6
    cells.append(md("""
---

# 6. Probabilistic forecasts — coverage **and** width

Every model here produces a 90% interval, by four different constructions:

| Construction | Models | How |
|---|---|---|
| Analytic | SARIMA, ETS | State-space prediction intervals |
| Native | Prophet | `yhat_lower` / `yhat_upper` from simulated trend changepoints |
| Uniform API | sktime Theta | `predict_interval(coverage=0.9)` |
| Built | LightGBM | A quantile objective at α = 0.05 / 0.95, **or** split conformal |

**Coverage alone is gameable and is never quoted alone here.** An interval of
ŷ ± 5000 covers 100% of anything and says nothing. Width alone is equally
meaningless without a coverage floor. The pair is the measurement.

Width is additionally normalised by the series' own mean level, so intervals on
a series averaging 670 units and one averaging 0.4 can sit on the same axis.
"""))
    cells.append(code("""
cal = pooled[pooled["window_type"] == "expanding"][
    ["model", "family", "coverage", "interval_width", "width_pct_of_mean",
     "pinball_lower", "pinball_upper", "calibrated"]
].sort_values("coverage", ascending=False)
print(f"nominal level: {config.NOMINAL_COVERAGE:.0%}  "
      f"(calibrated = within +/-{config.COVERAGE_TOLERANCE:.0%})\\n")
cal.round(3)
"""))
    cells.append(code("""
_ = P.plot_calibration(pooled[pooled["window_type"] == "expanding"],
                       "nb_retail_calibration.png",
                       "Calibration — Riyadh/Grocery, expanding window")
display(Image(filename="outputs/figures/nb_retail_calibration.png"))
"""))
    cells.append(md("""
### Coverage decays with horizon — which is why the conformal margin is per-step

A pooled coverage number can hit 90% exactly while being far too wide at h=1
and far too narrow at h=14. Only a per-step view shows it. This is precisely
why `LgbmConformalForecaster` computes its margin **per horizon step** rather
than pooling residuals across the whole window.
"""))
    cells.append(code("""
cov_rows = []
for (wt, name), outs in results.items():
    if wt != "expanding":
        continue
    ch = B.coverage_by_horizon(outs)
    if not ch.empty:
        cov_rows.append(ch)
cov_h = pd.concat(cov_rows, ignore_index=True)
_ = P.plot_coverage_by_horizon(cov_h, "nb_retail_coverage_h.png",
                               "Coverage by horizon step — Riyadh/Grocery")
display(Image(filename="outputs/figures/nb_retail_coverage_h.png"))
"""))
    cells.append(code("""
fold0 = {name: outs[0] for (wt, name), outs in results.items()
         if wt == "expanding" and outs and outs[0].error is None
         and name in ("seasonal_naive", "sarima", "ets", "prophet",
                      "lgbm_quantile", "lgbm_conformal")}
_ = P.plot_forecast_fan(fold0, SERIES, 0, "nb_retail_fan.png")
display(Image(filename="outputs/figures/nb_retail_fan.png"))
"""))

    # ------------------------------------------------------------------ 7
    cells.append(md("""
---

# 7. Model comparison and recommendation

### First: does the verdict survive a second window type?

Both window types scored identical test windows, so any difference is
attributable to the training window's shape alone.
"""))
    cells.append(code("""
wide = pooled.pivot_table(index=["model", "family"], columns="window_type",
                          values="wape").dropna()
wide["delta (rolling - expanding)"] = wide["rolling"] - wide["expanding"]
wide["better"] = np.where(wide["delta (rolling - expanding)"] < 0,
                          "rolling", "expanding")
wide.round(3).sort_values("expanding")
"""))
    cells.append(md("""
### Then: the same question across all four datasets

The committed results of the full pipeline run — every dataset, every series,
both window types. These are loaded rather than recomputed here: re-running
them inline would add roughly forty minutes to this notebook for numbers that
are already reproducible via `python capstone_pipeline.py`.
"""))
    cells.append(code("""
full = pd.read_csv(fetch("outputs/tables/pooled_metrics.csv"))
primary = {k: ("/".join(s.primary_group) if s.primary_group else k)
           for k, s in config.DATASETS.items()}
mask = full.apply(lambda r: r["series"] == primary[r["dataset"]], axis=1)
head = full[mask & (full["window_type"] == "expanding")]

out = []
for key, spec in config.DATASETS.items():
    sub = head[head["dataset"] == key]
    if sub.empty:
        continue
    m = spec.scale_free_metric
    cand = sub[(sub["family"] != "baseline") & sub[m].notna()]
    best = cand.loc[cand[m].idxmin()]
    calib = cand[cand["coverage"].apply(config.is_calibrated)]
    deploy = calib.loc[calib[m].idxmin()] if not calib.empty else best
    out.append({
        "dataset": key,
        "headline metric": m.upper(),
        "most accurate": best["model"],
        "score": round(best[m], 3),
        "recommended (calibrated)": deploy["model"],
        "its coverage": round(deploy["coverage"], 3),
        "its width % of level": round(deploy["width_pct_of_mean"], 1),
    })
pd.DataFrame(out)
"""))
    cells.append(code("""
n_cal = int(full["coverage"].apply(config.is_calibrated).sum())
print(f"Across all {len(full)} model x dataset x window-type runs, "
      f"{n_cal} ({n_cal/len(full):.0%}) produced a 90% interval "
      f"within +/-{config.COVERAGE_TOLERANCE:.0%} of nominal.\\n")
by_fam = (full.assign(cal=full["coverage"].apply(config.is_calibrated))
              .groupby("family")
              .agg(runs=("cal", "size"), calibrated=("cal", "sum"),
                   mean_coverage=("coverage", "mean"),
                   mean_width_pct=("width_pct_of_mean", "mean")))
by_fam.round(3)
"""))
    cells.append(md(f"""
### The decision framework

Accuracy is one axis of four, and on these four series it is rarely the binding
one. In the order the questions should be asked:

**1. How much history is there?** This rules tools out before anything else is
considered. The economic series has 108 points *in total* — a `lag_12` feature
costs 11% of it before a single tree is grown, and `mase(..., seasonal_period=12)`
is undefined on the shorter training windows. A classical model estimating a
handful of parameters is not merely safe there; it is what the sample size
permits.

**2. Does anyone have to explain the forecast?** ARIMA coefficients and
Prophet's decomposed components are readable by a domain expert who has never
seen the code. LightGBM feature importances say which *columns* mattered, not a
story about trend and seasonality anyone can put in a slide. When a headcount
plan must be defended to a manager, that outranks a point of WAPE.

**3. Does the decision need a calibrated interval, or just a number?** This is
where the ranking most often changes — and where the results above are least
comfortable. A staffing decision is made against the *upper bound*, not the
mean. An interval that under-covers is worse than no interval, because it
manufactures confidence. statsmodels, Prophet and sktime hand you one by
default; LightGBM does not, and "a LightGBM model with an interval" is a
decision to build one, not a default you inherit.

**4. What is the compute budget, and how many series?** Per-series classical
fits are cheap individually and scale linearly — a thousand series is a
thousand fits, on every fold, forever. One global tree model amortises training
across every series it covers. At six series this is irrelevant; at six
thousand it is the only axis that matters.

### Recommendations

**Demand planning (retail).** Stable weekly and yearly seasonality with a
modest trend is the shape the classical family was built for, and the interval
comes free and analytic. Reach for a tree model when you have *exogenous*
signal a classical model cannot absorb — a promo calendar, price, weather —
because that, not raw flexibility, is what it is genuinely better at.

**Workforce projection.** The model matters less than the validation design.
Every family degrades at the break; what separates them is how fast they
recover and whether their interval widens honestly meanwhile. Pair a classical
model with a conformal interval recomputed per fold, and monitor coverage in
production as a drift alarm — a calibrated interval that suddenly stops
covering is the earliest signal the regime moved.

**Economic indicator.** Short and monthly. A few-parameter model, an analytic
interval, and scepticism toward anything claiming to have learned rich
nonlinear structure from 108 points.

**Intermittent spare parts.** None of this course's four tools is the right
tool, and saying so is the honest conclusion rather than a failure to find one.
Almost all the signal is *whether* an order occurs, not how large it is;
Croston's method and TSB model exactly that — inter-arrival time and demand
size as separate processes — and neither is in this toolbox. What can be said
with confidence is narrower and still useful: report WAPE, never MAPE; expect
WAPE above 100% and understand why; and treat any point forecast here as a
re-order-point input, not a daily quantity.

---

## Limitations

- The data is **synthetic**; a model that suits one of these textures is
  evidence about that texture, not about real demand.
- The SARIMA search is **greedy** (12 fits/fold, not 32) and can miss a joint optimum.
- **Croston's/TSB are not implemented**, so the intermittent conclusion is
  "none of these four fits" rather than "here is the one that does".
- Prophet runs in **MAP mode**, so its intervals omit seasonality uncertainty.
- Calibration is assessed at **one nominal level** ({config.NOMINAL_COVERAGE:.0%}).
- **One seed** — this is a study of one realisation of each process.

Full written analysis: [`CAPSTONE_REPORT.md`](CAPSTONE_REPORT.md) ·
Tests: `python -m pytest tests/ -q` ({config.count_tests()} tests) ·
Pipeline: `python capstone_pipeline.py`
"""))

    nb = new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "colab": {"provenance": []},
    }
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true",
                    help="write the notebook without executing it")
    args = ap.parse_args()

    nb = build()
    nbf.write(nb, NB_PATH)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"built {NB_PATH.name}: {len(nb.cells)} cells ({n_code} code)")

    if args.no_exec:
        return 0

    from nbclient import NotebookClient

    print("executing (this runs the real backtest -- several minutes)...")
    t0 = time.time()
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(config.ROOT)}})
    client.execute()
    nbf.write(nb, NB_PATH)

    n_out = sum(1 for c in nb.cells
                if c.cell_type == "code" and c.get("outputs"))
    print(f"executed in {time.time() - t0:.0f}s -- "
          f"{n_out}/{n_code} code cells produced output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Central configuration for the capstone forecasting pipeline.

Everything that is a *choice* -- which series we forecast, how the folds are
cut, which nominal interval level we calibrate against, where artefacts land --
lives here rather than being scattered through the pipeline. Two reasons:

1. A reviewer can audit every experimental decision in one file instead of
   grepping for magic numbers.
2. The fold geometry on this course's data is tight enough that it has to be
   checked arithmetically (see ``FoldPlan.validate``), not eyeballed. The
   economic series has 108 monthly rows in total; the workforce series has to
   place a test window *on top of* a structural break at row 456 or the whole
   expanding-vs-rolling comparison tests nothing. Those constraints are
   asserted at load time instead of failing 20 minutes into a run.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

# The same seed ``data/generate_series.py`` uses. The course's CLAUDE.md is
# explicit that labs should not pick a different one -- there is no reason for
# the modelling randomness to be reproducible differently from the data it is
# forecasting.
RNG_SEED = 20260912

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COMMON_DIR = ROOT / "common"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"

for _d in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Evaluation policy
# --------------------------------------------------------------------------

#: Nominal coverage for every prediction interval in this project. One level,
#: used everywhere, so coverage numbers are comparable across model families
#: without a footnote about which band each one reported.
NOMINAL_COVERAGE = 0.90

#: The two quantiles that bound a 90% interval. Also the quantiles LightGBM's
#: quantile objective is trained against and the ones ``pinball_loss`` scores.
LOWER_Q = round((1.0 - NOMINAL_COVERAGE) / 2.0, 10)   # 0.05
UPPER_Q = round(1.0 - LOWER_Q, 10)                    # 0.95

#: How far empirical coverage may drift from nominal before we call an
#: interval miscalibrated rather than "close enough": +/-5 percentage points on
#: a 90% target, judged over the pooled test points of a whole backtest.
#: Deliberately explicit -- reference/metrics_cheatsheet.qmd warns against
#: eyeballing "close enough", so the threshold is stated once and applied
#: mechanically in the report.
COVERAGE_TOLERANCE = 0.05


def is_calibrated(empirical_coverage: float,
                  tolerance: float = COVERAGE_TOLERANCE) -> bool:
    """The single definition of "calibrated", used by scoring and by the tests.

    The epsilon is not pedantry. ``0.90 + 0.05`` is ``0.9500000000000001`` in
    binary floating point, so a model whose empirical coverage lands exactly on
    the stated tolerance boundary would be reported as *mis*calibrated by a
    naive ``<=``. Judging a model by an artefact of float representation is the
    kind of error that is invisible in a results table, so the rule lives in
    one function rather than being re-typed at each call site.
    """
    return abs(empirical_coverage - NOMINAL_COVERAGE) <= tolerance + 1e-9


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldPlan:
    """Walk-forward fold geometry for one dataset.

    Both window types score the *same* test windows -- that is a property of
    ``common/backtest.py``'s two split functions, and it is what makes an
    expanding-vs-rolling comparison a controlled experiment rather than two
    unrelated backtests. Only the training window's shape differs.
    """

    n_folds: int
    horizon: int
    min_train_size: int       # expanding: the floor the first fold must clear
    rolling_train_size: int   # rolling: the fixed window that slides forward

    def required_rows_expanding(self) -> int:
        return self.min_train_size + self.n_folds * self.horizon

    def required_rows_rolling(self) -> int:
        return self.rolling_train_size + self.n_folds * self.horizon

    def validate(self, n: int, label: str) -> None:
        """Fail loudly at configuration time, not mid-backtest."""
        if n < self.required_rows_expanding():
            raise ValueError(
                f"{label}: expanding plan needs >= "
                f"{self.required_rows_expanding()} rows "
                f"({self.min_train_size} min train + {self.n_folds} x "
                f"{self.horizon}), series has {n}"
            )
        if n < self.required_rows_rolling():
            raise ValueError(
                f"{label}: rolling plan needs >= "
                f"{self.required_rows_rolling()} rows "
                f"({self.rolling_train_size} fixed train + {self.n_folds} x "
                f"{self.horizon}), series has {n}"
            )

    def first_test_index(self, n: int) -> int:
        """Index of the first scored observation across the whole backtest.

        Both split helpers anchor their last test window at ``n``, so the
        scored region is always the final ``n_folds * horizon`` rows regardless
        of window type. Used to check that a structural break actually lands
        inside the scored region.
        """
        return n - self.n_folds * self.horizon


# --------------------------------------------------------------------------
# Dataset specifications
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about one of the four CSVs."""

    key: str
    filename: str
    date_col: str
    target_col: str
    freq: str                      # pandas offset alias
    seasonal_period: int           # m for SARIMA/ETS, sp for MASE
    group_cols: tuple[str, ...]    # () for single-series files
    folds: FoldPlan
    transform: str                 # "boxcox" | "log1p" | "none"
    scale_free_metric: str         # "mase" | "wape" -- the headline one
    description: str
    #: The one series carried through the deep single-series diagnostics.
    #: ``None`` for single-series files.
    primary_group: tuple[str, ...] | None = None
    #: Known structural break, as an ISO date. Drives the rolling-window
    #: argument and a pre/post error split in the report.
    structural_break: str | None = None
    #: Models deliberately NOT run here, and why. Recorded rather than silently
    #: skipped -- an omission a reader cannot see is indistinguishable from a
    #: bug.
    skip_models: dict[str, str] = field(default_factory=dict)

    @property
    def lags(self) -> tuple[int, ...]:
        return LAGS_MONTHLY if self.freq == "MS" else LAGS_DAILY

    @property
    def rolling_windows(self) -> tuple[int, ...]:
        return ROLL_MONTHLY if self.freq == "MS" else ROLL_DAILY

    @property
    def ljung_box_lags(self) -> int:
        return (LJUNG_BOX_LAGS_MONTHLY if self.freq == "MS"
                else LJUNG_BOX_LAGS_DAILY)

    @property
    def is_panel(self) -> bool:
        return bool(self.group_cols)


RETAIL = DatasetSpec(
    key="retail",
    filename="retail_demand.csv",
    date_col="date",
    target_col="units_sold",
    freq="D",
    seasonal_period=7,
    group_cols=("region", "category"),
    primary_group=("Riyadh", "Grocery"),
    # 1096 rows/series. Expanding: 730 (two full annual cycles, so every fold
    # has seen the yearly seasonality at least twice) + 6*14 = 814 <= 1096.
    # Rolling uses the same 730 so the two window types differ only in whether
    # older history is retained, not in how much history fold 0 gets.
    folds=FoldPlan(n_folds=6, horizon=14, min_train_size=730,
                   rolling_train_size=730),
    # Generated as level * trend * weekly * yearly * holiday * (1+promo) *
    # noise -- multiplicative throughout, with variance rising with level.
    # Box-Cox with lambda estimated per fold on the training window only.
    transform="boxcox",
    scale_free_metric="mase",
    description=(
        "Daily units sold, 3 regions x 2 categories, 2023-2025. Multiplicative "
        "weekly + yearly seasonality, an upward trend, drifting Hijri-like "
        "holiday bumps and sporadic promo shocks, Poisson-sampled."
    ),
)

WORKFORCE = DatasetSpec(
    key="workforce",
    filename="workforce_demand.csv",
    date_col="date",
    target_col="required_headcount",
    freq="D",
    seasonal_period=7,
    group_cols=(),
    # 731 rows. The break sits at index 456 (2025-04-01). To score a fold whose
    # TEST window straddles the break we need n_folds*horizon >= 731-456 = 275,
    # hence 20 x 14 = 280: the scored region starts at index 451, five days
    # before the break. Fewer folds would push the entire scored region past
    # the break and quietly test nothing about it.
    folds=FoldPlan(n_folds=20, horizon=14, min_train_size=451,
                   rolling_train_size=365),
    transform="boxcox",
    scale_free_metric="mase",
    description=(
        "Daily contact-centre headcount requirement, 2024-2025, one series, "
        "with a sustained x1.35 step-change on 2025-04-01."
    ),
    structural_break="2025-04-01",
)

ECONOMIC = DatasetSpec(
    key="economic",
    filename="economic_indicator.csv",
    date_col="month",
    target_col="activity_index",
    freq="MS",
    seasonal_period=12,
    group_cols=(),
    # Only 108 rows, total. Expanding: 72 (six years, so MASE at sp=12 is
    # always defined and the 42-month cycle is seen at least twice) + 6*6 = 108
    # exactly. Rolling has to drop to 60 to fit at all -- that five-year window
    # is itself part of the finding, not a tuning choice.
    folds=FoldPlan(n_folds=6, horizon=6, min_train_size=72,
                   rolling_train_size=60),
    # Additive by construction: trend + cycle + shock + Gaussian noise, with no
    # level-dependent variance to stabilise.
    transform="none",
    scale_free_metric="mase",
    description=(
        "Monthly non-oil activity index, 2017-2025, 108 points. Linear trend, "
        "a 42-month business cycle plus an 11-month harmonic, and a "
        "2020-shaped shock with partial exponential recovery."
    ),
    skip_models={
        "lgbm_global": (
            "Only one series -- a global model across series has nothing to "
            "pool. The per-series LightGBM is the honest comparison here."
        ),
    },
)

INTERMITTENT = DatasetSpec(
    key="intermittent",
    filename="intermittent_demand.csv",
    date_col="date",
    target_col="units_ordered",
    freq="D",
    seasonal_period=7,
    group_cols=("sku",),
    primary_group=("SKU-C3087",),   # the densest SKU: 8.3% non-zero
    # 731 rows/SKU. 500 + 8*14 = 612 <= 731.
    folds=FoldPlan(n_folds=8, horizon=14, min_train_size=500,
                   rolling_train_size=365),
    # ~95% zeros: a log or Box-Cox transform is undefined or degenerate, and
    # there is no variance-stabilising win to be had on a series that is mostly
    # a single repeated value.
    transform="none",
    # MASE's denominator is the in-sample seasonal-naive MAE, which on a
    # 95%-zero series is itself tiny and unstable. WAPE is the headline.
    scale_free_metric="wape",
    description=(
        "Daily spare-parts demand, 4 low-turnover SKUs, 2024-2025. 91.7%-97.9% "
        "zero rows depending on SKU; non-zero magnitudes are near-identical "
        "across SKUs, so essentially all the signal is *whether* an order "
        "occurs, not how large it is."
    ),
)

DATASETS: dict[str, DatasetSpec] = {
    d.key: d for d in (RETAIL, WORKFORCE, ECONOMIC, INTERMITTENT)
}


# --------------------------------------------------------------------------
# Model hyperparameters
# --------------------------------------------------------------------------

#: SARIMA order grid searched by AIC on each fold's training window.
#: statsmodels ships no ``auto_arima`` (that is a pmdarima/sktime feature and
#: outside this course's four tools -- see reference/tooling_guide.qmd), so the
#: grid is small, explicit, and justified from the ACF/PACF read rather than
#: exhaustive. Searching it per fold costs a few seconds; searching it once
#: over the whole series would leak test-window information into the order
#: choice, the subtler cousin of the scaler-fit-globally mistake.
SARIMA_ORDERS = [
    (1, 0, 0), (0, 0, 1), (1, 0, 1),
    (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
]

#: Seasonal orders, with ``None`` in the period slot filled at fit time from
#: the dataset's ``seasonal_period``.
SARIMA_SEASONAL_ORDERS = [
    (0, 0, 0, 0),
    (1, 0, 0, None),
    (0, 1, 1, None),
    (1, 1, 1, None),
]

#: Exponential-smoothing configurations, selected by AIC per fold -- the same
#: rule as SARIMA, so the two classical families are chosen comparably.
ETS_CONFIGS = [
    {"name": "SES", "trend": None, "seasonal": None, "damped_trend": False},
    {"name": "Holt", "trend": "add", "seasonal": None, "damped_trend": False},
    {"name": "Holt-damped", "trend": "add", "seasonal": None,
     "damped_trend": True},
    {"name": "Holt-Winters-add", "trend": "add", "seasonal": "add",
     "damped_trend": False},
    {"name": "Holt-Winters-damped", "trend": "add", "seasonal": "add",
     "damped_trend": True},
]

#: LightGBM. Deliberately modest: the fold budget is 20 folds x 2 window types
#: x 3 quantile/point models on the workforce series alone, and a larger
#: ensemble buys accuracy this comparison is not actually about.
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": RNG_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

#: Lag features, in observations, chosen per frequency: the daily sets get
#: within-week, week-over-week and four-week-ago lags; monthly gets
#: within-quarter and year-over-year.
LAGS_DAILY = (1, 2, 3, 7, 14, 21, 28)
LAGS_MONTHLY = (1, 2, 3, 6, 12)

#: Rolling-window widths for mean/std/min/max features, in observations.
ROLL_DAILY = (7, 14, 28)
ROLL_MONTHLY = (3, 6, 12)

#: Ljung-Box lag counts for residual diagnostics, by frequency -- roughly 2x
#: the seasonal period, the usual rule for a seasonal series.
LJUNG_BOX_LAGS_DAILY = 14
LJUNG_BOX_LAGS_MONTHLY = 24

#: Significance level for ADF and Ljung-Box.
ALPHA = 0.05

#: Fraction of each fold's training window held out, at its end, as the
#: conformal calibration set. 20% of a 730-day window is ~146 days -- long
#: enough for a stable residual quantile, short enough that the point model
#: still trains on four fifths of its history. Recomputed inside every fold, as
#: day3/05_probabilistic_forecasting.qmd requires for a series that may have
#: shifted regime.
CONFORMAL_CALIB_FRACTION = 0.20

#: The repository this project is published from. Used by the notebook's
#: ``fetch()`` helper so the same notebook runs from a clone or cold on Colab.
GITHUB_REPO = "Abdullah-FZN/calibrated-forecast-backtesting"
GITHUB_BRANCH = "main"

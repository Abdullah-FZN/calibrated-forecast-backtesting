"""Data ingestion and target transforms.

Two responsibilities, kept together because they are the two places a time
series pipeline most easily goes wrong before a single model is fit:

* **Ingestion** -- getting a regularly-spaced, gap-free, time-ordered series
  out of a long-format CSV, and *proving* it is regular rather than assuming
  it (``SeriesBundle.validate``). A silently missing date turns every lag
  feature downstream into a lag of the wrong length.
* **Target transforms** -- Box-Cox / log, with the parameter estimated on a
  training window and nothing else. The lambda that stabilises variance is a
  *fitted statistic*; estimating it once over the full series and reusing it
  inside a backtest is exactly the "global scaler" leak that
  day2/04_backtesting.qmd names, just wearing a different hat.

The transform API deliberately separates two inverse operations that are often
conflated:

``inverse_quantile``
    For interval bounds and quantiles. Box-Cox and log are strictly monotone,
    so a quantile maps through them exactly -- the back-transformed 95th
    percentile *is* the 95th percentile.

``inverse_mean``
    For point forecasts. A model fit in transformed space predicts the
    conditional *mean of the transformed variable*, which back-transforms to
    roughly the conditional *median* of the original variable, not its mean.
    On a right-skewed demand series that is a systematic low bias. The
    correction here is the standard second-order (delta-method) term, applied
    only when we have an honest in-sample residual variance to apply it with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import config


# ==========================================================================
# Ingestion
# ==========================================================================

@dataclass(frozen=True)
class SeriesBundle:
    """One univariate series, ready to backtest.

    Attributes
    ----------
    key : str
        Dataset key, e.g. ``"retail"``.
    group : tuple[str, ...]
        Identifying values for the group columns, ``()`` for single-series
        files. ``("Riyadh", "Grocery")`` for the retail primary series.
    dates : pd.DatetimeIndex
        Strictly increasing, gap-free at the dataset's declared frequency.
    values : np.ndarray
        Float target values, aligned with ``dates``.
    spec : config.DatasetSpec
    """

    key: str
    group: tuple[str, ...]
    dates: pd.DatetimeIndex
    values: np.ndarray
    spec: "config.DatasetSpec"

    @property
    def label(self) -> str:
        return "/".join(self.group) if self.group else self.key

    @property
    def n(self) -> int:
        return len(self.values)

    def validate(self) -> None:
        """Assert the series is actually regular before anything relies on it.

        A lag-7 feature means "one week ago" only if the rows really are one
        day apart. This check is cheap and catches the failure at load time
        rather than as an unexplained accuracy drop.
        """
        if len(self.dates) != len(self.values):
            raise ValueError(f"{self.label}: dates/values length mismatch")
        if not self.dates.is_monotonic_increasing:
            raise ValueError(f"{self.label}: dates are not sorted ascending")
        if self.dates.has_duplicates:
            raise ValueError(f"{self.label}: duplicate dates")
        expected = pd.date_range(
            self.dates[0], self.dates[-1], freq=self.spec.freq
        )
        if len(expected) != len(self.dates) or not expected.equals(self.dates):
            missing = expected.difference(self.dates)
            raise ValueError(
                f"{self.label}: irregular index at freq={self.spec.freq!r} -- "
                f"{len(missing)} missing period(s), first few: "
                f"{list(missing[:5])}"
            )
        if np.isnan(self.values).any():
            raise ValueError(f"{self.label}: target contains NaN")

    def index_of(self, when: str | pd.Timestamp) -> int:
        """Positional index of a date, for locating a structural break."""
        return int(self.dates.get_loc(pd.Timestamp(when)))

    def describe(self) -> dict:
        v = self.values
        nonzero = v[v > 0]
        return {
            "dataset": self.key,
            "series": self.label,
            "n": self.n,
            "start": str(self.dates[0].date()),
            "end": str(self.dates[-1].date()),
            "freq": self.spec.freq,
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "min": float(v.min()),
            "max": float(v.max()),
            "zero_rate": float((v == 0).mean()),
            "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
            # Coefficient of variation of the first vs last third of the
            # series: a crude but honest read on whether spread grows with
            # level, which is what a variance-stabilising transform is for.
            "cv_first_third": _cv(v[: self.n // 3]),
            "cv_last_third": _cv(v[-(self.n // 3):]),
        }


def _cv(x: np.ndarray) -> float:
    m = float(np.mean(x))
    return float(np.std(x, ddof=1) / m) if m else float("nan")


def load_raw(spec: "config.DatasetSpec") -> pd.DataFrame:
    """Read one dataset CSV with its date column parsed."""
    path = config.DATA_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_series.py` (the "
            "generator is seeded and deterministic) or copy the CSVs from the "
            "course repository."
        )
    df = pd.read_csv(path)
    df[spec.date_col] = pd.to_datetime(df[spec.date_col])
    return df.sort_values([*spec.group_cols, spec.date_col]).reset_index(
        drop=True
    )


def iter_series(spec: "config.DatasetSpec") -> list[SeriesBundle]:
    """Split one dataset into its constituent univariate series."""
    df = load_raw(spec)
    bundles: list[SeriesBundle] = []
    if not spec.group_cols:
        bundles.append(
            SeriesBundle(
                key=spec.key,
                group=(),
                dates=pd.DatetimeIndex(df[spec.date_col]),
                values=df[spec.target_col].to_numpy(dtype=float),
                spec=spec,
            )
        )
    else:
        for keys, sub in df.groupby(list(spec.group_cols), sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            sub = sub.sort_values(spec.date_col)
            bundles.append(
                SeriesBundle(
                    key=spec.key,
                    group=tuple(str(k) for k in keys),
                    dates=pd.DatetimeIndex(sub[spec.date_col]),
                    values=sub[spec.target_col].to_numpy(dtype=float),
                    spec=spec,
                )
            )
    for b in bundles:
        b.validate()
        spec.folds.validate(b.n, f"{spec.key}:{b.label}")
    return bundles


def get_series(spec: "config.DatasetSpec",
               group: tuple[str, ...] | None = None) -> SeriesBundle:
    """Fetch one named series, defaulting to the dataset's primary group."""
    target = group if group is not None else (spec.primary_group or ())
    for b in iter_series(spec):
        if b.group == tuple(target):
            return b
    raise KeyError(f"{spec.key}: no series {target!r}")


def load_panel(spec: "config.DatasetSpec") -> pd.DataFrame:
    """Long-format frame for global (cross-series) modelling.

    Columns: ``ds``, ``series_id``, ``y``. The retail and intermittent CSVs
    already arrive in this shape; this just normalises the column names so a
    global LightGBM model does not need to know which dataset it is looking at.
    """
    df = load_raw(spec)
    out = pd.DataFrame({
        "ds": df[spec.date_col],
        "y": df[spec.target_col].astype(float),
    })
    if spec.group_cols:
        out["series_id"] = (
            df[list(spec.group_cols)].astype(str).agg("/".join, axis=1)
        )
    else:
        out["series_id"] = spec.key
    return out.sort_values(["series_id", "ds"]).reset_index(drop=True)


# ==========================================================================
# Target transforms
# ==========================================================================

class TargetTransform:
    """Base class. A transform is *fit on a training window only*.

    Calling ``transform``/``inverse_*`` before ``fit`` raises, so a transform
    accidentally shared across folds cannot silently carry another fold's
    parameters. ``fitted_on`` records the length of the window it was fit on --
    cheap provenance that the leakage audit in ``features.py`` asserts against.
    """

    name = "none"

    def __init__(self) -> None:
        self._fitted = False
        self.fitted_on: int | None = None

    def fit(self, y_train: np.ndarray) -> "TargetTransform":
        self._fitted = True
        self.fitted_on = len(y_train)
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} used before fit() -- a transform must "
                "be fit on this fold's training window before it touches any "
                "data"
            )

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(y, dtype=float)

    def inverse_quantile(self, z: np.ndarray) -> np.ndarray:
        """Exact inverse, for quantiles and interval bounds."""
        self._check()
        return np.asarray(z, dtype=float)

    def inverse_mean(self, z: np.ndarray,
                     resid_var: float | None = None) -> np.ndarray:
        """Inverse for a point forecast, optionally bias-corrected."""
        return self.inverse_quantile(z)

    def params(self) -> dict:
        return {"transform": self.name}


class IdentityTransform(TargetTransform):
    """No transform. The right choice for an additive series, and the only
    valid choice for one containing zeros."""

    name = "none"


class Log1pTransform(TargetTransform):
    """``log1p`` / ``expm1``. No fitted parameter, but still fold-scoped so the
    call sites stay uniform."""

    name = "log1p"

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check()
        y = np.asarray(y, dtype=float)
        if (y < 0).any():
            raise ValueError("log1p transform requires y >= 0")
        return np.log1p(y)

    def inverse_quantile(self, z: np.ndarray) -> np.ndarray:
        self._check()
        return np.expm1(np.asarray(z, dtype=float))

    def inverse_mean(self, z: np.ndarray,
                     resid_var: float | None = None) -> np.ndarray:
        self._check()
        z = np.asarray(z, dtype=float)
        if resid_var is None:
            return np.expm1(z)
        # E[exp(Z)] = exp(mu + sigma^2/2) for Z ~ N(mu, sigma^2).
        return np.expm1(z + resid_var / 2.0)


def guerrero_lambda(y: np.ndarray, period: int,
                    bounds: tuple[float, float] = (-1.0, 1.5)) -> float:
    """Guerrero's (1993) variance-stabilising Box-Cox lambda.

    Why not ``scipy.stats.boxcox``'s maximum-likelihood lambda? Because ML
    picks the lambda that makes the *marginal* distribution of the series most
    normal, and on a strongly seasonal series that marginal is a mixture across
    weekdays and seasons -- so ML is answering a question we did not ask. Run
    against this course's retail series it returns a lambda below -1 (a
    reciprocal transform), which flattens the right tail beautifully and
    destroys the additive seasonal structure a SARIMA or ETS model is trying to
    fit.

    Guerrero instead picks the lambda that makes the *spread* of the series
    independent of its *level*, which is precisely what a variance-stabilising
    transform is for. Split the window into contiguous subseries one seasonal
    period long, take each one's mean ``m_i`` and standard deviation ``s_i``,
    and minimise the coefficient of variation of ``s_i / m_i**(1 - lambda)``.

    For a purely multiplicative series (spread proportional to level, which is
    how ``data/generate_series.py`` builds the retail and workforce series) the
    minimiser is lambda = 0 -- the log transform -- recovered from the data
    rather than assumed.
    """
    y = np.asarray(y, dtype=float)
    n_groups = len(y) // period
    if n_groups < 2:
        return 0.0
    # Use the most recent complete groups; a ragged leading remainder would
    # bias the first group's statistics.
    trimmed = y[len(y) - n_groups * period:].reshape(n_groups, period)
    means = trimmed.mean(axis=1)
    stds = trimmed.std(axis=1, ddof=1)
    ok = (means > 0) & np.isfinite(stds)
    if ok.sum() < 2:
        return 0.0
    means, stds = means[ok], stds[ok]

    def dispersion(lmbda: float) -> float:
        ratio = stds / np.power(means, 1.0 - lmbda)
        m = ratio.mean()
        if not np.isfinite(m) or m == 0:
            return np.inf
        return float(ratio.std(ddof=1) / m)

    grid = np.linspace(bounds[0], bounds[1], 201)
    scores = np.array([dispersion(l) for l in grid])
    return float(grid[int(np.argmin(scores))])


class BoxCoxTransform(TargetTransform):
    """Box-Cox with lambda estimated on the training window only.

    ``method="guerrero"`` (the default) selects lambda by variance
    stabilisation -- see :func:`guerrero_lambda` for why that beats the
    maximum-likelihood lambda on a seasonal series. ``method="mle"`` is kept so
    the notebook can show the two side by side rather than just asserting the
    choice.

    Box-Cox is defined for strictly positive input, so a shift is applied when
    the training window contains non-positive values -- and that shift, like
    lambda, is a training-window statistic.
    """

    name = "boxcox"

    def __init__(self, period: int = 1, method: str = "guerrero",
                 lambda_bounds: tuple[float, float] = (-1.0, 1.5),
                 inverse_clip_multiple: float = 10.0):
        super().__init__()
        self.lmbda: float | None = None
        self.shift: float = 0.0
        self.period = period
        self.method = method
        self.lambda_bounds = lambda_bounds
        #: Back-transformed values above this multiple of the training
        #: window's maximum are treated as numerical artefacts and clipped.
        #: Ten times the largest value ever observed is far outside any
        #: plausible demand forecast while still leaving interval upper bounds
        #: room to be genuinely wide.
        self.inverse_clip_multiple = inverse_clip_multiple
        self.train_max: float = float("inf")
        self.train_min: float = 0.0
        self.n_inverse_clipped: int = 0

    def fit(self, y_train: np.ndarray) -> "BoxCoxTransform":
        y = np.asarray(y_train, dtype=float)
        # Shift into strictly-positive territory if needed, using only this
        # window's minimum.
        self.shift = 0.0 if y.min() > 0 else float(1.0 - y.min())
        # Training-window scale, kept so the inverse can recognise a
        # back-transformed value that is physically impossible. Like lambda and
        # the shift, these are training-window statistics.
        self.train_max = float(y.max())
        self.train_min = float(y.min())
        self.n_inverse_clipped = 0
        shifted = y + self.shift
        if self.method == "mle":
            _, lmbda = stats.boxcox(shifted)
        else:
            lmbda = guerrero_lambda(shifted, self.period, self.lambda_bounds)
        # Clamp: an extreme lambda on a short window (the economic series'
        # 60-row rolling window) is numerically brittle for no accuracy gain.
        self.lmbda = float(np.clip(lmbda, *self.lambda_bounds))
        super().fit(y_train)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check()
        shifted = np.asarray(y, dtype=float) + self.shift
        shifted = np.maximum(shifted, 1e-9)
        if abs(self.lmbda) < 1e-8:
            return np.log(shifted)
        return (np.power(shifted, self.lmbda) - 1.0) / self.lmbda

    def inverse_quantile(self, z: np.ndarray) -> np.ndarray:
        """Exact inverse, guarded against the negative-lambda singularity.

        **Why the guard is not optional.** For ``lambda < 0`` the Box-Cox
        transform maps ``y in (0, inf)`` onto a *bounded* interval: as
        ``y -> inf``, ``z -> -1/lambda``. Any predicted ``z`` at or past that
        limit has no finite pre-image. Clamping ``lambda*z + 1`` to a small
        positive epsilon and raising it to ``1/lambda`` (a negative power) does
        not fail loudly -- it returns something like 1e17 and sails on.

        That is not hypothetical. On the retail series (lambda = -0.52, limit
        z = 1.905) one global-LightGBM fold predicted past the limit and
        produced a fold MAE of 6.4e8 on a series averaging 560 units, which
        propagated into a mean WAPE change of +1.6 million percent in the
        expanding-vs-rolling table. It looked like a catastrophic model
        failure; it was arithmetic.

        A forecast outside the transform's representable range is a numerical
        artefact, not a prediction, so it is clipped to a generous multiple of
        the training window's own maximum and the event is counted. Clipping
        is recorded rather than silent: a transform that clips often is telling
        you the model is extrapolating somewhere it should not be trusted.
        """
        self._check()
        z = np.asarray(z, dtype=float)
        if abs(self.lmbda) < 1e-8:
            out = np.exp(np.minimum(z, 700.0))       # exp overflows past ~709
        else:
            base = self.lmbda * z + 1.0
            if self.lmbda < 0:
                # Keep base strictly inside the valid region. The floor is set
                # by the largest value we are willing to report, not by an
                # arbitrary epsilon.
                ceiling = self.train_max * self.inverse_clip_multiple + self.shift
                base_floor = float(np.power(ceiling, self.lmbda))
                self.n_inverse_clipped += int(np.sum(base < base_floor))
                base = np.maximum(base, base_floor)
            else:
                base = np.maximum(base, 1e-9)
            out = np.power(base, 1.0 / self.lmbda)

        out = out - self.shift
        cap = self.train_max * self.inverse_clip_multiple
        over = out > cap
        if np.any(over):
            self.n_inverse_clipped += int(np.sum(over))
            out = np.minimum(out, cap)
        return out

    def inverse_mean(self, z: np.ndarray,
                     resid_var: float | None = None) -> np.ndarray:
        """Second-order back-transform correction.

        For ``y = g(z)`` with ``z ~ (mu, s2)``, ``E[y] ~= g(mu) + g''(mu) s2/2``.
        With ``g(z) = (lam z + 1)^(1/lam)`` this gives the standard Box-Cox
        retransformation term. Omitted (returning the median) when no residual
        variance is supplied, because a made-up variance is worse than a
        documented median.
        """
        self._check()
        median = self.inverse_quantile(z)
        if resid_var is None or resid_var <= 0:
            return median
        z = np.asarray(z, dtype=float)
        if abs(self.lmbda) < 1e-8:
            return np.exp(z + resid_var / 2.0) - self.shift
        base = np.maximum(self.lmbda * z + 1.0, 1e-9)
        lam = self.lmbda
        # g''(z) = (1-lam) * lam^... -- written out rather than simplified so
        # it can be checked against a textbook.
        second = (1.0 / lam) * (1.0 / lam - 1.0) * lam**2 * np.power(
            base, 1.0 / lam - 2.0
        )
        return median + 0.5 * second * resid_var

    def params(self) -> dict:
        return {"transform": self.name, "method": self.method,
                "lambda": self.lmbda, "shift": self.shift,
                "train_max": self.train_max,
                "inverse_clipped": self.n_inverse_clipped}


def make_transform(kind: str, period: int = 1, **kwargs) -> TargetTransform:
    """Factory. A fresh, unfitted instance every call -- fold-scoped by
    construction, so there is no object for two folds to share."""
    if kind == "none":
        return IdentityTransform()
    if kind == "log1p":
        return Log1pTransform()
    if kind == "boxcox":
        return BoxCoxTransform(period=period, **kwargs)
    raise ValueError(f"unknown transform {kind!r}")

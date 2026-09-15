"""Figures for the capstone report.

Design rules applied throughout, so they are not re-litigated per chart:

* **Colour carries model family, never rank.** Four groups -- baseline,
  classical, GAM, ML -- get a fixed assignment that does not change when a
  filter drops a model. The baseline takes neutral grey (it is a reference, not
  a competitor); the three real families take categorical slots 1-3, the subset
  documented as clearing the all-pairs colour-vision separation floors in both
  light and dark mode. A fourth categorical hue would put yellow next to orange
  and fail that check, which is why family, not model, is the colour dimension:
  seven models would have needed seven hues.
* **One y-axis, always.** Two measures of different scale get two panels, never
  a second axis.
* **Thin marks, solid hairline grid, recessive axes.** Dashed gridlines read as
  "threshold" when they are just a grid; the one dashed line in this module is
  an actual threshold (the nominal coverage target) and is labelled as such.
* **Legend whenever two or more series are drawn**, so identity is never
  carried by colour alone; direct labels are used selectively, never a number
  on every point.
* Every figure has a table equivalent under ``outputs/tables/`` -- these PNGs
  are a reading aid for the report, not the only representation of the numbers.

Figures render on an explicit light surface rather than a transparent one:
a transparent PNG inherits GitHub's dark theme behind dark ink and becomes
unreadable, so the surface is drawn deliberately.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import config  # noqa: E402

sys.path.insert(0, str(config.COMMON_DIR))
from metrics import coverage as _coverage  # noqa: E402

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots 1-3 plus neutral. See the module docstring for why the
#: colour dimension is family rather than model.
FAMILY_COLOR = {
    "baseline": INK_MUTED,
    "classical": "#2a78d6",   # slot 1, blue
    "gam": "#eb6834",         # slot 2, orange
    "ml": "#1baf7a",          # slot 3, aqua
}
#: Secondary encoding so family survives greyscale printing and the
#: all-pairs case, per the accessibility rule.
FAMILY_MARKER = {
    "baseline": "s", "classical": "o", "gam": "^", "ml": "D"}

STATUS_GOOD = "#006300"
STATUS_BAD = "#e34948"

LINE_W = 1.6
MARKER_SZ = 6


def apply_style() -> None:
    """Global matplotlib defaults matching the tokens above."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",      # solid hairline, never dashed
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def _finish(fig, path_name: str) -> str:
    """Save under outputs/figures and return the repo-relative path."""
    out = config.FIGURE_DIR / path_name
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(config.ROOT)).replace("\\", "/")


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _fam(row_family: str) -> str:
    return FAMILY_COLOR.get(row_family, INK_SECONDARY)


# --------------------------------------------------------------------------
# 1. Series overview
# --------------------------------------------------------------------------

def plot_series_overview(bundle, fname: str, break_date: str | None = None,
                         scored_from: int | None = None) -> str:
    """The raw series, with the scored backtest region and any break marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(bundle.dates, bundle.values, color=FAMILY_COLOR["classical"],
            linewidth=1.0)
    if scored_from is not None and scored_from < bundle.n:
        ax.axvspan(bundle.dates[scored_from], bundle.dates[-1],
                   color=INK_MUTED, alpha=0.10, linewidth=0)
        ax.text(bundle.dates[scored_from], ax.get_ylim()[1],
                "  backtest region", va="top", ha="left", fontsize=8,
                color=INK_SECONDARY)
    if break_date:
        bd = pd.Timestamp(break_date)
        ax.axvline(bd, color=STATUS_BAD, linewidth=1.2, linestyle="-")
        ax.text(bd, ax.get_ylim()[1], f" break {break_date} ", va="top",
                ha="left", fontsize=8, color=STATUS_BAD)
    ax.set_title(f"{bundle.key} — {bundle.label}")
    ax.set_ylabel(bundle.spec.target_col)
    _despine(ax)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 2. Decomposition
# --------------------------------------------------------------------------

def plot_decomposition(bundle, decomp, fname: str) -> str:
    """Observed / trend / seasonal / residual as small multiples.

    Four panels rather than four overlaid series: the components have
    genuinely different scales, and overlaying them would need the dual axis
    this project does not use.
    """
    apply_style()
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    panels = [
        ("Observed", bundle.values, FAMILY_COLOR["classical"]),
        ("Trend", decomp.trend, FAMILY_COLOR["gam"]),
        (f"Seasonal (period {decomp.period})", decomp.seasonal,
         FAMILY_COLOR["ml"]),
        ("Residual", decomp.resid, INK_MUTED),
    ]
    for ax, (title, vals, color) in zip(axes, panels):
        if title.startswith("Residual"):
            ax.axhline(0, color=AXIS, linewidth=0.8)
            ax.plot(bundle.dates, vals, color=color, linewidth=0.7,
                    marker="", alpha=0.9)
        else:
            ax.plot(bundle.dates, vals, color=color, linewidth=1.0)
        ax.set_ylabel(title, fontsize=8, color=INK_SECONDARY)
        _despine(ax)
    axes[0].set_title(
        f"{decomp.method} decomposition — {bundle.label}   "
        f"(trend strength {decomp.trend_strength:.2f}, "
        f"seasonal strength {decomp.seasonal_strength:.2f})"
    )
    fig.align_ylabels(axes)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 3. ACF / PACF
# --------------------------------------------------------------------------

def plot_acf_pacf(ac, seasonal_period: int, fname: str, label: str) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, vals, ci, title in (
        (axes[0], ac.acf, ac.confint_acf, "ACF"),
        (axes[1], ac.pacf, ac.confint_pacf, "PACF"),
    ):
        lags = np.arange(len(vals))
        band = ci[:, 1] - vals
        ax.fill_between(lags, -band, band, color=FAMILY_COLOR["classical"],
                        alpha=0.12, linewidth=0)
        ax.vlines(lags, 0, vals, color=FAMILY_COLOR["classical"],
                  linewidth=1.0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        # Direct-label only the seasonal multiples -- the point the plot is
        # being read for -- rather than every lag.
        if seasonal_period > 1:
            for k in range(seasonal_period, len(vals), seasonal_period):
                ax.plot([k], [vals[k]], marker="o", markersize=4,
                        color=FAMILY_COLOR["gam"], zorder=3)
        ax.set_title(title)
        ax.set_xlabel("lag")
        _despine(ax)
    axes[0].set_ylabel("correlation")
    fig.suptitle(
        f"{label} — orange dots mark multiples of the seasonal period "
        f"({seasonal_period})", fontsize=9, color=INK_SECONDARY, y=1.04)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 4. Variance stabilisation
# --------------------------------------------------------------------------

def plot_transform_effect(y_raw, y_tf, period: int, lmbda: float,
                          fname: str, label: str) -> str:
    """Level vs spread, before and after the transform.

    The claim a variance-stabilising transform makes is exactly that the right
    panel's cloud is flat. Two panels, same axes roles, no dual axis.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, vals, title, color in (
        (axes[0], np.asarray(y_raw, dtype=float), "Raw scale",
         FAMILY_COLOR["classical"]),
        (axes[1], np.asarray(y_tf, dtype=float),
         f"Box–Cox (λ = {lmbda:+.2f})", FAMILY_COLOR["ml"]),
    ):
        n = len(vals) // period
        blocks = vals[len(vals) - n * period:].reshape(n, period)
        m, s = blocks.mean(axis=1), blocks.std(axis=1, ddof=1)
        ax.plot(m, s, linestyle="", marker="o", markersize=4,
                color=color, alpha=0.65,
                markeredgecolor=SURFACE, markeredgewidth=0.5)
        if len(m) > 2:
            r = float(np.corrcoef(m, s)[0, 1])
            fit = np.polyfit(m, s, 1)
            xs = np.linspace(m.min(), m.max(), 20)
            ax.plot(xs, np.polyval(fit, xs), color=INK_SECONDARY,
                    linewidth=1.0)
            ax.set_title(f"{title}   r = {r:+.3f}")
        ax.set_xlabel(f"mean of each {period}-step block")
        _despine(ax)
    axes[0].set_ylabel("std. dev. of block")
    fig.suptitle(f"Level–spread dependence — {label}", fontsize=9,
                 color=INK_SECONDARY, y=1.03)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 5. Fold layout
# --------------------------------------------------------------------------

def plot_fold_layout(bundle, splits_by_type: dict, fname: str,
                     break_date: str | None = None) -> str:
    """Train/test windows per fold, expanding vs rolling, on one time axis.

    This is the figure that makes "the test windows are identical; only the
    training window differs" checkable rather than asserted.
    """
    apply_style()
    n_types = len(splits_by_type)
    fig, axes = plt.subplots(n_types, 1, figsize=(10, 2.0 + 1.8 * n_types),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (wt, splits) in zip(axes, splits_by_type.items()):
        for i, (tr, te) in enumerate(splits):
            ax.barh(i, (bundle.dates[tr.stop - 1] - bundle.dates[tr.start]).days,
                    left=bundle.dates[tr.start], height=0.62,
                    color=FAMILY_COLOR["classical"], alpha=0.35, linewidth=0)
            ax.barh(i, (bundle.dates[te.stop - 1] - bundle.dates[te.start]).days,
                    left=bundle.dates[te.start], height=0.62,
                    color=FAMILY_COLOR["gam"], linewidth=0)
        if break_date:
            ax.axvline(pd.Timestamp(break_date), color=STATUS_BAD,
                       linewidth=1.2)
        ax.set_ylabel(f"{wt}\nfold", fontsize=8, color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis="y", visible=False)
        _despine(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR["classical"],
                      alpha=0.35, label="train window"),
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR["gam"],
                      label="test window"),
    ]
    if break_date:
        handles.append(plt.Line2D([0], [0], color=STATUS_BAD, linewidth=1.2,
                                  label=f"structural break {break_date}"))
    axes[0].legend(handles=handles, loc="upper left", ncol=3,
                   bbox_to_anchor=(0, 1.35))
    axes[0].set_title(f"Walk-forward fold layout — {bundle.label}", loc="left")
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 6. Per-fold error
# --------------------------------------------------------------------------

def plot_per_fold_metric(per_fold: pd.DataFrame, metric: str, fname: str,
                         title: str, break_fold: int | None = None,
                         window_type: str | None = None) -> str:
    """One line per model, one point per fold.

    day2/04_backtesting.qmd's whole argument is that a mean across folds hides
    the fold that blew up, so the per-fold trace is plotted and the mean is
    left to the table.
    """
    apply_style()
    df = per_fold.copy()
    if window_type:
        df = df[df["window_type"] == window_type]
    df = df[df[metric].notna()]
    wts = sorted(df["window_type"].unique())
    fig, axes = plt.subplots(len(wts), 1, figsize=(10, 2.6 * len(wts)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, wt in zip(axes, wts):
        sub = df[df["window_type"] == wt]
        for model, g in sub.groupby("model"):
            fam = g["family"].iloc[0]
            ax.plot(g["fold"], g[metric], color=_fam(fam),
                    marker=FAMILY_MARKER.get(fam, "o"),
                    markersize=MARKER_SZ - 1, linewidth=LINE_W, alpha=0.85,
                    markeredgecolor=SURFACE, markeredgewidth=0.6,
                    label=model)
        if break_fold is not None:
            ax.axvline(break_fold, color=STATUS_BAD, linewidth=1.2)
            ax.text(break_fold, ax.get_ylim()[1], " break fold", va="top",
                    fontsize=8, color=STATUS_BAD)
        ax.set_ylabel(f"{metric}\n({wt})", fontsize=8, color=INK_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _despine(ax)
    axes[-1].set_xlabel("fold")
    axes[0].set_title(title, loc="left")
    axes[0].legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.42))
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 7. Calibration: coverage vs width
# --------------------------------------------------------------------------

def plot_calibration(pooled: pd.DataFrame, fname: str, title: str) -> str:
    """Coverage against interval width -- the pair, never coverage alone.

    The nominal level is a labelled threshold line and the tolerance band is
    shaded, so "calibrated" is read off the chart rather than asserted. Width
    is normalised by each series' own mean level so points from different
    datasets are comparable on one axis.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    nominal = config.NOMINAL_COVERAGE
    tol = config.COVERAGE_TOLERANCE

    ax.axhspan(nominal - tol, nominal + tol, color=STATUS_GOOD, alpha=0.07,
               linewidth=0)
    ax.axhline(nominal, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    ax.text(ax.get_xlim()[1], nominal, f" nominal {nominal:.0%} ",
            va="bottom", ha="right", fontsize=8, color=INK_SECONDARY)

    seen = set()
    for _, r in pooled.iterrows():
        fam = r.get("family", "classical")
        lbl = fam if fam not in seen else None
        seen.add(fam)
        ax.plot(r["width_pct_of_mean"], r["coverage"],
                marker=FAMILY_MARKER.get(fam, "o"), markersize=MARKER_SZ + 2,
                color=_fam(fam), linestyle="",
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=lbl)
    # Direct-label only the extremes: the widest, and the worst-calibrated.
    if len(pooled):
        notable = pd.concat([
            pooled.nlargest(1, "width_pct_of_mean"),
            pooled.loc[[(pooled["coverage"] - nominal).abs().idxmax()]],
        ]).drop_duplicates(subset=["model", "dataset"])
        for _, r in notable.iterrows():
            ax.annotate(f"{r['model']}", (r["width_pct_of_mean"],
                                          r["coverage"]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("mean interval width, % of the series' own mean level "
                  "(narrower is better)")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.03)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", title="family", title_fontsize=8)
    _despine(ax)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 8. Forecast fan
# --------------------------------------------------------------------------

def plot_forecast_fan(outcomes_by_model: dict, bundle, fold: int,
                      fname: str, context_multiple: float = 4.0) -> str:
    """Actual vs forecast with the interval band, for one fold.

    One panel per model rather than seven bands overlaid -- overlapping
    translucent bands are unreadable and would need more categorical hues than
    the palette validates for.

    Context history is sized as a multiple of the *horizon*, not a fixed number
    of observations: 60 points of context is one sensible screen on a daily
    series and five years of squashed history on a monthly one, which leaves
    the forecast window -- the entire subject of the chart -- occupying a
    tenth of the width.
    """
    apply_style()
    models_ = list(outcomes_by_model)
    n = len(models_)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 1.9 * n), sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)
    horizon = len(outcomes_by_model[models_[0]].y_true)
    context = int(round(context_multiple * horizon))
    for ax, name in zip(axes, models_):
        o = outcomes_by_model[name]
        lo_idx = max(0, o.test_start - context)
        hist_d = bundle.dates[lo_idx:o.test_start]
        hist_v = bundle.values[lo_idx:o.test_start]
        fam = o.meta.get("family", "classical")
        ax.plot(hist_d, hist_v, color=INK_MUTED, linewidth=0.9)
        ax.plot(o.dates_test, o.y_true, color=INK, linewidth=1.4)
        ax.fill_between(o.dates_test, o.lower, o.upper, color=_fam(fam),
                        alpha=0.20, linewidth=0)
        ax.plot(o.dates_test, o.point, color=_fam(fam), linewidth=LINE_W)
        # Shared-module coverage, so the number annotated on the chart is
        # the same number the results table reports.
        cov = _coverage(o.y_true, o.lower, o.upper)
        ax.set_ylabel(name, fontsize=8, color=INK_SECONDARY)
        ax.text(0.995, 0.92, f"coverage {cov:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                color=STATUS_GOOD if config.is_calibrated(cov)
                else STATUS_BAD)
        _despine(ax)
    # Build the legend from neutral proxies rather than letting it inherit the
    # first panel's family colour, which would imply every panel's forecast is
    # that colour. Each panel's own hue carries the family; the legend only has
    # to explain the three roles.
    proxies = [
        plt.Line2D([0], [0], color=INK, linewidth=1.4, label="actual"),
        plt.Line2D([0], [0], color=INK_SECONDARY, linewidth=LINE_W,
                   label="forecast (panel colour = model family)"),
        plt.Rectangle((0, 0), 1, 1, color=INK_SECONDARY, alpha=0.20,
                      label=f"{config.NOMINAL_COVERAGE:.0%} interval"),
        plt.Line2D([0], [0], color=INK_MUTED, linewidth=0.9,
                   label="training history"),
    ]
    axes[0].legend(handles=proxies, loc="upper left", ncol=4,
                   bbox_to_anchor=(0, 1.62))
    axes[0].set_title(
        f"{bundle.label} — fold {fold} "
        f"({outcomes_by_model[models_[0]].dates_test[0].date()} to "
        f"{outcomes_by_model[models_[0]].dates_test[-1].date()})", loc="left")
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 9. Coverage by horizon step
# --------------------------------------------------------------------------

def plot_coverage_by_horizon(rows: pd.DataFrame, fname: str,
                             title: str) -> str:
    """Does the interval hold up as h grows?

    Pooled coverage can hit nominal exactly while being far too wide at h=1 and
    far too narrow at h=14. Only a per-step view shows that.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    nominal = config.NOMINAL_COVERAGE
    ax.axhspan(nominal - config.COVERAGE_TOLERANCE,
               nominal + config.COVERAGE_TOLERANCE, color=STATUS_GOOD,
               alpha=0.07, linewidth=0)
    ax.axhline(nominal, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    for model, g in rows.groupby("model"):
        fam = g["family"].iloc[0]
        ax.plot(g["h"], g["coverage"], color=_fam(fam),
                marker=FAMILY_MARKER.get(fam, "o"), markersize=MARKER_SZ - 2,
                linewidth=LINE_W, markeredgecolor=SURFACE,
                markeredgewidth=0.6, label=model, alpha=0.9)
    ax.set_xlabel("horizon step h")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncol=3, fontsize=7)
    _despine(ax)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 10. Cross-dataset model comparison
# --------------------------------------------------------------------------

def plot_model_comparison(pooled: pd.DataFrame, metric: str, fname: str,
                          title: str, baseline_ref: float | None = None) -> str:
    """Grouped bars: one panel per dataset, one bar per model.

    Bars are separated by a surface gap rather than outlined, and only the
    best bar in each panel is direct-labelled.
    """
    apply_style()
    datasets = list(dict.fromkeys(pooled["dataset"]))
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.1 * len(datasets), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = pooled[pooled["dataset"] == ds].sort_values(metric)
        colors = [_fam(f) for f in sub["family"]]
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub[metric], color=colors, height=0.68, linewidth=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["model"], fontsize=7)
        ax.invert_yaxis()
        best = sub.iloc[0]
        ax.annotate(f"{best[metric]:.2f}", (best[metric], 0),
                    textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8, color=INK)
        if baseline_ref is not None:
            ax.axvline(baseline_ref, color=INK_SECONDARY, linewidth=1.0,
                       linestyle="--")
        ax.set_title(ds, fontsize=9)
        ax.grid(axis="y", visible=False)
        _despine(ax)
    axes[0].set_xlabel(metric)
    fig.suptitle(title, fontsize=10, y=1.04, x=0.02, ha="left",
                 color=INK, weight="semibold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f)
               for f, c in FAMILY_COLOR.items()]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=7,
                    title="family", title_fontsize=7)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 11. Feature importance
# --------------------------------------------------------------------------

def plot_feature_importance(importance: dict, fname: str, title: str,
                            top_n: int = 18) -> str:
    apply_style()
    s = pd.Series(importance).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(s) + 1.3))
    ax.barh(np.arange(len(s)), s.to_numpy(), color=FAMILY_COLOR["ml"],
            height=0.68, linewidth=0)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("LightGBM split count")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return _finish(fig, fname)


# --------------------------------------------------------------------------
# 12. Expanding vs rolling, head to head
# --------------------------------------------------------------------------

def plot_window_comparison(pooled: pd.DataFrame, metric: str, fname: str,
                           title: str) -> str:
    """Paired dots: the same model, the same test windows, two training shapes.

    A connecting line between the pair makes the direction of the effect
    readable without a second axis or a difference column.
    """
    apply_style()
    wide = pooled.pivot_table(index=["dataset", "model", "family"],
                              columns="window_type", values=metric).dropna()
    if wide.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no paired results", ha="center", va="center")
        return _finish(fig, fname)
    wide = wide.reset_index()
    order = wide.sort_values(["dataset", "model"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.32 * len(order) + 1.8))
    y = np.arange(len(order))
    for i, r in order.iterrows():
        ax.plot([r["expanding"], r["rolling"]], [i, i], color=GRID,
                linewidth=2.2, solid_capstyle="round", zorder=1)
    ax.plot(order["expanding"], y, linestyle="", marker="o",
            markersize=MARKER_SZ, color=FAMILY_COLOR["classical"],
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="expanding",
            zorder=2)
    ax.plot(order["rolling"], y, linestyle="", marker="D",
            markersize=MARKER_SZ, color=FAMILY_COLOR["gam"],
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="rolling",
            zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.dataset} · {r.model}" for r in
                        order.itertuples()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return _finish(fig, fname)

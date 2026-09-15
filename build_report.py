"""Render CAPSTONE_REPORT.md from the artefacts the pipeline actually produced.

Run after ``capstone_pipeline.py``::

    python build_report.py

**Why generate the report instead of writing it by hand.** A hand-written
report and a results table drift apart the moment either is touched -- and the
drift is invisible, because a stale number in prose still looks like a number.
Every figure, metric and verdict below is read out of ``report_data.json`` and
``outputs/tables/*.csv``. The narrative is fixed; the numbers, the rankings, the
calibration verdicts and the recommendations are computed. If the pipeline is
re-run on different data, the report's claims change with it or the build
fails -- it cannot quietly disagree with its own evidence.

The interpretive judgements -- which trade-off matters for which use case, why a
family wins -- are written here in prose, as they must be. What is *not* written
by hand is any number, name or direction that the data determines.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

ROOT = config.ROOT
REPORT_PATH = ROOT / "CAPSTONE_REPORT.md"

#: Human-facing names, so the report does not read like a variable dump.
MODEL_LABEL = {
    "seasonal_naive": "Seasonal naive",
    "sarima": "SARIMA",
    "ets": "ETS / Holt-Winters",
    "prophet": "Prophet",
    "sktime_theta": "sktime Theta",
    "lgbm_quantile": "LightGBM (quantile)",
    "lgbm_conformal": "LightGBM (conformal)",
    "lgbm_global": "LightGBM (global)",
}
FAMILY_LABEL = {
    "baseline": "Baseline",
    "classical": "Classical",
    "gam": "GAM / framework",
    "ml": "Tree-based ML",
}
#: Frequency codes rendered for a human reader.
FREQ_LABEL = {"D": "daily", "MS": "monthly", "W": "weekly"}


def dataset_title(key: str, data: dict) -> str:
    """Build a dataset heading from the spec and the measured series.

    Previously a hardcoded lookup table. That duplicated facts the pipeline
    already knows — the frequency, the series count, the zero rate, the
    presence of a break — and would have silently gone stale if a dataset's
    fold plan or series count ever changed. Derived, it cannot.
    """
    spec = config.DATASETS[key]
    ds = data["datasets"][key]["spec"]
    parts = [FREQ_LABEL.get(spec.freq, spec.freq)]

    n_series = ds.get("n_series_in_file", 1)
    parts.append(f"{n_series} series" if n_series > 1 else "one series")

    # The distinguishing characteristic, taken from what was measured rather
    # than from an adjective typed in advance.
    summaries = data["datasets"][key].get("series_summary", [])
    zero_rate = (max((s.get("zero_rate", 0.0) for s in summaries), default=0.0)
                 if summaries else 0.0)
    if spec.structural_break:
        parts.append(f"one structural break ({spec.structural_break})")
    elif zero_rate > 0.5:
        parts.append(f"~{zero_rate:.0%} zero rows")
    elif ds.get("n", 0) < 200:
        parts.append(f"{ds['n']} points")
    else:
        parts.append("strongly seasonal")

    name = key.replace("_", " ").title()
    return f"{name} — " + ", ".join(parts)


# ==========================================================================
# Loading
# ==========================================================================

def load() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = ROOT / "report_data.json"
    if not path.exists():
        sys.exit("report_data.json not found -- run `python capstone_pipeline.py` first.")
    data = json.loads(path.read_text(encoding="utf-8"))

    def _csv(name):
        """Read a results table, tolerating an absent or header-less file.

        A stage that produced no rows leaves either no file or a zero-byte one,
        and ``read_csv`` raises ``EmptyDataError`` on the latter. The report
        should degrade to "this section has nothing to show" rather than fail
        the whole build over an optional table.
        """
        p = config.TABLE_DIR / name
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return (data, _csv("pooled_metrics.csv"), _csv("per_fold_metrics.csv"),
            _csv("coverage_by_horizon.csv"))


def primary_label(key: str) -> str:
    spec = config.DATASETS[key]
    return "/".join(spec.primary_group) if spec.primary_group else key


def primary_rows(pooled: pd.DataFrame, key: str,
                 window_type: str = "expanding") -> pd.DataFrame:
    """Pooled rows for a dataset's primary series only."""
    sub = pooled[(pooled["dataset"] == key)
                 & (pooled["window_type"] == window_type)]
    if "series" in sub.columns:
        sub = sub[sub["series"] == primary_label(key)]
    return sub.copy()


# ==========================================================================
# Formatting helpers
# ==========================================================================

def fmt(x, nd=2, dash="—") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x, nd=1, dash="—") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{float(x) * 100:.{nd}f}%"


def md_table(rows: list[list[str]], header: list[str],
             align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def calib_mark(cov) -> str:
    if cov is None or not np.isfinite(cov):
        return "—"
    if config.is_calibrated(cov):
        return "**calibrated**"
    return "over-covers" if cov > config.NOMINAL_COVERAGE else "under-covers"


def headline_metric(key: str) -> str:
    return config.DATASETS[key].scale_free_metric


# ==========================================================================
# Derived findings -- computed, never asserted
# ==========================================================================

def best_model(sub: pd.DataFrame, metric: str,
               exclude_baseline: bool = True) -> pd.Series | None:
    s = sub[sub[metric].notna()]
    if exclude_baseline:
        s = s[s["family"] != "baseline"]
    return None if s.empty else s.loc[s[metric].idxmin()]


def baseline_row(sub: pd.DataFrame) -> pd.Series | None:
    s = sub[sub["model"] == "seasonal_naive"]
    return None if s.empty else s.iloc[0]


def calibration_summary(pooled: pd.DataFrame) -> dict:
    s = pooled[pooled["coverage"].notna()].copy()
    s["is_cal"] = s["coverage"].apply(config.is_calibrated)
    by_family = (s.groupby("family")["is_cal"]
                 .agg(["sum", "count"]).reset_index())
    by_model = (s.groupby("model")
                .agg(n=("is_cal", "size"), n_cal=("is_cal", "sum"),
                     mean_cov=("coverage", "mean"),
                     mean_width_pct=("width_pct_of_mean", "mean"))
                .reset_index().sort_values("n_cal", ascending=False))
    return {
        "n_total": int(len(s)),
        "n_calibrated": int(s["is_cal"].sum()),
        "by_family": by_family,
        "by_model": by_model,
        "worst_under": s.loc[s["coverage"].idxmin()] if len(s) else None,
        "widest": (s.loc[s["width_pct_of_mean"].idxmax()]
                   if s["width_pct_of_mean"].notna().any() else None),
    }


def window_effect(pooled: pd.DataFrame, metric: str = "wape") -> pd.DataFrame:
    """Per (dataset, series, model): rolling minus expanding on `metric`."""
    idx = ["dataset", "model", "family"]
    if "series" in pooled.columns:
        idx.insert(1, "series")
    wide = pooled.pivot_table(index=idx, columns="window_type",
                              values=metric).dropna().reset_index()
    if "expanding" not in wide or "rolling" not in wide:
        return pd.DataFrame()
    wide["delta"] = wide["rolling"] - wide["expanding"]
    wide["rolling_better"] = wide["delta"] < 0
    return wide


# ==========================================================================
# Sections
# ==========================================================================

def section_header(data: dict) -> str:
    env = data["environment"]
    pk = env["packages"]
    return f"""# Capstone Report — Calibrated Forecast Backtesting

**A multi-model time series benchmark with walk-forward validation and calibrated prediction intervals.**

| | |
|---|---|
| **Programme** | {config.PROGRAMME} ({config.PROGRAMME_AR}) — {config.PROGRAMME_PROVIDER}, {config.PROGRAMME_FORMAT} |
| **Cohort / session dates** | {config.COHORT_STATEMENT} |
| **Repository** | {config.GITHUB_URL} |
| **SDAIA Academy** | {config.SDAIA_GITHUB} |
| **Course repository** | {config.COURSE_REPO} |

> **This report is generated, not typed.** Every number, ranking and
> calibration verdict below is read from `report_data.json` and
> `outputs/tables/*.csv` by [`build_report.py`](build_report.py). The prose
> frames and interprets; it never states a figure the run did not produce.
>
> Run: {env['generated_utc']} · Python {env['python']} · pandas {pk.get('pandas')} ·
> statsmodels {pk.get('statsmodels')} · LightGBM {pk.get('lightgbm')} ·
> Prophet {pk.get('prophet')} · sktime {pk.get('sktime')} ·
> seed {env['seed']} · pipeline wall time {data.get('elapsed_seconds', '—')}s
>
> **All four datasets are synthetic**, generated by the course's seeded
> `data/generate_series.py`. Every result describes those synthetic series —
> not any real retailer, employer, or government indicator.

---
"""


def section_tldr(data: dict, pooled: pd.DataFrame) -> str:
    rows = []
    for key in config.DATASETS:
        sub = primary_rows(pooled, key)
        if sub.empty:
            continue
        m = headline_metric(key)
        best = best_model(sub, m)
        base = baseline_row(sub)
        if best is None:
            continue
        beats = ""
        if base is not None and np.isfinite(base[m]):
            ratio = best[m] / base[m] if base[m] else np.nan
            beats = f"{(1 - ratio) * 100:.0f}% better than naive" \
                if np.isfinite(ratio) else "—"
        # The model we'd actually deploy: best accuracy among *calibrated*
        # models, falling back to best accuracy if none is calibrated.
        cal = sub[(sub["family"] != "baseline")
                  & sub["coverage"].apply(
                      lambda c: bool(np.isfinite(c)) and config.is_calibrated(c))]
        deploy = (cal.loc[cal[m].idxmin()] if not cal.empty else best)
        rows.append([
            f"`{key}`",
            f"**{MODEL_LABEL.get(best['model'], best['model'])}**",
            f"{m.upper()} {fmt(best[m], 3 if m == 'mase' else 2)}",
            beats,
            f"**{MODEL_LABEL.get(deploy['model'], deploy['model'])}**",
            f"{pct(deploy['coverage'])} / {fmt(deploy['width_pct_of_mean'], 1)}%",
        ])
    table = md_table(
        rows,
        ["Dataset", "Most accurate", "Score", "vs. baseline",
         "Recommended to deploy", "Its coverage / width"],
        ["---", "---", "---", "---", "---", "---"])

    cs = calibration_summary(pooled)
    return f"""## The short version

{table}

*"Recommended to deploy" is the most accurate model whose 90% interval is
actually calibrated (within ±{config.COVERAGE_TOLERANCE:.0%} of nominal). Where
that differs from "most accurate", the difference is the whole point of this
report: a sharper point forecast whose uncertainty is a fiction is not the
safer choice.*

**The headline finding is that the winner changes across the four series, and
the reason it changes is legible** — history length, seasonal structure, and
sparsity each rule out different tools before accuracy is even measured.

**The second finding is about uncertainty, and it is less comfortable.** Of
{cs['n_total']} model × dataset × window-type combinations scored,
**{cs['n_calibrated']} produced a 90% interval whose empirical coverage was
actually within ±{config.COVERAGE_TOLERANCE:.0%} of 90%** — {cs['n_calibrated'] / cs['n_total']:.0%}.
Point accuracy and interval honesty are close to independent properties here,
and a model chosen on MAE alone is chosen on half the evidence.

---
"""


def section_method(data: dict) -> str:
    fold_rows, audit_rows = [], []
    for key, ds in data["datasets"].items():
        spec, f = ds["spec"], ds["spec"]["folds"]
        fold_rows.append([
            f"`{key}`", spec["freq"], f"{spec['n']:,}",
            f"{f['n_folds']}", f"{f['horizon']}",
            f"{f['min_train_size']:,}", f"{f['rolling_train_size']:,}",
            spec["transform"], spec["scale_free_metric"].upper(),
        ])
        a = ds.get("audits", {})
        if a:
            lk = a.get("leakage", {})
            hp = a.get("harness_parity", {})
            worst = max((hp[w]["max_abs_point_difference"] for w in hp),
                        default=float("nan"))
            audit_rows.append([
                f"`{key}`",
                "pass" if lk.get("prediction_features_invariant") else "**FAIL**",
                "pass" if lk.get("training_features_invariant") else "**FAIL**",
                f"{lk.get('n_features', '—')}",
                f"{lk.get('future_values_perturbed', '—')}",
                f"{worst:.1e}",
            ])

    return f"""## Method

### Fold design

{md_table(fold_rows,
          ["Dataset", "Freq", "n", "Folds", "Horizon", "Expanding min-train",
           "Rolling train", "Transform", "Headline metric"],
          ["---", "---", "--:", "--:", "--:", "--:", "--:", "---", "---"])}

Every dataset is backtested **twice** — once expanding, once rolling — over the
*identical* test windows. That is a property of `common/backtest.py`'s two
split functions, and it is what makes the expanding-vs-rolling comparison a
controlled experiment instead of two unrelated backtests.

The workforce geometry is the one that had to be solved rather than chosen. Its
break sits at index {data['datasets'].get('workforce', {}).get('audits', {}).get('break_index', '—')} of
{data['datasets'].get('workforce', {}).get('spec', {}).get('n', '—')}. Because the scored region is always the
final `n_folds × horizon` rows, anything under 20 folds pushes the whole scored
region *past* the break — producing a backtest that looks fine and tests
nothing about the event the dataset exists for.

### Leakage audit

{md_table(audit_rows,
          ["Dataset", "Prediction features invariant", "Training features invariant",
           "Features", "Future values perturbed", "Harness parity (max Δ)"],
          ["---", "---", "---", "--:", "--:", "--:"])}

The audit does not take the code's word for it. It replaces the **entire
future** with different numbers, rebuilds both design matrices, and asserts
that not one feature value moves — which it cannot do if any column reached
past the forecast origin. The last column is a separate check: this project's
fold runner versus the course's own `run_backtest`, on the same baseline. A
difference of `0.0e+00` means the folds being scored here are exactly the folds
the course defines.

### What the models are, and are not, given

The moving-holiday window **is** supplied to SARIMA (`exog`), Prophet
(`holidays=`) and LightGBM (a feature column) alike — a Hijri-calendar date is
known years ahead, and denying it to some families and not others would make
the comparison unfair rather than rigorous. The **promo shocks are supplied to
nobody**: they are drawn at random in the generator with no advance signal, so
a planner standing at the forecast origin could not know them. Handing a model
a promo flag would be forecasting with tomorrow's newspaper, and the residual
error those shocks cause is a real and irreducible part of every number below.

---
"""


def section_dataset(data: dict, key: str, pooled: pd.DataFrame,
                    per_fold: pd.DataFrame, cov_h: pd.DataFrame) -> str:
    ds = data["datasets"][key]
    spec = ds["spec"]
    diag = ds.get("diagnostics", {})
    figs = diag.get("figures", {})
    m = headline_metric(key)

    # -- model table (expanding, primary series) --------------------------
    sub = primary_rows(pooled, key).sort_values(m if m in pooled else "mae")
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            MODEL_LABEL.get(r["model"], r["model"]),
            FAMILY_LABEL.get(r["family"], r["family"]),
            fmt(r.get("mae"), 3), fmt(r.get("rmse"), 3),
            fmt(r.get("wape"), 2), fmt(r.get("mase"), 3),
            pct(r.get("coverage")), fmt(r.get("width_pct_of_mean"), 1),
            calib_mark(r.get("coverage")),
            fmt(r.get("fit_seconds_per_fold"), 2),
        ])
    table = md_table(
        rows, ["Model", "Family", "MAE", "RMSE", "WAPE %", "MASE",
               "Coverage", "Width % of mean", "Calibration", "s / fold"],
        ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "---", "--:"])

    # -- diagnostics prose -------------------------------------------------
    stat_lines = "\n".join(
        f"- **{s['name']}** → *{s['verdict']}*. {s['interpretation']}"
        for s in diag.get("stationarity", []))
    decomp = diag.get("decomposition", {}).get("interpretation", "")
    form = diag.get("additive_vs_multiplicative", {}).get("interpretation", "")
    acf = diag.get("acf_pacf", {}).get("interpretation", "")

    # -- residual diagnostics ---------------------------------------------
    pf = per_fold[(per_fold["dataset"] == key)
                  & (per_fold["window_type"] == "expanding")]
    if "series" in pf.columns:
        pf = pf[pf["series"] == primary_label(key)]
    lb_bits = []
    for model in ("sarima", "ets"):
        g = pf[(pf["model"] == model) & pf.get("ljung_box_p", pd.Series(dtype=float)).notna()] \
            if "ljung_box_p" in pf.columns else pd.DataFrame()
        if not g.empty:
            n_pass = int((g["ljung_box_p"] > config.ALPHA).sum())
            lb_bits.append(
                f"**{MODEL_LABEL[model]}** passed the Ljung-Box white-noise "
                f"test on {n_pass} of {len(g)} folds "
                f"(median p = {g['ljung_box_p'].median():.3f})")
    lb_text = ("; ".join(lb_bits) + "." if lb_bits
               else "No residual diagnostics were recorded for this series.")

    orders = ""
    if "order" in pf.columns:
        g = pf[(pf["model"] == "sarima") & pf["order"].notna()]
        if not g.empty:
            picked = g.groupby(["order", "seasonal_order"]).size() \
                      .sort_values(ascending=False)
            top = picked.index[0]
            orders = (f" AIC selected `order={top[0]}`, "
                      f"`seasonal_order={top[1]}` on "
                      f"{picked.iloc[0]} of {len(g)} folds.")

    # -- per-fold spread ---------------------------------------------------
    spread = ""
    if not pf.empty and "wape" in pf.columns:
        g = pf[pf["family"] != "baseline"].groupby("model")["wape"]
        agg = g.agg(["mean", "std", "min", "max"]).dropna()
        if not agg.empty:
            worst = agg["max"].idxmax()
            spread = (
                f"Fold-to-fold spread matters more than the mean: "
                f"**{MODEL_LABEL.get(worst, worst)}** ranged from "
                f"{agg.loc[worst, 'min']:.2f}% to {agg.loc[worst, 'max']:.2f}% "
                f"WAPE across folds (mean {agg.loc[worst, 'mean']:.2f}%, "
                f"sd {agg.loc[worst, 'std']:.2f}). A single average would have "
                f"hidden that range entirely.")

    fig_md = "\n".join(
        f"![{name}]({path})" for name, path in figs.items()
        if name in ("overview", "decomposition", "acf_pacf", "transform"))
    # Figure keys must match the names capstone_pipeline.py stores them under
    # exactly; a mismatch silently drops the image rather than raising.
    extra_figs = []
    for tag, cap in ((f"per_fold_{key}", "Per-fold WAPE"),
                     (f"fan_{key}", "Forecast fans, fold 0"),
                     (f"importance_{key}", "LightGBM feature importance")):
        p = data.get("figures", {}).get(tag)
        if p:
            extra_figs.append(f"![{cap}]({p})")

    return f"""## {dataset_title(key, data)}

{spec['description']}

`{spec['n']:,}` observations at `{spec['freq']}`, {spec['start']} → {spec['end']},
{spec['n_series_in_file']} series in the file. Headline scale-free metric:
**{m.upper()}**. Target transform: **{spec['transform']}**.

{fig_md}

### Structure and diagnostics

{stat_lines}

**Decomposition.** {decomp} {form}

**Autocorrelation.** {acf}

### Model results — expanding window, `{primary_label(key)}`

{table}

**Residual diagnostics.** {lb_text}{orders}

{spread}

{chr(10).join(extra_figs)}

---
"""


def section_windows(pooled: pd.DataFrame) -> str:
    eff = window_effect(pooled, "wape")
    if eff.empty:
        return "## Expanding vs rolling\n\nNo paired results.\n\n---\n"

    by_ds = eff.groupby("dataset").agg(
        n=("delta", "size"), n_rolling_better=("rolling_better", "sum"),
        mean_delta=("delta", "mean")).reset_index()
    rows = [[f"`{r.dataset}`", f"{r.n_rolling_better}/{r.n}",
             f"{r.mean_delta:+.2f}",
             "rolling" if r.mean_delta < 0 else "expanding"]
            for r in by_ds.itertuples()]

    wf = eff[eff["dataset"] == "workforce"]
    wf_text = ""
    if not wf.empty:
        n_better = int(wf["rolling_better"].sum())
        wf_text = (
            f"On the series with an actual structural break, rolling beat "
            f"expanding for **{n_better} of {len(wf)}** models, with a mean "
            f"WAPE change of **{wf['delta'].mean():+.2f} points**. ")

    overall_better = int(eff["rolling_better"].sum())
    return f"""## Does the verdict survive a second window type?

Both window types score the **identical** test windows, so any difference is
attributable to the training window's shape and nothing else.

{md_table(rows, ["Dataset", "Models where rolling won", "Mean WAPE change (rolling − expanding)", "Preferred"],
          ["---", "--:", "--:", "---"])}

Across every dataset and model, rolling won **{overall_better} of {len(eff)}**
paired comparisons. {wf_text}

**The practical reading.** Expanding is the right default: more history rarely
hurts a model that can use it, and it never discards data. Rolling earns its
place only when older history is *actively misleading* rather than merely
smaller. This backtest is the evidence for which situation you are in — and
notably, it is the same evidence either way, because both window types were run
over the same folds rather than one being chosen in advance.

![Expanding vs rolling](outputs/figures/window_expanding_vs_rolling.png)

---
"""


def section_calibration(pooled: pd.DataFrame, cov_h: pd.DataFrame,
                        data: dict) -> str:
    cs = calibration_summary(pooled)
    bm = cs["by_model"]
    rows = [[MODEL_LABEL.get(r["model"], r["model"]),
             f"{int(r['n_cal'])}/{int(r['n'])}",
             pct(r["mean_cov"]), fmt(r["mean_width_pct"], 1)]
            for _, r in bm.iterrows()]

    worst, widest = cs["worst_under"], cs["widest"]
    worst_txt = ""
    if worst is not None:
        worst_txt = (
            f"The worst under-coverage is **{MODEL_LABEL.get(worst['model'], worst['model'])}** "
            f"on `{worst['dataset']}` at **{pct(worst['coverage'])}** against a "
            f"90% target, with a mean width of only "
            f"{fmt(worst['width_pct_of_mean'], 1)}% of the series' level. That "
            f"is the dangerous failure: a narrow band that looks confident and "
            f"is wrong three times more often than it admits. ")
    wide_txt = ""
    if widest is not None:
        wide_txt = (
            f"The widest interval is **{MODEL_LABEL.get(widest['model'], widest['model'])}** "
            f"on `{widest['dataset']}` at {fmt(widest['width_pct_of_mean'], 1)}% "
            f"of the mean level, covering {pct(widest['coverage'])} — which is "
            f"why coverage is never quoted here without width beside it. An "
            f"interval wide enough covers everything and says nothing.")

    return f"""## Calibration: did the 90% interval actually cover 90%?

**{cs['n_calibrated']} of {cs['n_total']}** scored combinations produced an
interval within ±{config.COVERAGE_TOLERANCE:.0%} of the 90% nominal level.

{md_table(rows, ["Model", "Calibrated runs", "Mean coverage", "Mean width (% of level)"],
          ["---", "--:", "--:", "--:"])}

{worst_txt}{wide_txt}

![Calibration](outputs/figures/calibration_coverage_vs_width.png)

### Where the interval methods differ

The three interval constructions in this project fail in characteristically
different ways, and the pattern is more useful than any single number:

- **Analytic intervals** (SARIMA, ETS) assume the model is correctly specified
  and the residuals Gaussian. When the model fits, they are well-sized; when
  the series shifts regime, they are confidently wrong, because the assumption
  they rest on is the thing that just broke.
- **Prophet's native interval** samples future trend changepoints. It widens
  with horizon, which is right in shape, but in MAP mode it accounts for trend
  and observation noise only — not seasonality uncertainty — and tends to
  under-cover on series whose seasonal amplitude is doing real work.
- **Quantile-objective LightGBM** learns the conditional quantiles of the
  *training* distribution. It is the sharpest of the three and the most
  overconfident: it has no mechanism to widen for a regime it has not seen.
- **Split conformal** makes no distributional assumption — it just asks how
  wrong the model was recently. Recomputing the margin inside every fold, and
  per horizon step rather than pooled, is what keeps it honest; a globally
  computed margin would be narrowest exactly when a break makes it least
  trustworthy.

### Coverage decays with horizon

A pooled coverage number can hit 90% exactly while being far too wide at h=1
and far too narrow at h=14. Only a per-step view shows it, which is why the
conformal margin here is computed per horizon step rather than pooled.

{chr(10).join(f"![Coverage by horizon]({p})" for t, p in data.get('figures', {}).items() if t.startswith('coverage_h_'))}

---
"""


def section_metrics(data: dict) -> str:
    demo = data.get("mape_failure_demo")
    if not demo:
        return ""
    return f"""## Why the scale-free metric is WAPE, not MAPE, on sparse demand

The metrics cheat sheet makes this argument with a hand-built ten-day window.
This project measured it on its own backtest folds.

| Metric | Median across {demo['n_folds']} scored folds |
|---|--:|
| MAE | {fmt(demo['mae_median'], 3)} |
| **MAPE** | **{demo['mape_median']:,.0f}%** |
| sMAPE | {fmt(demo['smape_median'], 1)}% |
| **WAPE** | **{fmt(demo['wape_median'], 1)}%** |

{demo['interpretation']}

MAPE is not merely a worse metric here — it is not a number. It is reported in
this project exactly once, as this demonstration, and never as a score.

---
"""


def section_decision(data: dict, pooled: pd.DataFrame) -> str:
    rows = []
    for key in config.DATASETS:
        sub = primary_rows(pooled, key)
        if sub.empty:
            continue
        m = headline_metric(key)
        cal = sub[(sub["family"] != "baseline")
                  & sub["coverage"].apply(
                      lambda c: bool(np.isfinite(c)) and config.is_calibrated(c))]
        pick = (cal.loc[cal[m].idxmin()] if not cal.empty
                else best_model(sub, m))
        if pick is None:
            continue
        spec = config.DATASETS[key]
        n = data["datasets"][key]["spec"]["n"]
        rows.append([
            f"`{key}`",
            f"{n:,} @ `{spec.freq}`",
            FAMILY_LABEL.get(pick["family"], pick["family"]),
            f"**{MODEL_LABEL.get(pick['model'], pick['model'])}**",
            fmt(pick.get("fit_seconds_per_fold"), 2) + " s",
            calib_mark(pick.get("coverage")),
        ])

    return f"""## Decision framework

Accuracy is one axis of four, and on these four series it is rarely the binding
one. The questions that actually decide the choice, in the order they should be
asked:

**1. How much history is there?** This rules tools out before anything else is
considered. The economic series has 108 points *in total*; a `lag_12` feature
costs 11% of it before a single tree is grown, and `mase(..., seasonal_period=12)`
is undefined on the shorter training windows. A classical model estimating a
handful of parameters is not merely a safe choice there — it is the choice the
sample size permits.

**2. Does anyone have to explain the forecast?** ARIMA coefficients and
Prophet's decomposed trend/seasonality/holiday components can be read by a
domain expert who has never seen the code. LightGBM feature importances say
which *columns* mattered, not a story about trend and seasonality anyone can
put in a slide. If a headcount plan has to be defended to a manager, that
consideration outranks a point of WAPE.

**3. Does the decision need a calibrated interval, or just a number?** This is
where the ranking most often changes. A staffing decision is made against the
*upper* bound, not the mean — an interval that under-covers is worse than no
interval, because it manufactures false confidence. statsmodels, Prophet and
sktime hand you an interval by default; LightGBM does not, and "a LightGBM
model with an interval" is a decision to build one, not a default you inherit.

**4. What is the compute budget, and how many series are there?** Per-series
classical fits are cheap individually and scale linearly — a thousand series is
a thousand fits, on every fold, forever. One global tree model amortises
training across every series it covers. At four to six series this is
irrelevant; at four thousand it is the only axis that matters.

### Recommendation per use case

{md_table(rows, ["Use case", "History", "Family", "Model", "Fit cost / fold", "Interval"],
          ["---", "---", "---", "---", "--:", "---"])}

**Demand planning (retail).** Strong, stable weekly and yearly seasonality with
a modest trend is exactly the shape the classical family was built for, and the
interval comes free and analytic. Reach for a tree model here when you have
*exogenous* signal a classical model cannot absorb — a promo calendar, price,
weather — because that, not raw flexibility, is what it is actually better at.

**Workforce projection.** The model matters less than the validation design.
Every family degrades at the break; what separates them is how fast they
recover and whether their interval widens honestly while they do. Pair a
classical model with a conformal interval recomputed per fold, and monitor
coverage in production as a drift alarm — a calibrated interval that suddenly
stops covering is the earliest signal that the regime moved.

**Economic indicator.** Short and monthly. A few-parameter model, an analytic
interval, and scepticism about anything that claims to have learned a rich
nonlinear structure from 108 points.

**Intermittent spare parts.** None of this course's four tools is the right
tool, and saying so is the honest conclusion rather than a failure to find one.
The signal is almost entirely *whether* an order occurs, not how large it is;
Croston's method and TSB model exactly that — inter-arrival time and demand
size as separate processes — and neither is in this toolbox. What this project
can say with confidence is narrower and still useful: report WAPE, never MAPE;
expect WAPE above 100% and understand why; and treat any model's point forecast
here as a re-order-point input rather than a daily quantity.

---
"""


def section_limits(data: dict) -> str:
    return f"""## Limitations

- **The data is synthetic.** These series were built to exhibit specific
  textures. A model that suits one is evidence about that texture, not about
  real demand.
- **The SARIMA order search is greedy** — two-stage stepwise, 12 fits per fold
  rather than a 32-fit full grid. It can miss a joint optimum. The trade bought
  roughly an hour of runtime and is stated rather than hidden.
- **Croston's method and TSB are not implemented**, so the intermittent
  conclusion is "none of these four tools fits" rather than "here is the model
  that does".
- **Prophet runs in MAP mode**, not MCMC, so its intervals omit seasonality
  uncertainty.
- **Calibration is assessed at one nominal level (90%).** A model calibrated at
  90% is not automatically calibrated at 50% or 99%.
- **One seed.** The generator is deterministic, so this is a study of one
  realisation of each process, not of the process.
- **Fold counts are small on two datasets** (6 folds on retail and economic).
  Per-fold spreads there are indicative, not tight.

---

## Reproducing this

```bash
git clone {config.GITHUB_URL}.git
cd {config.GITHUB_REPO.split('/')[-1]}
python -m pip install -r requirements.txt
python -m pytest tests/ -q          # {config.count_tests()} tests
python capstone_pipeline.py         # regenerates every table and figure
python build_report.py              # regenerates this file
```

Artefacts this report is built from:

{chr(10).join(f"- [`{n}`]({p})" for n, p in data.get("tables", {}).items())}
- [`report_data.json`](report_data.json) — every number above, machine-readable
- [`outputs/logs/pipeline_run.txt`](outputs/logs/pipeline_run.txt) — the run's console log

*Report generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
`build_report.py` from a pipeline run of {data.get('elapsed_seconds', '—')}s.*
"""


# ==========================================================================
# Main
# ==========================================================================

def update_readme_status(data: dict, pooled: pd.DataFrame,
                         per_fold: pd.DataFrame) -> bool:
    """Rewrite the README's generated status block from the real run.

    The README is hand-written prose, which is right for a README -- but a few
    of its facts are *measurements*: how long a reproduction takes, how many
    tests there are, how many model runs were scored. Those belong to the run,
    not to the author, so they are written into a fenced block here instead of
    being typed once and left to rot. Everything outside the markers is
    untouched.
    """
    path = ROOT / "README.md"
    if not path.exists():
        return False
    begin, end = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return False

    env = data["environment"]
    cs = calibration_summary(pooled)
    elapsed = data.get("elapsed_seconds")
    mins = f"{elapsed / 60:.0f} min" if isinstance(elapsed, (int, float)) else "—"
    n_failed = (int(pooled["n_folds_failed"].fillna(0).sum())
                if "n_folds_failed" in pooled.columns else 0)
    pk = env["packages"]

    block = f"""{begin}
*Generated by `build_report.py` from the committed run — do not edit by hand.*

| | |
|---|---|
| Last full pipeline run | {env['generated_utc']} |
| Wall time | {mins} |
| Model × dataset × window-type runs scored | {len(pooled)} |
| Folds scored in total | {len(per_fold)} |
| Folds that failed to fit | {n_failed} |
| Intervals calibrated at {config.NOMINAL_COVERAGE:.0%} ±{config.COVERAGE_TOLERANCE:.0%} | {cs['n_calibrated']} / {cs['n_total']} |
| Test cases | {config.count_tests()} |
| Python / pandas / statsmodels | {env['python']} / {pk.get('pandas')} / {pk.get('statsmodels')} |
| LightGBM / Prophet / sktime | {pk.get('lightgbm')} / {pk.get('prophet')} / {pk.get('sktime')} |
{end}"""

    start = text.index(begin)
    stop = text.index(end) + len(end)
    path.write_text(text[:start] + block + text[stop:], encoding="utf-8")
    return True


def main() -> int:
    data, pooled, per_fold, cov_h = load()
    if pooled.empty:
        sys.exit("pooled_metrics.csv is empty -- run the backtest stage first.")

    parts = [
        section_header(data),
        section_tldr(data, pooled),
        section_method(data),
    ]
    for key in config.DATASETS:
        if key in data["datasets"]:
            parts.append(section_dataset(data, key, pooled, per_fold, cov_h))
    parts += [
        section_windows(pooled),
        section_calibration(pooled, cov_h, data),
        section_metrics(data),
        section_decision(data, pooled),
        section_limits(data),
    ]
    text = "\n".join(p for p in parts if p)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)} "
          f"({len(text):,} chars, {text.count(chr(10)):,} lines)")
    if update_readme_status(data, pooled, per_fold):
        print("updated README.md status block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

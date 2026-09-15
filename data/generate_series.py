"""Generate the course's synthetic time series datasets — the golden thread.

Every lab in this course loads one of these four CSVs. They are synthetic by
design (no download, no API key, fully reproducible) but built to have real
forecasting texture: trend, multiple seasonal periods, holiday effects,
promo/structural shocks, and — for the intermittent series — the kind of
mostly-zero sparsity that breaks naive methods. Numbers reported anywhere in
this course (module pages, lab notebooks, the capstone brief) describe THESE
series, not any real retailer, employer, or government indicator.

Run:
    python generate_series.py
writes four CSVs into this directory. Deterministic (seeded) — re-running
reproduces byte-identical output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260912
rng = np.random.default_rng(SEED)


def _weekly_seasonality(dow: np.ndarray, weekday_mult: np.ndarray) -> np.ndarray:
    return weekday_mult[dow]


def _yearly_seasonality(day_of_year: np.ndarray, period: float = 365.25) -> np.ndarray:
    return np.sin(2 * np.pi * day_of_year / period) + 0.4 * np.sin(
        4 * np.pi * day_of_year / period
    )


def _hijri_like_holiday_bumps(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """A handful of multi-day demand spikes per year at drifting dates.

    Stands in for Ramadan/Eid-style moving holidays: real dates drift ~11
    days/year on the Gregorian calendar, which is exactly the kind of
    seasonal-but-not-fixed-calendar-date effect that trips up a naive
    day-of-year seasonal model and is worth teaching against.
    """
    bump = np.zeros(len(dates))
    year0 = dates[0].year
    for k, year in enumerate(range(year0, dates[-1].year + 1)):
        anchor = pd.Timestamp(year=year, month=4, day=10) - pd.Timedelta(days=11 * k)
        window = pd.date_range(anchor - pd.Timedelta(days=3), anchor + pd.Timedelta(days=9))
        for d in window:
            mask = dates == d
            if mask.any():
                bump[mask] += rng.uniform(0.35, 0.6)
    return bump


def make_retail_demand(rng: np.random.Generator) -> pd.DataFrame:
    """Daily units sold, 3 regions x 2 categories, ~3 years.

    Strong weekly seasonality (weekend lift), yearly seasonality, an upward
    trend, holiday bumps, sporadic promo shocks, and multiplicative noise.
    This is the primary series used on Day 1 and Day 2.
    """
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    n = len(dates)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values

    regions = ["Riyadh", "Jeddah", "Dammam"]
    categories = ["Grocery", "Electronics"]

    base_level = {"Grocery": 480.0, "Electronics": 120.0}
    region_mult = {"Riyadh": 1.15, "Jeddah": 1.0, "Dammam": 0.8}
    weekday_mult = {
        "Grocery": np.array([0.95, 0.92, 0.93, 0.97, 1.05, 1.35, 1.25]),
        "Electronics": np.array([0.85, 0.85, 0.88, 0.9, 1.0, 1.45, 1.4]),
    }
    trend_per_year = {"Grocery": 0.06, "Electronics": 0.14}
    noise_sigma = {"Grocery": 0.05, "Electronics": 0.12}

    holiday_bump = _hijri_like_holiday_bumps(dates, rng)
    years_elapsed = (dates - dates[0]).days / 365.25

    rows = []
    for region in regions:
        for cat in categories:
            level = base_level[cat] * region_mult[region]
            trend = 1.0 + trend_per_year[cat] * years_elapsed
            weekly = _weekly_seasonality(dow, weekday_mult[cat])
            yearly = 1.0 + 0.18 * _yearly_seasonality(doy)
            holiday = 1.0 + holiday_bump * (1.6 if cat == "Grocery" else 0.9)

            # Sporadic 2-4 day promo shocks, ~10 per series per year.
            promo = np.zeros(n)
            n_promos = int(3 * years_elapsed[-1]) + rng.integers(6, 11)
            starts = rng.integers(0, n - 5, size=n_promos)
            for s in starts:
                length = rng.integers(2, 5)
                promo[s : s + length] += rng.uniform(0.25, 0.7)

            noise = rng.normal(1.0, noise_sigma[cat], size=n)
            demand = level * trend * weekly * yearly * holiday * (1 + promo) * noise
            demand = np.clip(demand, 0, None)
            demand = rng.poisson(np.maximum(demand, 0.1)).astype(float)

            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "region": region,
                        "category": cat,
                        "units_sold": demand,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def make_workforce_demand(rng: np.random.Generator) -> pd.DataFrame:
    """Daily contact-centre headcount requirement, ~2 years, with one
    structural break (a sustained volume step-change partway through) —
    the case Day 3 uses to show a model that looked fine in backtesting
    fail after the regime shifts.
    """
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n = len(dates)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values

    weekday_mult = np.array([1.1, 1.15, 1.15, 1.1, 1.05, 0.55, 0.45])
    trend = 1.0 + 0.10 * ((dates - dates[0]).days / 365.25)
    yearly = 1.0 + 0.08 * _yearly_seasonality(doy)

    break_point = pd.Timestamp("2025-04-01")
    step = np.where(dates >= break_point, 1.35, 1.0)

    base = 60.0
    noise = rng.normal(1.0, 0.06, size=n)
    headcount = base * trend * weekday_mult[dow] * yearly * step * noise
    headcount = np.clip(np.round(headcount), 5, None)

    return pd.DataFrame({"date": dates, "required_headcount": headcount})


def make_economic_indicator(rng: np.random.Generator) -> pd.DataFrame:
    """Monthly non-oil activity index, 9 years — short history, low
    frequency, a slow business cycle and a recession-shaped dip. Used on
    Day 3 to contrast with the two daily series above.
    """
    dates = pd.date_range("2017-01-01", "2025-12-31", freq="MS")
    n = len(dates)
    t = np.arange(n)

    trend = 100 + 0.55 * t
    cycle = 6.0 * np.sin(2 * np.pi * t / 42) + 2.5 * np.sin(2 * np.pi * t / 11)

    # A 2020-shaped demand shock: a sharp dip and a partial, gradual recovery.
    shock = np.zeros(n)
    shock_start = np.where(dates.year == 2020)[0]
    if len(shock_start):
        s0 = shock_start[0]
        depth = -22.0
        recovery = np.clip(np.arange(n) - s0, 0, None) / 14.0
        shock = np.where(
            np.arange(n) >= s0,
            depth * np.exp(-recovery) ,
            0.0,
        )

    noise = rng.normal(0, 1.3, size=n)
    index = trend + cycle + shock + noise

    return pd.DataFrame({"month": dates, "activity_index": np.round(index, 2)})


def make_intermittent_demand(rng: np.random.Generator) -> pd.DataFrame:
    """Daily spare-parts demand for 4 low-turnover SKUs, ~2 years — mostly
    zero with sporadic spikes. The series that punishes a model choice that
    ignores objective 7 ("compare model families to pick the right one").
    """
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n = len(dates)

    skus = {
        "SKU-A1102": 0.06,
        "SKU-B2044": 0.03,
        "SKU-C3087": 0.10,
        "SKU-D4021": 0.02,
    }

    rows = []
    for sku, p_order_day in skus.items():
        orders = rng.binomial(1, p_order_day, size=n)
        qty = np.where(orders == 1, rng.integers(1, 9, size=n), 0)
        rows.append(pd.DataFrame({"date": dates, "sku": sku, "units_ordered": qty}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    import pathlib

    out_dir = pathlib.Path(__file__).parent

    retail = make_retail_demand(np.random.default_rng(SEED))
    workforce = make_workforce_demand(np.random.default_rng(SEED + 1))
    econ = make_economic_indicator(np.random.default_rng(SEED + 2))
    intermittent = make_intermittent_demand(np.random.default_rng(SEED + 3))

    retail.to_csv(out_dir / "retail_demand.csv", index=False)
    workforce.to_csv(out_dir / "workforce_demand.csv", index=False)
    econ.to_csv(out_dir / "economic_indicator.csv", index=False)
    intermittent.to_csv(out_dir / "intermittent_demand.csv", index=False)

    print("retail_demand.csv       ", retail.shape, retail["date"].min(), retail["date"].max())
    print("workforce_demand.csv    ", workforce.shape)
    print("economic_indicator.csv  ", econ.shape)
    print("intermittent_demand.csv ", intermittent.shape,
          "zero-rate=", (intermittent["units_ordered"] == 0).mean().round(3))


if __name__ == "__main__":
    main()

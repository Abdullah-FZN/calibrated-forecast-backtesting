# Vendored course utilities

`metrics.py` and `backtest.py` in this directory are copied **verbatim and
unmodified** from the course repository:

<https://github.com/MohammadYusif/time-series-forecasting-ai-systems>
(`common/metrics.py`, `common/backtest.py`)

They are vendored rather than reimplemented for two reasons the capstone brief
is explicit about:

1. Every metric in this project is computed by the course's own
   implementation, so a WAPE or MASE reported here is the same number the
   course would compute — including its documented edge cases (`wape`'s
   all-zero-window fallback, `mase`'s two deliberate `ValueError`s).
2. The walk-forward fold geometry comes from the course's
   `expanding_window_splits` / `rolling_window_splits`, so the folds scored
   here are the folds the course defines. `backtesting.verify_against_course_harness`
   additionally runs the seasonal-naive baseline through the course's own
   `run_backtest` and asserts the point forecasts are identical to this
   project's runner.

Copying them in keeps the repository self-contained and reproducible offline.
The notebook fetches the same two files at runtime (falling back to this local
copy first) exactly as `setup.qmd` describes, so a Colab runtime with nothing
cloned behaves identically.

**Do not edit these two files.** Any change would break the guarantee above.
Project-specific scoring and fold orchestration live in `backtesting.py`
instead.

## Datasets

`../data/*.csv` and `../data/generate_series.py` are likewise copied from the
same repository. The generator is seeded (`SEED = 20260912`) and deterministic:
re-running `python data/generate_series.py` reproduces the four CSVs
byte-for-byte.

All four series are **synthetic**. Every number this project reports describes
those synthetic series, not any real retailer, employer, or government
indicator.

"""
Step 3 – Feature engineering.

Merges bell pepper data with weather data, then builds the feature matrix:
  - Lag features: stuks and yield_per_ha from previous 1, 2, and 4 quarters
  - Rolling statistics: 4-quarter rolling mean and std of stuks per vendor+category
  - Weather features: from the same quarter's weather (for retrospective) and
    lagged weather (for forward-looking prediction 2 weeks / 1 quarter ahead)
  - Calendar: year, quarter index, season dummies
  - Vendor + category as label-encoded categoricals (XGBoost handles these natively)

Two target columns are created:
  target_stuks       – total stuks this quarter (for quarterly prediction)
  target_stuks_agg   – total stuks aggregated over all vendors per category+quarter

Output: features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BELL_CSV   = Path("bellpepper_long.csv")
WEATHER_CSV = Path("weather_quarterly.csv")
OUTPUT_CSV  = Path("features.csv")

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def quarter_to_period(year: int, quarter: str) -> int:
    """Convert year+quarter to a monotonically increasing integer (e.g. 2013Q1=0)."""
    base_year = 2013
    return (year - base_year) * 4 + QUARTER_ORDER[quarter] - 1


def build_features() -> pd.DataFrame:
    bell    = pd.read_csv(BELL_CSV)
    weather = pd.read_csv(WEATHER_CSV)

    # ── 1. Merge weather ────────────────────────────────────────────────────
    df = bell.merge(weather, on=["year", "quarter"], how="left")

    # Monotonic period index for sorting and lag operations
    df["period"] = df.apply(lambda r: quarter_to_period(r["year"], r["quarter"]), axis=1)
    df["q_num"]  = df["quarter"].map(QUARTER_ORDER)

    df = df.sort_values(["vendor", "category_code", "period"]).reset_index(drop=True)

    # ── 2. Yield per hectare (more stable signal than raw stuks) ────────────
    df["yield_per_ha"] = np.where(df["opp_ha"] > 0, df["stuks"] / df["opp_ha"], np.nan)

    # ── 3. Lag features per (vendor, category) ──────────────────────────────
    group_key = ["vendor", "category_code"]

    for lag in [1, 2, 4]:
        df[f"stuks_lag{lag}"]       = df.groupby(group_key)["stuks"].shift(lag)
        df[f"yield_ha_lag{lag}"]    = df.groupby(group_key)["yield_per_ha"].shift(lag)

    # Rolling mean and std (4-quarter trailing window, shifted so it doesn't leak)
    df["stuks_roll4_mean"] = (
        df.groupby(group_key)["stuks"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    )
    df["stuks_roll4_std"] = (
        df.groupby(group_key)["stuks"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    )

    # ── 4. Weather lags (previous quarter's weather as an early predictor) ──
    weather_cols = ["temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days"]
    weather_q = weather.copy()
    weather_q["period"] = weather_q.apply(
        lambda r: quarter_to_period(r["year"], r["quarter"]), axis=1
    )
    weather_q = weather_q.set_index("period")[weather_cols]

    for col in weather_cols:
        # lagged by 1 quarter (available 1 quarter before delivery)
        df[f"{col}_lag1"] = df["period"].map(
            lambda p, c=col, wq=weather_q: wq.loc[p - 1, c] if (p - 1) in wq.index else np.nan
        )

    # ── 5. Category-level aggregated totals (for aggregate prediction) ──────
    cat_total = (
        df.groupby(["category_code", "year", "quarter"])["stuks"]
        .sum()
        .rename("cat_total_stuks")
        .reset_index()
    )
    df = df.merge(cat_total, on=["category_code", "year", "quarter"], how="left")

    # ── 6. Label-encode categorical columns ─────────────────────────────────
    for col in ["vendor", "category_name"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes

    # ── 7. Season dummies (Q1 = cold, Q3 = peak summer, etc.) ───────────────
    df["is_peak_season"] = (df["q_num"].isin([2, 3])).astype(int)

    # ── 8. Select final feature columns ─────────────────────────────────────
    feature_cols = [
        # identifiers (not used as features, kept for grouping)
        "year", "quarter", "period", "category_code", "category_name",
        "vendor",
        # model features
        "opp_ha",
        "q_num", "is_peak_season",
        "vendor_enc", "category_name_enc",
        "stuks_lag1", "stuks_lag2", "stuks_lag4",
        "yield_ha_lag1", "yield_ha_lag2", "yield_ha_lag4",
        "stuks_roll4_mean", "stuks_roll4_std",
        # current-quarter weather
        "temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days",
        # lagged weather (1 quarter back – available for early prediction)
        "temp_mean_c_lag1", "precip_mm_lag1", "gdd_lag1", "sun_hours_lag1",
        # targets
        "stuks",          # per-vendor, per-category target
        "cat_total_stuks", # aggregate per category target
    ]

    df_out = df[feature_cols].copy()
    return df_out


if __name__ == "__main__":
    df = build_features()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns → {OUTPUT_CSV}")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

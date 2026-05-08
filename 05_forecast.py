"""
Step 5 – Forecasting: predict the next quarter's bell pepper quantity.

Two prediction modes:
  1. Quarterly forecast:  given a target (year, quarter), predict stuks
     per vendor+category and aggregate per category.

  2. Early / 2-week-ahead forecast: same model, but weather features come
     from the PREVIOUS quarter (already available 2 weeks before the quarter
     starts). Current-quarter weather columns are filled with the previous
     quarter's values as a proxy.

Usage:
  python 05_forecast.py --year 2026 --quarter Q2
  python 05_forecast.py --year 2026 --quarter Q2 --early

Outputs:
  forecast_<year>_<quarter>.csv  – per-vendor, per-category predictions
  forecast_agg_<year>_<quarter>.csv – category-level totals
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

FEATURES_CSV         = Path("features.csv")
MODEL_VENDOR_PATH    = Path("model_vendor.json")
MODEL_AGGREGATE_PATH = Path("model_aggregate.json")
WEATHER_CSV          = Path("weather_quarterly.csv")

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
PREV_QUARTER  = {"Q1": ("Q4", -1), "Q2": ("Q1", 0), "Q3": ("Q2", 0), "Q4": ("Q3", 0)}

FEATURE_COLS = [
    "opp_ha",
    "q_num", "is_peak_season",
    "vendor_enc", "category_name_enc",
    "stuks_lag1", "stuks_lag2", "stuks_lag4",
    "yield_ha_lag1", "yield_ha_lag2", "yield_ha_lag4",
    "stuks_roll4_mean", "stuks_roll4_std",
    "temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days",
    "temp_mean_c_lag1", "precip_mm_lag1", "gdd_lag1", "sun_hours_lag1",
]

AGG_FEATURE_COLS = [
    "q_num", "is_peak_season", "category_name_enc",
    "temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days",
    "temp_mean_c_lag1", "precip_mm_lag1", "gdd_lag1", "sun_hours_lag1",
    "total_opp_ha",
    "cat_stuks_lag1", "cat_stuks_lag2", "cat_stuks_lag4",
    "cat_stuks_roll4",
]


def quarter_to_period(year, quarter, base=2013):
    return (year - base) * 4 + QUARTER_ORDER[quarter] - 1


def get_weather_row(weather: pd.DataFrame, year: int, quarter: str) -> pd.Series:
    row = weather[(weather["year"] == year) & (weather["quarter"] == quarter)]
    if row.empty:
        return None
    return row.iloc[0]


def build_forecast_rows(df: pd.DataFrame, weather: pd.DataFrame,
                        target_year: int, target_quarter: str,
                        early: bool = False) -> pd.DataFrame:
    """
    Build feature rows for all (vendor, category) combinations
    that were active in recent history.
    """
    target_period = quarter_to_period(target_year, target_quarter)
    prev_q, prev_y_offset = PREV_QUARTER[target_quarter]
    prev_year = target_year + prev_y_offset

    # Last 4 periods of actual data for lag computation
    recent = df[df["period"] <= target_period - 1].copy()
    # Use most recent opp_ha per vendor+category as the current value
    latest_ha = (
        recent.sort_values("period")
        .groupby(["vendor", "category_code", "category_name",
                  "vendor_enc", "category_name_enc"])["opp_ha"]
        .last()
        .reset_index()
    )

    # Weather for target quarter (use lagged as proxy if early mode or not yet available)
    wx_current = get_weather_row(weather, target_year, target_quarter)
    wx_prev    = get_weather_row(weather, prev_year, prev_q)

    weather_cols_current = ["temp_mean_c", "temp_max_c", "precip_mm",
                            "sun_hours", "gdd", "frost_days"]
    weather_cols_lag1    = ["temp_mean_c_lag1", "precip_mm_lag1",
                            "gdd_lag1", "sun_hours_lag1"]

    rows = []
    for _, ha_row in latest_ha.iterrows():
        vendor   = ha_row["vendor"]
        cat_code = ha_row["category_code"]

        vendor_hist = recent[
            (recent["vendor"] == vendor) & (recent["category_code"] == cat_code)
        ].sort_values("period")

        def lag_stuks(n):
            period = target_period - n
            row = vendor_hist[vendor_hist["period"] == period]
            return float(row["stuks"].iloc[0]) if len(row) > 0 else np.nan

        def lag_yield(n):
            period = target_period - n
            row = vendor_hist[vendor_hist["period"] == period]
            return float(row["yield_per_ha"].iloc[0]) if len(row) > 0 else np.nan

        recent_stuks = vendor_hist["stuks"].values
        roll_mean = float(np.mean(recent_stuks[-4:])) if len(recent_stuks) >= 2 else np.nan
        roll_std  = float(np.std(recent_stuks[-4:]))  if len(recent_stuks) >= 2 else np.nan

        feat = {
            "vendor":             vendor,
            "category_code":      cat_code,
            "category_name":      ha_row["category_name"],
            "vendor_enc":         ha_row["vendor_enc"],
            "category_name_enc":  ha_row["category_name_enc"],
            "opp_ha":             ha_row["opp_ha"],
            "q_num":              QUARTER_ORDER[target_quarter],
            "is_peak_season":     int(QUARTER_ORDER[target_quarter] in [2, 3]),
            "stuks_lag1":         lag_stuks(1),
            "stuks_lag2":         lag_stuks(2),
            "stuks_lag4":         lag_stuks(4),
            "yield_ha_lag1":      lag_yield(1),
            "yield_ha_lag2":      lag_yield(2),
            "yield_ha_lag4":      lag_yield(4),
            "stuks_roll4_mean":   roll_mean,
            "stuks_roll4_std":    roll_std,
        }

        # Weather: if early mode, use previous quarter's weather as proxy
        if early or wx_current is None:
            src = wx_prev
        else:
            src = wx_current

        for col in weather_cols_current:
            feat[col] = float(src[col]) if src is not None else np.nan

        # Lag-1 weather always comes from previous quarter
        lag_map = {"temp_mean_c_lag1": "temp_mean_c",
                   "precip_mm_lag1":   "precip_mm",
                   "gdd_lag1":         "gdd",
                   "sun_hours_lag1":   "sun_hours"}
        for lag_col, src_col in lag_map.items():
            feat[lag_col] = float(wx_prev[src_col]) if wx_prev is not None else np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


def build_agg_forecast_row(forecast_rows: pd.DataFrame,
                           df: pd.DataFrame, weather: pd.DataFrame,
                           target_year: int, target_quarter: str,
                           early: bool) -> pd.DataFrame:
    """Build aggregate (per category) feature rows."""
    prev_q, prev_y_offset = PREV_QUARTER[target_quarter]
    prev_year = target_year + prev_y_offset
    target_period = quarter_to_period(target_year, target_quarter)

    wx_current = get_weather_row(weather, target_year, target_quarter)
    wx_prev    = get_weather_row(weather, prev_year, prev_q)

    # Aggregate historical data per category
    df_agg_hist = (
        df[df["period"] <= target_period - 1]
        .groupby(["category_code", "category_name", "category_name_enc",
                  "year", "quarter", "period"])
        .agg(cat_total_stuks=("stuks", "sum"), total_opp_ha=("opp_ha", "sum"))
        .reset_index()
        .sort_values(["category_code", "period"])
    )

    agg_rows = []
    for cat_code in forecast_rows["category_code"].unique():
        cat_hist = df_agg_hist[df_agg_hist["category_code"] == cat_code]
        cat_name_enc = forecast_rows[
            forecast_rows["category_code"] == cat_code
        ]["category_name_enc"].iloc[0]
        cat_name = forecast_rows[
            forecast_rows["category_code"] == cat_code
        ]["category_name"].iloc[0]
        total_ha = forecast_rows[
            forecast_rows["category_code"] == cat_code
        ]["opp_ha"].sum()

        def lag_cat(n):
            p = target_period - n
            row = cat_hist[cat_hist["period"] == p]
            return float(row["cat_total_stuks"].iloc[0]) if len(row) > 0 else np.nan

        recent_cat = cat_hist["cat_total_stuks"].values
        roll = float(np.mean(recent_cat[-4:])) if len(recent_cat) >= 2 else np.nan

        feat = {
            "category_code":     cat_code,
            "category_name":     cat_name,
            "category_name_enc": cat_name_enc,
            "q_num":             QUARTER_ORDER[target_quarter],
            "is_peak_season":    int(QUARTER_ORDER[target_quarter] in [2, 3]),
            "total_opp_ha":      total_ha,
            "cat_stuks_lag1":    lag_cat(1),
            "cat_stuks_lag2":    lag_cat(2),
            "cat_stuks_lag4":    lag_cat(4),
            "cat_stuks_roll4":   roll,
        }
        src = wx_prev if (early or wx_current is None) else wx_current
        for col in ["temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days"]:
            feat[col] = float(src[col]) if src is not None else np.nan
        lag_map = {"temp_mean_c_lag1": "temp_mean_c", "precip_mm_lag1": "precip_mm",
                   "gdd_lag1": "gdd", "sun_hours_lag1": "sun_hours"}
        for lag_col, src_col in lag_map.items():
            feat[lag_col] = float(wx_prev[src_col]) if wx_prev is not None else np.nan

        agg_rows.append(feat)

    return pd.DataFrame(agg_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int,  required=True)
    parser.add_argument("--quarter", type=str,  required=True, choices=["Q1","Q2","Q3","Q4"])
    parser.add_argument("--early",   action="store_true",
                        help="Use only previous-quarter weather (2-week-ahead mode)")
    args = parser.parse_args()

    df      = pd.read_csv(FEATURES_CSV)
    weather = pd.read_csv(WEATHER_CSV)

    # Compute yield_per_ha (needed for lag_yield)
    df["yield_per_ha"] = np.where(df["opp_ha"] > 0, df["stuks"] / df["opp_ha"], np.nan)
    df["period"] = df.apply(
        lambda r: quarter_to_period(r["year"], r["quarter"]), axis=1
    )

    model_vendor = xgb.XGBRegressor()
    model_vendor.load_model(str(MODEL_VENDOR_PATH))

    model_agg = xgb.XGBRegressor()
    model_agg.load_model(str(MODEL_AGGREGATE_PATH))

    mode_str = "EARLY (2-week-ahead)" if args.early else "QUARTERLY"
    print(f"\n{'='*55}")
    print(f"  Forecast  {args.year} {args.quarter}  [{mode_str}]")
    print(f"{'='*55}")

    # ── Per-vendor forecast ──────────────────────────────────────────────────
    forecast_rows = build_forecast_rows(
        df, weather, args.year, args.quarter, args.early
    )
    X_pred = forecast_rows[FEATURE_COLS]
    forecast_rows["predicted_stuks"] = model_vendor.predict(X_pred).clip(0).round().astype(int)

    out_vendor = Path(f"forecast_{args.year}_{args.quarter}.csv")
    forecast_rows[["category_code", "category_name", "vendor",
                   "opp_ha", "predicted_stuks"]].to_csv(out_vendor, index=False)
    print(f"\nPer-vendor forecast → {out_vendor}")
    print(forecast_rows[["category_name", "vendor", "opp_ha", "predicted_stuks"]]
          .sort_values("predicted_stuks", ascending=False)
          .to_string(index=False))

    # ── Aggregate forecast ───────────────────────────────────────────────────
    agg_rows = build_agg_forecast_row(
        forecast_rows, df, weather, args.year, args.quarter, args.early
    )
    X_agg = agg_rows[AGG_FEATURE_COLS]
    agg_rows["predicted_total_stuks"] = model_agg.predict(X_agg).clip(0).round().astype(int)

    out_agg = Path(f"forecast_agg_{args.year}_{args.quarter}.csv")
    agg_rows[["category_code", "category_name",
              "total_opp_ha", "predicted_total_stuks"]].to_csv(out_agg, index=False)
    print(f"\nAggregate forecast → {out_agg}")
    print(agg_rows[["category_name", "predicted_total_stuks"]].to_string(index=False))


if __name__ == "__main__":
    main()

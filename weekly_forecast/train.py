"""
train.py — Bell Pepper Weekly Forecast: Model Training
=======================================================
Run this script to (re)train the forecast model from scratch.

Usage:
    python train.py

What it does:
  1. Loads historical weekly weight data (data/historical_weights.csv)
  2. Fetches real historical weather from Open-Meteo for Mechelen, Belgium
  3. Builds the feature matrix (lags + weather features)
  4. Trains an XGBoost model with walk-forward cross-validation
  5. Saves the trained model to model/xgb_model.pkl
  6. Prints CV performance metrics

Required file layout:
    data/historical_weights.csv   ← year, week, total_gewicht_kg
    model/                        ← created automatically
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
LAT, LON = 51.028, 4.480          # Mechelen, Belgium
DATA_PATH = "data/historical_weights.csv"
MODEL_PATH = "model/xgb_model.pkl"
WEATHER_CACHE = "data/weather_historical.csv"

FEATURE_COLS = [
    "lag1", "lag2", "lag4", 
    "tmean_lag1", "tmean_lag2",
    "precip_lag1", "precip_lag2",
    "gdd_lag1", "gdd_lag2",
    "rad_lag1", "rad_lag2",
    "gdd_roll4",
    "week_sin", "week_cos",
    "year_trend",
]

# ── Step 1: Load historical weights ───────────────────────────────────────────
def load_weights():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    # Accept flexible column names
    col_map = {}
    for c in df.columns:
        if "year" in c:
            col_map[c] = "year"
        elif "week" in c:
            col_map[c] = "week"
        elif "gewicht" in c or "kg" in c or "weight" in c or "total" in c:
            col_map[c] = "total_kg"
    df = df.rename(columns=col_map)[["year", "week", "total_kg"]]
    df = df.sort_values(["year", "week"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} weekly records ({df['year'].min()}–{df['year'].max()})")
    return df


# ── Step 2: Fetch historical weather from Open-Meteo ──────────────────────────
def fetch_historical_weather(start_year, end_year):
    if os.path.exists(WEATHER_CACHE):
        df = pd.read_csv(WEATHER_CACHE, parse_dates=["date"])
        cached_max = df["date"].dt.year.max()
        if cached_max >= end_year:
            print(f"  Using cached weather ({WEATHER_CACHE})")
            return df
        print(f"  Cache exists but outdated (up to {cached_max}), re-fetching...")

    start = f"{start_year - 1}-12-01"   # a few weeks before to cover week-1 lags
    year_end = pd.Timestamp(year=end_year, month=12, day=31)
    end_date = min(year_end, pd.Timestamp.today())
    end = end_date.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "start_date": start,
        "end_date":   end,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "shortwave_radiation_sum",
        ],
        "timezone": "Europe/Brussels",
    }
    print(f"  Fetching weather from Open-Meteo ({start} → {end})...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date":      pd.to_datetime(d["time"]),
        "tmax":      d["temperature_2m_max"],
        "tmin":      d["temperature_2m_min"],
        "tmean":     d["temperature_2m_mean"],
        "precip_mm": d["precipitation_sum"],
        "rad_mj":    d["shortwave_radiation_sum"],
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv(WEATHER_CACHE, index=False)
    print(f"  Saved {len(df)} daily records to {WEATHER_CACHE}")
    return df


def aggregate_weather_weekly(df_daily):
    df_daily = df_daily.copy()
    df_daily["year"] = df_daily["date"].dt.isocalendar().year.astype(int)
    df_daily["week"] = df_daily["date"].dt.isocalendar().week.astype(int)
    df_daily["gdd"]  = (df_daily["tmean"] - 10).clip(lower=0)   # GDD base 10°C

    df_w = df_daily.groupby(["year", "week"]).agg(
        tmean    = ("tmean",     "mean"),
        precip   = ("precip_mm", "sum"),
        rad      = ("rad_mj",    "sum"),
        gdd      = ("gdd",       "sum"),
    ).reset_index()
    return df_w


# ── Step 3: Build feature matrix ──────────────────────────────────────────────
def build_features(df_weights, df_weather):
    df = df_weights.merge(df_weather, on=["year", "week"], how="left")
    df = df.sort_values(["year", "week"]).reset_index(drop=True)

    # Target lags
    for lag in [1, 2, 4]:
        df[f"lag{lag}"] = df["total_kg"].shift(lag)

    # Weather lags (1 & 2 weeks prior — known at prediction time)
    for lag in [1, 2]:
        df[f"tmean_lag{lag}"]  = df["tmean"].shift(lag)
        df[f"precip_lag{lag}"] = df["precip"].shift(lag)
        df[f"gdd_lag{lag}"]    = df["gdd"].shift(lag)
        df[f"rad_lag{lag}"]    = df["rad"].shift(lag)

    # Rolling 4-week GDD (lagged by 1 so it's known at prediction time)
    df["gdd_roll4"] = df["gdd"].shift(1).rolling(4).sum()

    # Cyclical week
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # Year trend
    df["year_trend"] = (df["year"] - df["year"].min()) / 10.0

    return df


# ── Step 4: Walk-forward cross-validation + final model ───────────────────────
def train_and_evaluate(df):
    df_model = df.dropna(subset=FEATURE_COLS + ["total_kg"]).copy()
    years = sorted(df_model["year"].unique())

    cv_results = []
    print("\n  Walk-forward cross-validation:")
    print(f"  {'Test year':<12} {'Train weeks':<14} {'MAE (kg)':<14} {'MAPE active%'}")
    print("  " + "─" * 58)

    for i, test_year in enumerate(years[1:], start=1):
        train = df_model[df_model["year"].isin(years[:i])]
        test  = df_model[df_model["year"] == test_year]
        if len(train) < 20 or len(test) < 10:
            continue

        mdl = _make_model()
        mdl.fit(train[FEATURE_COLS], np.log1p(train["total_kg"].clip(lower=0)))
        preds = np.expm1(mdl.predict(test[FEATURE_COLS]))

        mae = mean_absolute_error(test["total_kg"], preds)
        active = test["total_kg"] >= 10_000
        mape_active = (
            np.mean(np.abs((test["total_kg"][active] - preds[active]) / test["total_kg"][active])) * 100
            if active.sum() > 0 else float("nan")
        )
        cv_results.append({"test_year": test_year, "mae": mae, "mape_active": mape_active})
        print(f"  {test_year:<12} {len(train):<14} {mae:>10,.0f}    {mape_active:>6.1f}%")

    df_cv = pd.DataFrame(cv_results)
    print(f"\n  Mean MAE:         {df_cv['mae'].mean():>10,.0f} kg")
    print(f"  Mean MAPE active: {df_cv['mape_active'].mean():>10.1f} %")

    # Final model on all data
    final = _make_model()
    final.fit(df_model[FEATURE_COLS], np.log1p(df_model["total_kg"].clip(lower=0)))

    # Feature importances
    fi = pd.Series(final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  Top feature importances:")
    for feat, imp in fi.head(6).items():
        print(f"    {feat:<25} {imp:.3f}")

    return final, df_cv


def _make_model():
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bell Pepper Weekly Forecast — Model Training")
    print("=" * 60)

    print("\n[1/4] Loading historical weight data...")
    df_weights = load_weights()

    print("\n[2/4] Fetching historical weather (Mechelen, Belgium)...")
    df_daily   = fetch_historical_weather(df_weights["year"].min(), df_weights["year"].max())
    df_weather = aggregate_weather_weekly(df_daily)

    print("\n[3/4] Building feature matrix...")
    df_features = build_features(df_weights, df_weather)
    usable = df_features.dropna(subset=FEATURE_COLS + ["total_kg"])
    print(f"  Feature matrix: {len(df_features)} rows → {len(usable)} usable after lag creation")

    print("\n[4/4] Training XGBoost model...")
    model, cv = train_and_evaluate(df_features)

    os.makedirs("model", exist_ok=True)
    payload = {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "cv_results":   cv,
        "train_years":  sorted(df_features["year"].unique().tolist()),
        "lat":          LAT,
        "lon":          LON,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print("   Run `streamlit run app.py` to launch the forecast app.")
    print("=" * 60)


if __name__ == "__main__":
    main()

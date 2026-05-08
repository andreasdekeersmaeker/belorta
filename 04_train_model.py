"""
Step 4 – Model training, walk-forward cross-validation, and SHAP analysis.

Two models are trained:
  1. Per-vendor model: predicts stuks for each (vendor, category, quarter)
  2. Aggregate model:  predicts total stuks per (category, quarter) — uses
     category-level feature aggregations as input

Walk-forward (expanding-window) cross-validation is used to avoid leakage:
  - Train on periods 0..t, validate on period t+1
  - Reports MAE, RMSE, and MAPE per fold and overall

SHAP values are computed on the hold-out set to explain feature importance.

Outputs:
  model_vendor.json      – trained XGBoost model (per-vendor)
  model_aggregate.json   – trained XGBoost model (aggregate)
  cv_results.csv         – walk-forward CV metrics
  shap_summary.png       – SHAP feature importance plot
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_CSV = Path("features.csv")
MODEL_VENDOR_PATH     = Path("model_vendor.json")
MODEL_AGGREGATE_PATH  = Path("model_aggregate.json")
CV_RESULTS_CSV        = Path("cv_results.csv")
SHAP_PLOT             = Path("shap_summary.png")

# ── Feature columns used as model inputs ──────────────────────────────────
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

XGB_PARAMS = dict(
    n_estimators     = 400,
    learning_rate    = 0.05,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 1.0,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    tree_method      = "hist",
)


def mape(y_true, y_pred, eps=1):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > eps
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def walk_forward_cv(df: pd.DataFrame, feature_cols: list, target_col: str,
                    min_train_periods: int = 8, step: int = 1):
    """
    Expanding-window walk-forward cross-validation.
    Each fold trains on all data up to period t, validates on period t+1.
    """
    periods = sorted(df["period"].unique())
    results = []

    for i in range(min_train_periods, len(periods) - step + 1, step):
        train_periods = periods[:i]
        val_periods   = periods[i: i + step]

        train = df[df["period"].isin(train_periods)].dropna(subset=feature_cols + [target_col])
        val   = df[df["period"].isin(val_periods)].dropna(subset=feature_cols + [target_col])

        if len(train) < 10 or len(val) == 0:
            continue

        X_train, y_train = train[feature_cols], train[target_col]
        X_val,   y_val   = val[feature_cols],   val[target_col]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        preds = model.predict(X_val).clip(0)

        results.append({
            "val_period": val_periods[0],
            "n_train":    len(train),
            "n_val":      len(val),
            "MAE":        mean_absolute_error(y_val, preds),
            "RMSE":       np.sqrt(mean_squared_error(y_val, preds)),
            "MAPE":       mape(y_val, preds),
        })
        print(f"  Period {val_periods[0]:3d} | MAE={results[-1]['MAE']:>9.0f} | "
              f"RMSE={results[-1]['RMSE']:>9.0f} | MAPE={results[-1]['MAPE']:>6.1f}%")

    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame, feature_cols: list, target_col: str) -> xgb.XGBRegressor:
    """Train on ALL available data."""
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X, y = df_clean[feature_cols], df_clean[target_col]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, verbose=False)
    return model


def plot_shap(model, df: pd.DataFrame, feature_cols: list, output_path: Path, title: str):
    df_clean = df.dropna(subset=feature_cols).sample(min(500, len(df)), random_state=42)
    X = df_clean[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved → {output_path}")


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows, {df['period'].nunique()} unique periods")

    # ── Model 1: per-vendor prediction ──────────────────────────────────────
    print("\n=== Walk-forward CV: per-vendor model ===")
    cv_vendor = walk_forward_cv(df, FEATURE_COLS, "stuks")
    print(f"\nOverall — MAE: {cv_vendor['MAE'].mean():.0f} | "
          f"RMSE: {cv_vendor['RMSE'].mean():.0f} | "
          f"MAPE: {cv_vendor['MAPE'].mean():.1f}%")

    print("\nTraining final per-vendor model on all data …")
    model_vendor = train_final_model(df, FEATURE_COLS, "stuks")
    model_vendor.save_model(str(MODEL_VENDOR_PATH))
    print(f"Saved → {MODEL_VENDOR_PATH}")

    # ── Model 2: aggregate per-category prediction ───────────────────────────
    # Build one row per (category, quarter) for the aggregate model
    agg_feature_cols = [
        "q_num", "is_peak_season", "category_name_enc",
        "temp_mean_c", "temp_max_c", "precip_mm", "sun_hours", "gdd", "frost_days",
        "temp_mean_c_lag1", "precip_mm_lag1", "gdd_lag1", "sun_hours_lag1",
    ]
    # Add total opp_ha and lagged cat totals for aggregate model
    df_agg = (
        df.groupby(["year", "quarter", "period", "category_code", "category_name",
                    "category_name_enc", "q_num", "is_peak_season",
                    "temp_mean_c", "temp_max_c", "precip_mm", "sun_hours",
                    "gdd", "frost_days",
                    "temp_mean_c_lag1", "precip_mm_lag1", "gdd_lag1", "sun_hours_lag1"])
        .agg(cat_total_stuks=("stuks", "sum"), total_opp_ha=("opp_ha", "sum"))
        .reset_index()
    )
    df_agg = df_agg.sort_values(["category_code", "period"])
    for lag in [1, 2, 4]:
        df_agg[f"cat_stuks_lag{lag}"] = (
            df_agg.groupby("category_code")["cat_total_stuks"].shift(lag)
        )
    df_agg["cat_stuks_roll4"] = (
        df_agg.groupby("category_code")["cat_total_stuks"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    )

    agg_feature_cols_full = agg_feature_cols + [
        "total_opp_ha",
        "cat_stuks_lag1", "cat_stuks_lag2", "cat_stuks_lag4",
        "cat_stuks_roll4",
    ]

    print("\n=== Walk-forward CV: aggregate model ===")
    cv_agg = walk_forward_cv(df_agg, agg_feature_cols_full, "cat_total_stuks")
    print(f"\nOverall — MAE: {cv_agg['MAE'].mean():.0f} | "
          f"RMSE: {cv_agg['RMSE'].mean():.0f} | "
          f"MAPE: {cv_agg['MAPE'].mean():.1f}%")

    print("\nTraining final aggregate model on all data …")
    model_agg = train_final_model(df_agg, agg_feature_cols_full, "cat_total_stuks")
    model_agg.save_model(str(MODEL_AGGREGATE_PATH))
    print(f"Saved → {MODEL_AGGREGATE_PATH}")

    # ── Save CV results ──────────────────────────────────────────────────────
    cv_vendor["model"] = "per_vendor"
    cv_agg["model"]    = "aggregate"
    pd.concat([cv_vendor, cv_agg]).to_csv(CV_RESULTS_CSV, index=False)
    print(f"\nCV results → {CV_RESULTS_CSV}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    plot_shap(model_vendor, df, FEATURE_COLS, SHAP_PLOT,
              "Feature importance – per-vendor model (SHAP beeswarm)")

"""
06_evaluate_and_visualise.py

Compares model predictions against reality for any historical period,
and produces a set of visualisation plots.

Usage:
    # Evaluate the last 4 quarters (default)
    python 06_evaluate_and_visualise.py

    # Evaluate a specific year
    python 06_evaluate_and_visualise.py --year 2023

    # Evaluate a specific quarter
    python 06_evaluate_and_visualise.py --year 2023 --quarter Q3

    # Evaluate all quarters for all years (full backtest)
    python 06_evaluate_and_visualise.py --all

Outputs (saved to ./plots/):
    actual_vs_predicted_scatter.png   – scatter plot of all predictions vs actuals
    timeseries_by_category.png        – actual vs predicted over time per category
    error_by_quarter.png              – MAE broken down by quarter (seasonality of errors)
    error_by_vendor.png               – which vendors are hardest to predict
    cv_learning_curve.png             – how model accuracy improved as more data was added
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_CSV      = Path("features.csv")
MODEL_VENDOR_PATH = Path("model_vendor.json")
CV_RESULTS_CSV    = Path("cv_results.csv")
PLOTS_DIR         = Path("plots")

QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

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

# Colour per category – consistent across all plots
CATEGORY_COLOURS = {
    "198 - PAPRIKA DIV.":    "#e63946",
    "200 - PAPRIKA GROEN":   "#2a9d8f",
    "201 - PAPRIKA ROOD":    "#e76f51",
    "202 - PAPRIKA GEEL":    "#e9c46a",
    "203 - PAPRIKA ORANJE":  "#f4a261",
    "204 - PAPRIKA PAARS":   "#9b5de5",
    "205 - PAPRIKA WIT":     "#adb5bd",
    "206 - PAPRIKA MUNT":    "#52b788",
    "207 - PAPRIKA LILA":    "#c77dff",
    "208 - PAPRIKA BRUIN":   "#8d6346",
    "209 - PAPRIKA MIX":     "#457b9d",
    "210 - MINI-PAPRIKA":    "#f72585",
    "217 - PUNTPAPRIKA":     "#4cc9f0",
    "219 - PEPERS":          "#b5e48c",
}
DEFAULT_COLOUR = "#888888"


# ── Helpers ───────────────────────────────────────────────────────────────────

def quarter_to_period(year, quarter, base=2013):
    return (year - base) * 4 + QUARTER_ORDER[quarter] - 1


def period_label(period, base=2013):
    year = base + period // 4
    q    = ["Q1", "Q2", "Q3", "Q4"][period % 4]
    return f"{year}-{q}"


def mape(y_true, y_pred, eps=1):
    mask = np.array(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(
        np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])
    )


def style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent styling to an axes object."""
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="white", linewidth=1.2, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dee2e6")
    ax.spines["bottom"].set_color("#dee2e6")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#555")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#555")
    ax.tick_params(colors="#555", labelsize=9)


# ── Core: generate predictions for a set of periods ──────────────────────────

def generate_predictions(df: pd.DataFrame, model: xgb.XGBRegressor,
                          periods: list[int]) -> pd.DataFrame:
    """
    For each period in `periods`, use only data from BEFORE that period
    as features (lags, rolling stats). The stuks of that period is the
    actual value to compare against.

    This mirrors walk-forward CV: predictions are made period by period
    using only information available at the time.
    """
    results = []

    for period in periods:
        rows = df[df["period"] == period].copy()
        rows = rows.dropna(subset=FEATURE_COLS)
        if rows.empty:
            continue

        X = rows[FEATURE_COLS]
        preds = model.predict(X).clip(0)

        rows = rows.copy()
        rows["predicted_stuks"] = preds.round().astype(int)
        rows["error"]           = rows["predicted_stuks"] - rows["stuks"]
        rows["abs_error"]       = rows["error"].abs()
        rows["period_label"]    = period_label(period)
        results.append(rows)

    if not results:
        raise ValueError("No predictions could be generated for the selected periods.")

    return pd.concat(results, ignore_index=True)


# ── Plot 1: Actual vs Predicted scatter ───────────────────────────────────────

def plot_scatter(pred_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    style_ax(ax,
             title="Actual vs predicted stuks  (per vendor · per quarter)",
             xlabel="Actual stuks",
             ylabel="Predicted stuks")

    categories = pred_df["category_name"].unique()
    for cat in sorted(categories):
        sub = pred_df[pred_df["category_name"] == cat]
        colour = CATEGORY_COLOURS.get(cat, DEFAULT_COLOUR)
        ax.scatter(sub["stuks"], sub["predicted_stuks"],
                   color=colour, alpha=0.55, s=18, label=cat, zorder=3)

    # Perfect prediction line
    max_val = max(pred_df["stuks"].max(), pred_df["predicted_stuks"].max())
    ax.plot([0, max_val], [0, max_val], color="#333", linewidth=1.2,
            linestyle="--", label="Perfect prediction", zorder=4)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Summary metrics in corner
    mae  = pred_df["abs_error"].mean()
    rmse = np.sqrt((pred_df["error"] ** 2).mean())
    mp   = mape(pred_df["stuks"], pred_df["predicted_stuks"])
    ax.text(0.98, 0.04,
            f"MAE  {mae:>10,.0f}\nRMSE {rmse:>10,.0f}\nMAPE {mp:>9.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#dee2e6", alpha=0.9))

    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              markerscale=1.4, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 2: Time series – actual vs predicted per category ────────────────────

def plot_timeseries(pred_df: pd.DataFrame, out_path: Path):
    # Aggregate to category level per period
    agg = (
        pred_df.groupby(["period", "period_label", "category_name"])
        .agg(actual=("stuks", "sum"), predicted=("predicted_stuks", "sum"))
        .reset_index()
        .sort_values("period")
    )

    categories = sorted(agg["category_name"].unique())
    ncols = 2
    nrows = (len(categories) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2), squeeze=False)
    fig.suptitle("Actual vs predicted  –  total stuks per category over time",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, cat in enumerate(categories):
        ax  = axes[i // ncols][i % ncols]
        sub = agg[agg["category_name"] == cat].sort_values("period")
        colour = CATEGORY_COLOURS.get(cat, DEFAULT_COLOUR)

        ax.fill_between(sub["period"], sub["actual"], alpha=0.15, color=colour)
        ax.plot(sub["period"], sub["actual"],
                color=colour, linewidth=2, label="Actual", zorder=3)
        ax.plot(sub["period"], sub["predicted"],
                color=colour, linewidth=1.5, linestyle="--",
                label="Predicted", zorder=3)

        style_ax(ax, title=cat.split(" - ", 1)[-1])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"
                                                           if x >= 1e6 else f"{x:,.0f}"))

        # x-axis: show year labels, not period numbers
        periods_in_sub = sub["period"].values
        year_starts    = [p for p in periods_in_sub if p % 4 == 0]
        ax.set_xticks(year_starts)
        ax.set_xticklabels([period_label(p)[:4] for p in year_starts],
                           rotation=45, ha="right", fontsize=8)

        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    # Hide empty subplots
    for j in range(len(categories), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 3: Error by quarter (seasonal bias) ──────────────────────────────────

def plot_error_by_quarter(pred_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction error by season", fontsize=13, fontweight="bold")

    # Left: MAE per quarter
    ax = axes[0]
    style_ax(ax, title="Mean absolute error by quarter",
             xlabel="Quarter", ylabel="MAE (stuks)")
    mae_q = pred_df.groupby("quarter")["abs_error"].mean().reindex(["Q1","Q2","Q3","Q4"])
    colours = ["#4361ee", "#4cc9f0", "#f77f00", "#e63946"]
    bars = ax.bar(mae_q.index, mae_q.values, color=colours, width=0.5, zorder=3)
    for bar, val in zip(bars, mae_q.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Right: mean error (bias) per quarter – are we over or under-predicting?
    ax2 = axes[1]
    style_ax(ax2, title="Mean error by quarter  (+ = over-prediction)",
             xlabel="Quarter", ylabel="Mean error (stuks)")
    mean_err_q = pred_df.groupby("quarter")["error"].mean().reindex(["Q1","Q2","Q3","Q4"])
    bar_colours = ["#e63946" if v > 0 else "#2a9d8f" for v in mean_err_q.values]
    bars2 = ax2.bar(mean_err_q.index, mean_err_q.values, color=bar_colours,
                    width=0.5, zorder=3)
    ax2.axhline(0, color="#333", linewidth=1)
    for bar, val in zip(bars2, mean_err_q.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (500 if val >= 0 else -1500),
                 f"{val:+,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 4: Error by vendor ───────────────────────────────────────────────────

def plot_error_by_vendor(pred_df: pd.DataFrame, out_path: Path):
    # Only include vendors with meaningful actual deliveries
    vendor_stats = (
        pred_df[pred_df["stuks"] > 0]
        .groupby("vendor")
        .agg(
            mae        = ("abs_error", "mean"),
            total_stuks= ("stuks", "sum"),
            n_quarters = ("stuks", "count"),
        )
        .sort_values("mae", ascending=False)
        .head(20)   # top 20 hardest-to-predict vendors
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(vendor_stats) * 0.45)))
    style_ax(ax, title="Top 20 vendors by mean absolute error",
             xlabel="MAE (stuks)", ylabel="")

    colours = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(vendor_stats))
    )
    bars = ax.barh(vendor_stats["vendor"], vendor_stats["mae"],
                   color=colours, height=0.6, zorder=3)
    for bar, val in zip(bars, vendor_stats["mae"]):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8)

    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 5: CV learning curve ─────────────────────────────────────────────────

def plot_cv_learning_curve(cv_path: Path, out_path: Path):
    if not cv_path.exists():
        print(f"  Skipping learning curve – {cv_path} not found")
        return

    cv = pd.read_csv(cv_path)
    vendor_cv = cv[cv["model"] == "per_vendor"].copy()
    vendor_cv["period_label"] = vendor_cv["val_period"].apply(period_label)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Walk-forward CV: model accuracy over time",
                 fontsize=13, fontweight="bold")

    # Left: MAE over time
    ax = axes[0]
    style_ax(ax, title="MAE per validation fold",
             xlabel="Validated quarter", ylabel="MAE (stuks)")
    ax.plot(vendor_cv["val_period"], vendor_cv["MAE"],
            color="#4361ee", linewidth=1.8, marker="o", markersize=4, zorder=3)
    # Rolling average to show trend
    rolling = vendor_cv["MAE"].rolling(4, min_periods=2).mean()
    ax.plot(vendor_cv["val_period"], rolling,
            color="#e63946", linewidth=2, linestyle="--",
            label="4-fold rolling avg", zorder=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xticks(vendor_cv["val_period"].iloc[::4])
    ax.set_xticklabels(
        [period_label(p)[:4] for p in vendor_cv["val_period"].iloc[::4]],
        rotation=45, ha="right", fontsize=8
    )
    ax.legend(fontsize=9)

    # Right: training set size vs MAE (does more data help?)
    ax2 = axes[1]
    style_ax(ax2, title="Training size vs MAE",
             xlabel="Training samples", ylabel="MAE (stuks)")
    ax2.scatter(vendor_cv["n_train"], vendor_cv["MAE"],
                color="#4361ee", alpha=0.6, s=25, zorder=3)
    # Trend line
    z = np.polyfit(vendor_cv["n_train"], vendor_cv["MAE"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(vendor_cv["n_train"].min(), vendor_cv["n_train"].max(), 100)
    ax2.plot(x_range, p(x_range), color="#e63946", linewidth=1.5,
             linestyle="--", label="Trend", zorder=4)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Print summary table ───────────────────────────────────────────────────────

def print_summary(pred_df: pd.DataFrame):
    print("\n" + "="*65)
    print("  PREDICTION SUMMARY")
    print("="*65)

    # Overall
    mae  = pred_df["abs_error"].mean()
    rmse = np.sqrt((pred_df["error"] ** 2).mean())
    mp   = mape(pred_df["stuks"], pred_df["predicted_stuks"])
    total_actual    = pred_df["stuks"].sum()
    total_predicted = pred_df["predicted_stuks"].sum()

    print(f"\n  Overall (all vendors, all periods evaluated)")
    print(f"  {'MAE':<20} {mae:>12,.0f} stuks")
    print(f"  {'RMSE':<20} {rmse:>12,.0f} stuks")
    print(f"  {'MAPE':<20} {mp:>11.1f}%")
    print(f"  {'Total actual':<20} {total_actual:>12,} stuks")
    print(f"  {'Total predicted':<20} {total_predicted:>12,} stuks")
    print(f"  {'Total bias':<20} {total_predicted - total_actual:>+12,} stuks")

    # Per category
    print(f"\n  {'Category':<30} {'Actual':>12} {'Predicted':>12} {'MAE':>10} {'MAPE':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    cat_stats = (
        pred_df[pred_df["stuks"] > 0]
        .groupby("category_name")
        .apply(lambda g: pd.Series({
            "actual":    g["stuks"].sum(),
            "predicted": g["predicted_stuks"].sum(),
            "mae":       g["abs_error"].mean(),
            "mape":      mape(g["stuks"], g["predicted_stuks"]),
        }))
        .reset_index()
        .sort_values("actual", ascending=False)
    )
    for _, row in cat_stats.iterrows():
        name = row["category_name"].split(" - ", 1)[-1][:28]
        print(f"  {name:<30} {row['actual']:>12,.0f} {row['predicted']:>12,.0f} "
              f"{row['mae']:>10,.0f} {row['mape']:>7.1f}%")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int,  default=None)
    parser.add_argument("--quarter", type=str,  default=None,
                        choices=["Q1", "Q2", "Q3", "Q4"])
    parser.add_argument("--all",     action="store_true",
                        help="Evaluate all historical periods")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)

    # Load data and model
    print("Loading features and model …")
    df = pd.read_csv(FEATURES_CSV)
    df["period"] = df.apply(
        lambda r: quarter_to_period(r["year"], r["quarter"]), axis=1
    )

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_VENDOR_PATH))

    # Decide which periods to evaluate
    all_periods = sorted(df["period"].unique())

    # Only evaluate periods that have enough lag history (need at least lag-4)
    min_period = 4
    valid_periods = [p for p in all_periods if p >= min_period]

    if args.all:
        eval_periods = valid_periods
        label = "all historical quarters"

    elif args.year and args.quarter:
        target_period = quarter_to_period(args.year, args.quarter)
        if target_period not in valid_periods:
            raise ValueError(f"{args.year} {args.quarter} not in data or insufficient lag history")
        eval_periods = [target_period]
        label = f"{args.year} {args.quarter}"

    elif args.year:
        eval_periods = [
            quarter_to_period(args.year, q)
            for q in ["Q1", "Q2", "Q3", "Q4"]
            if quarter_to_period(args.year, q) in valid_periods
        ]
        label = f"all of {args.year}"

    else:
        # Default: last 4 quarters with actual data (stuks > 0 exists)
        has_data = df[df["stuks"] > 0]["period"].unique()
        last_4   = sorted(has_data)[-4:]
        eval_periods = [p for p in last_4 if p in valid_periods]
        label = "last 4 quarters"

    print(f"Evaluating: {label}  ({len(eval_periods)} periods, "
          f"{', '.join(period_label(p) for p in eval_periods[:4])}"
          f"{'…' if len(eval_periods) > 4 else ''})")

    # Generate predictions
    pred_df = generate_predictions(df, model, eval_periods)

    # Print summary
    print_summary(pred_df)

    # Save prediction table
    pred_csv = PLOTS_DIR / f"predictions_{label.replace(' ', '_')}.csv"
    pred_df[["year", "quarter", "category_name", "vendor",
             "stuks", "predicted_stuks", "error", "abs_error"]].to_csv(pred_csv, index=False)
    print(f"  Prediction table → {pred_csv}")

    # Plots
    print("\nGenerating plots …")
    plot_scatter(pred_df,        PLOTS_DIR / "actual_vs_predicted_scatter.png")
    plot_timeseries(pred_df,     PLOTS_DIR / "timeseries_by_category.png")
    plot_error_by_quarter(pred_df, PLOTS_DIR / "error_by_quarter.png")
    plot_error_by_vendor(pred_df,  PLOTS_DIR / "error_by_vendor.png")
    plot_cv_learning_curve(CV_RESULTS_CSV, PLOTS_DIR / "cv_learning_curve.png")

    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
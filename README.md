# Bell Pepper Yield Forecasting Pipeline

Predicts quarterly bell pepper quantities (stuks) per vendor, category, and in aggregate,
using historical harvest data combined with Belgian weather data.

---

## Project structure

```
bellpepper_pipeline/
├── run_pipeline.py           ← master runner (start here)
├── 01_parse_excel.py         ← parse pivot Excel → long CSV
├── 02_fetch_weather.py       ← fetch historical weather from Open-Meteo
├── 03_feature_engineering.py ← build feature matrix (lags, rolling stats, weather merge)
├── 04_train_model.py         ← walk-forward CV + XGBoost training + SHAP
├── 05_forecast.py            ← generate forecasts for any (year, quarter)
└── README.md
```

Data files generated at runtime (in the same folder):
```
bellpepper_long.csv        ← parsed long-format data
weather_quarterly.csv      ← quarterly weather aggregates (Belgium)
features.csv               ← merged feature matrix
model_vendor.json          ← trained per-vendor XGBoost model
model_aggregate.json       ← trained aggregate XGBoost model
cv_results.csv             ← walk-forward cross-validation metrics
shap_summary.png           ← SHAP feature importance plot
forecast_YYYY_QN.csv       ← per-vendor forecast output
forecast_agg_YYYY_QN.csv   ← category-aggregate forecast output
```

---

## Setup

```bash
pip install openpyxl xgboost lightgbm shap scikit-learn pandas numpy matplotlib requests
```

Copy your Excel file (`bellpepers.xlsx`) one level above this folder, or adjust
`EXCEL_PATH` in `01_parse_excel.py`.

---

## Running the full pipeline

```bash
# Full run: parse → weather → features → train → forecast Q2 2026
python run_pipeline.py --year 2026 --quarter Q2

# Skip training (models already saved), only forecast
python run_pipeline.py --forecast-only --year 2026 --quarter Q3

# 2-week-ahead / early mode: uses only last quarter's weather as a proxy
# (available ~2 weeks before the quarter starts)
python run_pipeline.py --forecast-only --year 2026 --quarter Q2 --early
```

---

## How each step works

### Step 1 – Parse Excel
The source Excel is a pivot table with a 3-level row hierarchy:
- Category code (198, 200, 201, 202, 203)
- Category name (Paprika Div., Paprika Groen, Rood, Geel, Oranje)
- Vendor (e.g. "P  17 - GROBAR")

Columns repeat for each year × quarter with `Sum of Stuks` (quantity) and
`Sum of Opp_Ha` (area in hectares, constant per vendor).

Output: one row per (vendor, category, year, quarter).

### Step 2 – Weather
Fetches daily historical weather from the free Open-Meteo archive API for the
Mechelen/Antwerp region (centre of Belgian bell pepper cultivation).

Aggregated per quarter:
- `temp_mean_c`, `temp_max_c` – temperature
- `precip_mm` – total precipitation
- `sun_hours` – sunshine duration
- `gdd` – growing degree days (base 10°C, standard for vegetables)
- `frost_days` – days below 0°C

**To add your own weather data**: replace or extend `weather_quarterly.csv`
with the same column format.

### Step 3 – Feature engineering
Key features:
- **Lag features** (lag 1, 2, 4 quarters): most important signal; last quarter's
  yield is the single strongest predictor.
- **Rolling statistics**: 4-quarter trailing mean and standard deviation per
  (vendor, category).
- **Weather**: current-quarter and lagged-quarter aggregates.
- **Categorical encoding**: vendor and category as integer codes (XGBoost-native).
- **Area (opp_ha)**: constant per vendor, scales raw quantity to yield per Ha.

### Step 4 – Model training
- Algorithm: **XGBoost** (gradient boosted trees)
- Evaluation: **Walk-forward cross-validation** — trains on periods 0..t,
  validates on t+1. This strictly prevents future leakage.
- Metrics: MAE, RMSE, MAPE per fold.
- SHAP values are computed on a random sample to explain which features drive
  predictions most.

Two models:
1. **Per-vendor model**: predicts `stuks` for each (vendor, category, quarter).
2. **Aggregate model**: predicts total `stuks` per (category, quarter) using
   category-level features.

### Step 5 – Forecasting
Two modes:
- **Quarterly mode** (`--quarter QN`): uses current-quarter weather if available,
  otherwise falls back to previous quarter's weather.
- **Early / 2-week-ahead mode** (`--early`): always uses the previous quarter's
  weather as a proxy. This simulates predicting 2 weeks before the quarter starts,
  when the current quarter's weather is not yet complete.

---

## Improving the model over time

1. **Add weekly data**: run step 1 on weekly data (if you get it), aggregate
   to quarters in step 3, or train a separate weekly-granularity model.
2. **Add forecast weather**: once you have a weather forecast for the upcoming
   quarter, use those values instead of the lagged proxy in early mode.
3. **Tune hyperparameters**: use `optuna` to run a Bayesian search over
   XGBoost parameters, using walk-forward CV as the objective.
4. **Per-category models**: train a separate model per category if the
   overall model's MAPE is high for a specific colour.
5. **Retrain regularly**: call `04_train_model.py` after each quarter of new
   data is added to the Excel.

---

## Notes on missing data
- Vendors that don't deliver in a quarter have `stuks = 0` (not NaN).
- Rows with no area (`opp_ha = NaN`) are kept but will have `NaN` lag features
  until enough history accumulates (typically after 2 quarters).
- The model handles NaN inputs natively (XGBoost uses a learned default direction
  for missing values at each split).

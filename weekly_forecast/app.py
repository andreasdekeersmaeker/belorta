"""
app.py — Bell Pepper Weekly Incoming Weight Forecast
=====================================================
Run with:  streamlit run app.py

Requires:
  model/xgb_model.pkl          ← produced by train.py
  data/historical_weights.csv  ← your training dataset (for lag-52 lookups)
"""

import pickle, io, warnings, os
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
LAT, LON = 51.028, 4.480   # Mechelen, Belgium
MODEL_PATH   = "model/xgb_model.pkl"
HISTORY_PATH = "data/historical_weights.csv"
USER_DATA_PATH = "data/user_data.csv"

TEMPLATE_CSV = "year,week,total_kg\n2024,10,450000\n2024,11,512000\n2024,12,488000\n"

# ── Weather helpers ────────────────────────────────────────────────────────────

def _fetch_json(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def fetch_recent_and_forecast_weather():
    """
    Returns daily DataFrame [date, tmean, precip, rad, gdd]
    covering the last 60 days (historical) + next 16 days (forecast).
    Cached for 1 hour.
    """
    today = date.today()
    # default range: last 70 days up to today
    hist_end_dt = today
    hist_start_dt = today - timedelta(days=70)

    # Historical (Open-Meteo archive) — try with today, on failure retry with end date -1 day
    try:
        hist = _fetch_json("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": LAT, "longitude": LON,
            "start_date": hist_start_dt.isoformat(), "end_date": hist_end_dt.isoformat(),
            "daily": ["temperature_2m_mean", "precipitation_sum", "shortwave_radiation_sum"],
            "timezone": "Europe/Brussels",
        })
    except Exception:
        # retry with end date one day earlier (some archive endpoints fail for today's date)
        try:
            hist_end_dt = today - timedelta(days=1)
            hist_start_dt = hist_end_dt - timedelta(days=70)
            hist = _fetch_json("https://archive-api.open-meteo.com/v1/archive", {
                "latitude": LAT, "longitude": LON,
                "start_date": hist_start_dt.isoformat(), "end_date": hist_end_dt.isoformat(),
                "daily": ["temperature_2m_mean", "precipitation_sum", "shortwave_radiation_sum"],
                "timezone": "Europe/Brussels",
            })
        except Exception:
            # re-raise original failure to be handled by caller
            raise

    # Forecast (Open-Meteo forecast, free tier = 16 days)
    fcast = _fetch_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON,
        "daily": ["temperature_2m_mean", "precipitation_sum", "shortwave_radiation_sum"],
        "timezone": "Europe/Brussels",
        "forecast_days": 16,
    })

    rows = []
    for src in [hist, fcast]:
        d = src["daily"]
        for i, dt in enumerate(d["time"]):
            rows.append({
                "date":   pd.to_datetime(dt),
                "tmean":  d["temperature_2m_mean"][i],
                "precip": d["precipitation_sum"][i],
                "rad":    d["shortwave_radiation_sum"][i],
            })

    df = (pd.DataFrame(rows)
          .drop_duplicates("date")
          .sort_values("date")
          .reset_index(drop=True))
    df["gdd"] = (df["tmean"] - 10).clip(lower=0)
    return df


def aggregate_weather_weekly(df_daily):
    df = df_daily.copy()
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df.groupby(["year", "week"]).agg(
        tmean  = ("tmean",  "mean"),
        precip = ("precip", "sum"),
        rad    = ("rad",    "sum"),
        gdd    = ("gdd",    "sum"),
    ).reset_index()


# ── Data helpers ──────────────────────────────────────────────────────────────

def normalise_weights_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accept flexible column names; return standard year/week/total_kg."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for c in df.columns:
        if "year" in c:
            col_map[c] = "year"
        elif "week" in c:
            col_map[c] = "week"
        elif any(k in c for k in ["gewicht", "kg", "weight", "total"]):
            col_map[c] = "total_kg"
    df = df.rename(columns=col_map)
    missing = {"year", "week", "total_kg"} - set(df.columns)
    if missing:
        raise ValueError(f"Could not find columns: {missing}. Expected year, week, total_kg.")
    return (df[["year", "week", "total_kg"]]
            .dropna()
            .astype({"year": int, "week": int, "total_kg": float}))


@st.cache_data
def load_training_history():
    """Load the bundled historical dataset (used for lag-52 fallback)."""
    try:
        df = pd.read_csv(HISTORY_PATH)
        return normalise_weights_df(df)
    except Exception:
        return pd.DataFrame(columns=["year", "week", "total_kg"])


def load_user_data():
    """Load user-saved delivery data from disk."""
    try:
        if os.path.exists(USER_DATA_PATH):
            df = pd.read_csv(USER_DATA_PATH)
            return normalise_weights_df(df)
    except Exception:
        pass
    return pd.DataFrame(columns=["year", "week", "total_kg"])


def save_user_data(df: pd.DataFrame):
    """Save user delivery data to disk."""
    try:
        os.makedirs("data", exist_ok=True)
        df.to_csv(USER_DATA_PATH, index=False)
        return True
    except Exception:
        return False


def get_user_data_range():
    """Return (min_year, min_week, max_year, max_week) of saved user data, or None."""
    saved = load_user_data()
    if len(saved) == 0:
        return None
    saved = saved.sort_values(["year", "week"])
    first = saved.iloc[0]
    last = saved.iloc[-1]
    return (int(first["year"]), int(first["week"]), int(last["year"]), int(last["week"]))


def get_training_data_range():
    """Return (min_year, min_week, max_year, max_week) of training history, or None."""
    history = load_training_history()
    if len(history) == 0:
        return None
    history = history.sort_values(["year", "week"])
    first = history.iloc[0]
    last = history.iloc[-1]
    return (int(first["year"]), int(first["week"]), int(last["year"]), int(last["week"]))


def merge_histories(base_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    """User data takes priority over training history for overlapping weeks."""
    combined = pd.concat([base_df, user_df]).drop_duplicates(
        subset=["year", "week"], keep="last"
    )
    return combined.sort_values(["year", "week"]).reset_index(drop=True)


# ── Feature builder ───────────────────────────────────────────────────────────

def prev_yw(year, week):
    return (year - 1, 52) if week == 1 else (year, week - 1)


def nth_prev(year, week, n):
    y, w = year, week
    for _ in range(n):
        y, w = prev_yw(y, w)
    return y, w


def lookup_kg(history, year, week):
    row = history[(history["year"] == year) & (history["week"] == week)]
    return float(row["total_kg"].iloc[0]) if len(row) > 0 else np.nan


def lookup_wx(weather_weekly, year, week, col):
    row = weather_weekly[(weather_weekly["year"] == year) & (weather_weekly["week"] == week)]
    return float(row[col].iloc[0]) if len(row) > 0 else np.nan


def build_feature_row(history, weather_weekly, target_year, target_week, year_min):
    lags = {n: nth_prev(target_year, target_week, n) for n in [1, 2, 4]}
    gdd_roll4 = sum(
        lookup_wx(weather_weekly, *nth_prev(target_year, target_week, i), "gdd")
        for i in range(1, 5)
    )
    return {
        "lag1":        lookup_kg(history, *lags[1]),
        "lag2":        lookup_kg(history, *lags[2]),
        "lag4":        lookup_kg(history, *lags[4]),
        "tmean_lag1":  lookup_wx(weather_weekly, *lags[1], "tmean"),
        "tmean_lag2":  lookup_wx(weather_weekly, *lags[2], "tmean"),
        "precip_lag1": lookup_wx(weather_weekly, *lags[1], "precip"),
        "precip_lag2": lookup_wx(weather_weekly, *lags[2], "precip"),
        "gdd_lag1":    lookup_wx(weather_weekly, *lags[1], "gdd"),
        "gdd_lag2":    lookup_wx(weather_weekly, *lags[2], "gdd"),
        "rad_lag1":    lookup_wx(weather_weekly, *lags[1], "rad"),
        "rad_lag2":    lookup_wx(weather_weekly, *lags[2], "rad"),
        "gdd_roll4":   gdd_roll4,
        "week_sin":    np.sin(2 * np.pi * target_week / 52),
        "week_cos":    np.cos(2 * np.pi * target_week / 52),
        "year_trend":  (target_year - year_min) / 10.0,
    }


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_week(model_pkg, history, weather_weekly, target_year, target_week):
    # Ensure history and weather_weekly have expected columns to avoid KeyError during lookups
    if history is None or "year" not in history.columns:
        history = pd.DataFrame(columns=["year", "week", "total_kg"])
    if weather_weekly is None or "year" not in weather_weekly.columns:
        weather_weekly = pd.DataFrame(columns=["year", "week", "tmean", "precip", "rad", "gdd"])

    feat_row = build_feature_row(
        history, weather_weekly, target_year, target_week,
        year_min=model_pkg["train_years"][0],
    )
    n_missing = sum(1 for v in feat_row.values() if isinstance(v, float) and np.isnan(v))
    clean = {k: (0.0 if isinstance(v, float) and np.isnan(v) else v) for k, v in feat_row.items()}
    X = pd.DataFrame([clean])[model_pkg["feature_cols"]]
    pred_kg = float(np.expm1(model_pkg["model"].predict(X)[0]))
    return pred_kg, n_missing, feat_row


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="🫑 Bell Pepper Forecast", layout="wide")
st.title("🫑 Bell Pepper Incoming Weight Forecast")
st.caption("Weekly total incoming weight prediction across all vendors · Weather: Mechelen, Belgium (auto-fetched)")

# Load model
try:
    model_pkg = load_model()
    model_ok  = True
except FileNotFoundError:
    model_ok  = False
    st.error("⚠️ No trained model found. Run `python train.py` first.", icon="🚨")

base_history = load_training_history()

# Initialize session state
if "user_df" not in st.session_state:
    st.session_state.user_df = load_user_data()
if "last_saved" not in st.session_state:
    st.session_state.last_saved = None

# Get data ranges for sidebar caption
user_range = get_user_data_range()
training_range = get_training_data_range()

if user_range:
    y1, w1, y2, w2 = user_range
    user_info = f"{y1}-W{w1:02d} to {y2}-W{w2:02d}"
else:
    user_info = "(none)"

if training_range:
    y1, w1, y2, w2 = training_range
    training_info = f"{y1}-W{w1:02d} to {y2}-W{w2:02d}"
else:
    training_info = "(none)"

data_info = f"**User data:** {user_info} · **Training data:** {training_info}"

# ────────────────── Sidebar: data input ──────────────────────────────────────
with st.sidebar:
    st.header("📋 Your delivery data")
    caption_text = (
        f"Provide recent actual incoming weights. "
        f"**4 weeks minimum**.\n\n"
        f"**{data_info}**"
    )
    st.caption(caption_text)

    method = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)
    user_df = None

    if method == "Manual entry":
        today = date.today()
        cy, cw, _ = today.isocalendar()
        st.caption(f"Current week: {cy}-W{cw:02d}")
        
        # Build default rows (last 8 weeks)
        default_rows = []
        y, w = cy, cw
        for _ in range(8):
            y, w = prev_yw(y, w)
            default_rows.insert(0, {"year": y, "week": w, "total_kg": 0.0})
        
        # Overlay saved data on default rows
        default_df = pd.DataFrame(default_rows)
        saved_data = st.session_state.user_df
        if len(saved_data) > 0:
            # Merge: keep all rows, prioritize saved data
            merged = pd.concat([default_df, saved_data]).drop_duplicates(
                subset=["year", "week"], keep="last"
            ).sort_values(["year", "week"]).reset_index(drop=True)
        else:
            merged = default_df

        edited = st.data_editor(
            merged,
            num_rows="dynamic",
            column_config={
                "year":     st.column_config.NumberColumn("Year",     min_value=2000, max_value=2035, step=1),
                "week":     st.column_config.NumberColumn("ISO Week", min_value=1,    max_value=53,   step=1),
                "total_kg": st.column_config.NumberColumn("Weight (kg)", min_value=0, format="%d"),
            },
            width="stretch",
        )
        filled = edited[edited["total_kg"] > 0].copy()
        if len(filled) > 0:
            user_df = filled.astype({"year": int, "week": int, "total_kg": float})
            st.session_state.user_df = user_df
        
        # Save button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save to memory", type="primary", width="stretch"):
                if user_df is not None and len(user_df) > 0:
                    if save_user_data(user_df):
                        st.session_state.last_saved = user_df
                        st.success("✅ Data saved to memory!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save data")
                else:
                    st.warning("⚠️ No data to save")
        with col2:
            if st.button("🗑️ Clear memory", width="stretch"):
                if os.path.exists(USER_DATA_PATH):
                    os.remove(USER_DATA_PATH)
                st.session_state.user_df = pd.DataFrame(columns=["year", "week", "total_kg"])
                st.info("🗑️ Memory cleared")
                st.rerun()

    else:
        st.markdown("Upload a CSV with columns: `year`, `week`, `total_kg`")
        with st.expander("📄 Download template"):
            st.download_button("⬇️ template.csv", data=TEMPLATE_CSV,
                               file_name="template.csv", mime="text/csv")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                raw = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
                user_df = normalise_weights_df(raw)
                st.success(f"✅ {len(user_df)} rows loaded ({user_df['year'].min()}–{user_df['year'].max()})")
            except Exception as e:
                st.error(f"Could not parse: {e}")

    if user_df is not None and len(user_df) >= 4:
        st.success(f"✅ {len(user_df)} week(s) of data ready")
    elif user_df is not None:
        st.warning(f"⚠️ Only {len(user_df)} week(s). Add more for better accuracy.")


# ────────────────── Main area ─────────────────────────────────────────────────
col_chart, col_wx = st.columns([3, 2])

with col_chart:
    st.subheader("📊 Provided delivery data")
    if user_df is not None and len(user_df) > 0:
        plot = user_df.copy()
        plot["label"] = plot["year"].astype(str) + "-W" + plot["week"].astype(str).str.zfill(2)
        st.line_chart(plot.set_index("label")["total_kg"], width="stretch")
    else:
        st.info("Enter or upload delivery data in the sidebar.")

with col_wx:
    st.subheader("🌤️ Current weather (Mechelen)")
    weather_weekly = pd.DataFrame()
    weather_ok = False
    status_box = st.empty()

    try:
        with st.spinner("Fetching from Open-Meteo..."):
            df_daily = fetch_recent_and_forecast_weather()
            weather_weekly = aggregate_weather_weekly(df_daily)
        weather_ok = True

        td = date.today()
        cy, cw, _ = td.isocalendar()
        cur = weather_weekly[(weather_weekly["year"] == cy) & (weather_weekly["week"] == cw)]
        if len(cur) > 0:
            r = cur.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Avg temp", f"{r['tmean']:.1f} °C")
            c2.metric("🌧️ Precip",   f"{r['precip']:.0f} mm")
            c3.metric("🌱 GDD",       f"{r['gdd']:.0f}")
            status_box.success(f"✅ Live weather · Week {cy}-W{cw:02d}")
        else:
            status_box.success("✅ Weather loaded")

    except Exception as e:
        status_box.warning(f"⚠️ Weather unavailable: {e}")


# ────────────────── Forecast ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Forecast")

c1, c2, _ = st.columns([1, 1, 3])
with c1:
    horizon = c1.selectbox("Horizon", ["Next week (t+1)", "Next 2 weeks (t+1 & t+2)"])
with c2:
    c2.write("")
    go = c2.button(
        "▶ Run forecast", type="primary",
        disabled=(not model_ok or user_df is None or len(user_df) < 1),
        width="stretch",
    )

if go:
    today = date.today()
    cy, cw, _ = today.isocalendar()
    n_weeks = 2 if "2 weeks" in horizon else 1
    # If the user didn't edit/upload in this session, fall back to saved user data in session_state
    effective_user_df = user_df if user_df is not None else st.session_state.get("user_df", pd.DataFrame(columns=["year","week","total_kg"]))
    # Ensure integer types for reliable lookups
    if len(effective_user_df) > 0:
        effective_user_df = effective_user_df.astype({"year": int, "week": int, "total_kg": float})
    combined = merge_histories(base_history, effective_user_df)

    # Use the latest week in combined history as the anchor,
    # so lag1 = that latest week (which the user has entered)
    last_row = combined.sort_values(["year", "week"]).iloc[-1]
    anchor_year, anchor_week = int(last_row["year"]), int(last_row["week"])

    results = []
    for offset in range(1, n_weeks + 1):
        ty, tw = anchor_year, anchor_week + offset
        if tw > 52:
            tw -= 52; ty += 1
        try:
            pred_kg, n_miss, feat_row = predict_week(
                model_pkg, combined, weather_weekly, ty, tw
            )
            results.append(dict(label=f"{ty}-W{tw:02d}", pred=pred_kg,
                                n_miss=n_miss, feat=feat_row))
            
            # Add prediction to combined history for next iteration's lags
            # This ensures multi-week forecasts correctly use predicted values:
            #  - lag1 (1 week prior) comes from the prediction we just made
            #  - lag2, lag4, lag52 (2, 4, 52 weeks prior) come from actual historical data
            if offset < n_weeks:
                new_row = pd.DataFrame([{"year": int(ty), "week": int(tw), "total_kg": float(pred_kg)}])
                combined = pd.concat([combined, new_row]).drop_duplicates(
                    subset=["year", "week"], keep="last"
                ).reset_index(drop=True)
        except Exception as e:
            st.error(f"Prediction failed (week + {offset}): {e}")

    if results:
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                lo, hi = res["pred"] * 0.72, res["pred"] * 1.28
                st.metric(f"📦 {res['label']}",
                          f"{res['pred']:,.0f} kg",
                          f"{res['pred']/1000:.1f} t")
                st.caption(f"Estimated range: {lo:,.0f} – {hi:,.0f} kg")
                if res["n_miss"] > 0:
                    st.caption(f"⚠️ {res['n_miss']} lag(s) missing (set to 0)")

        with st.expander("🔍 Feature values (first forecast week)"):
            fd = pd.DataFrame([
                {"Feature": k,
                 "Value": f"{v:,.2f}" if isinstance(v, float) and not np.isnan(v) else "missing"}
                for k, v in results[0]["feat"].items()
            ])
            st.dataframe(fd, hide_index=True, width="stretch")

        st.caption(
            "Range = ±28% based on walk-forward CV MAPE on active production weeks 2017–2025. "
            "Model: XGBoost · Weather source: Open-Meteo (Mechelen, Belgium)."
        )

elif not model_ok:
    st.info("Run `python train.py` to train the model first.")
elif user_df is None or len(user_df) < 1:
    st.info("Add at least one week of delivery data in the sidebar.")


# ────────────────── Footer ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Setup & usage guide"):
    st.markdown("""
**Installation:**
```bash
pip install streamlit xgboost scikit-learn pandas requests openpyxl
```

**First run:**
```bash
# 1. Place your historical data (year, week, total_kg) at:
#    data/historical_weights.csv
#
# 2. Train the model (fetches real weather automatically):
python train.py

# 3. Launch the app:
streamlit run app.py
```

**CSV format:**
```
year,week,total_kg
2024,10,450000
2024,11,512000
2024,12,488000
```

**Retraining:** add rows to `data/historical_weights.csv` and run `python train.py` again.

**How accuracy works:**
- `lag52` (same week last year) is the strongest predictor — provide at least 52 weeks of history
- Temperature and GDD from the prior 2 weeks are the most important weather features
- Active season accuracy (weeks with >10 t): ~28% MAPE on 2017–2025 holdout
""")

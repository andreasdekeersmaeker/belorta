"""
Step 2 – Fetch historical quarterly weather for Belgium via Open-Meteo.

Uses the ERA5-based historical archive (no API key needed).
Location: Mechelen / Antwerp region – centre of Belgian bell pepper cultivation.

Aggregates daily data to quarterly summaries:
  temp_mean_c    – mean daily temperature (°C)
  temp_max_c     – mean of daily max temps (°C)
  precip_mm      – total precipitation (mm)
  sun_hours      – total sunshine duration (hours)
  gdd            – growing degree days (base 10°C, standard for vegetables)
  frost_days     – days with min temp < 0°C

Output: weather_quarterly.csv
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_CSV = Path("weather_quarterly.csv")

# Mechelen, centre of Belgian paprika belt
LAT  = 51.03
LON  = 4.48

# Covered years  ← expand as needed
START_DATE = "2013-01-01"
END_DATE   = "2025-12-31"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude":   LAT,
    "longitude":  LON,
    "start_date": START_DATE,
    "end_date":   END_DATE,
    "daily": ",".join([
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "sunshine_duration",   # seconds
    ]),
    "timezone": "Europe/Brussels",
}

QUARTER_MAP = {1: "Q1", 2: "Q1", 3: "Q1",
               4: "Q2", 5: "Q2", 6: "Q2",
               7: "Q3", 8: "Q3", 9: "Q3",
               10: "Q4", 11: "Q4", 12: "Q4"}

BASE_TEMP = 10.0  # GDD base temperature for bell peppers


def fetch_weather() -> pd.DataFrame:
    print("Fetching weather data from Open-Meteo …")
    r = requests.get(OPEN_METEO_URL, params=PARAMS, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "time":                   "date",
        "temperature_2m_mean":    "tmean",
        "temperature_2m_max":     "tmax",
        "temperature_2m_min":     "tmin",
        "precipitation_sum":      "precip",
        "sunshine_duration":      "sun_sec",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["month"].map(QUARTER_MAP)

    # Growing degree days: max(0, Tmean – base)
    df["gdd_day"]    = (df["tmean"] - BASE_TEMP).clip(lower=0)
    df["frost_day"]  = (df["tmin"] < 0).astype(int)
    df["sun_hours"]  = df["sun_sec"] / 3600

    agg = (
        df.groupby(["year", "quarter"])
        .agg(
            temp_mean_c  = ("tmean",     "mean"),
            temp_max_c   = ("tmax",      "mean"),
            precip_mm    = ("precip",    "sum"),
            sun_hours    = ("sun_hours", "sum"),
            gdd          = ("gdd_day",   "sum"),
            frost_days   = ("frost_day", "sum"),
        )
        .reset_index()
    )
    return agg


if __name__ == "__main__":
    df = fetch_weather()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} quarter-rows → {OUTPUT_CSV}")
    print(df.head(8).to_string())

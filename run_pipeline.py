"""
run_pipeline.py – Orchestrates all 5 steps of the bell pepper forecasting pipeline.

Usage:
  # Run full pipeline (parse → weather → features → train → forecast)
  python run_pipeline.py

  # Only forecast (models already trained)
  python run_pipeline.py --forecast-only --year 2026 --quarter Q2

  # Early mode (2-week-ahead forecast, uses only last quarter's weather)
  python run_pipeline.py --forecast-only --year 2026 --quarter Q2 --early
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent


def run(script: str, extra_args: list[str] = None):
    cmd = [sys.executable, str(PIPELINE_DIR / script)] + (extra_args or [])
    print(f"\n{'─'*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=PIPELINE_DIR)
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-only", action="store_true",
                        help="Skip parsing/training, only run forecast")
    parser.add_argument("--year",    type=int, default=2026)
    parser.add_argument("--quarter", type=str, default="Q2",
                        choices=["Q1", "Q2", "Q3", "Q4"])
    parser.add_argument("--early",   action="store_true",
                        help="Use only last-quarter weather for 2-week-ahead mode")
    args = parser.parse_args()

    if not args.forecast_only:
        # Step 1 – parse Excel
        run("01_parse_excel.py")

        # Step 2 – fetch weather from Open-Meteo
        run("02_fetch_weather.py")

        # Step 3 – feature engineering
        run("03_feature_engineering.py")

        # Step 4 – train models + walk-forward CV
        run("04_train_model.py")

    # Step 5 – forecast
    forecast_args = ["--year", str(args.year), "--quarter", args.quarter]
    if args.early:
        forecast_args.append("--early")
    run("05_forecast.py", forecast_args)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()

# Logistics Booking Demand Forecasting

This project prepares booking records from the Excel workbook, evaluates
feature-based models when enough daily history is available, and writes a
90-day daily forecast with a monthly summary.

## Project setup

From the project root, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the pipeline

```powershell
python -m src.run_pipeline
```

The pipeline streams booking-date cells from all worksheets in
`data/raw/BK_AUG2026.xlsx`, counts transaction rows by booking date, and
filters out rows outside the workbook's expected August 2026 period. It fills
unrecorded calendar dates with zero demand and saves the rebuilt daily and
feature-engineered datasets under `data/processed/`.

When at least 10 complete feature rows remain after lag creation, the pipeline
trains and evaluates the configured models. For short histories, it tries a
reduced feature set using one- and seven-day lags before skipping ML evaluation.
With limited history, holdout metrics are exploratory and unstable; the
weekday-average baseline remains available as a reference.

Generated outputs:

- `outputs/forecasts/daily_90_day_forecast.csv`
- `outputs/forecasts/monthly_forecast_summary.csv`
- `outputs/forecasts/model_training_predictions.csv` when ML evaluation runs
- `outputs/reports/evaluation_metrics.csv` when ML evaluation runs
- `outputs/pipeline_status.json`

Start the Streamlit dashboard with:

```powershell
python -m streamlit run app/app.py
```

## Data limitations

The supplied workbook contains one month of 2026 bookings split across the
`1-15` and `16-31` worksheets. Even when reduced features make ML evaluation
possible, one month is too short for reliable model selection or longer-term
forecasting. Forecasts inherit the data's coverage and quality limitations.


from pathlib import Path
import json
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent

DAILY_FILE = PROJECT_DIR / "data" / "processed" / "daily_booking_demand.csv"
PREDICTION_FILE = (
    PROJECT_DIR / "outputs" / "forecasts" / "model_training_predictions.csv"
)
FORECAST_FILE = (
    PROJECT_DIR / "outputs" / "forecasts" / "daily_90_day_forecast.csv"
)
MONTHLY_FORECAST_FILE = (
    PROJECT_DIR / "outputs" / "forecasts" / "monthly_forecast_summary.csv"
)
METRICS_FILE = PROJECT_DIR / "outputs" / "reports" / "evaluation_metrics.csv"
STATUS_FILE = PROJECT_DIR / "outputs" / "pipeline_status.json"

st.set_page_config(
    page_title="Logistics Booking Forecasting",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Logistics Booking Demand Forecasting")
st.write(
    "AI-Driven Time-Series Forecasting and "
    "Predictive Analytics for Logistics Booking Demand"
)

# Historical demand
st.header("1. Historical Booking Demand")

if DAILY_FILE.exists():
    df = pd.read_csv(DAILY_FILE)

    if {"DATE", "BOOKING_COUNT"}.issubset(df.columns):
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["BOOKING_COUNT"] = pd.to_numeric(
            df["BOOKING_COUNT"], errors="coerce"
        )
        df = df.dropna(
            subset=["DATE", "BOOKING_COUNT"]
        ).sort_values("DATE")

        if not df.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily records", len(df))
            col2.metric("Total bookings", f"{df['BOOKING_COUNT'].sum():,.0f}")
            col3.metric("Average bookings/day", f"{df['BOOKING_COUNT'].mean():,.1f}")

            st.line_chart(df.set_index("DATE")["BOOKING_COUNT"])
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.warning("The daily dataset has no valid rows.")
    else:
        st.error("Dataset must contain DATE and BOOKING_COUNT columns.")
else:
    st.warning(
        "Daily dataset not found. Prepare "
        "data/processed/daily_booking_demand.csv first."
    )

# Evaluation metrics
st.header("2. Model Evaluation")

pipeline_status = (
    json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    if STATUS_FILE.exists()
    else None
)
ml_evaluated = (
    pipeline_status["ml_evaluated"]
    if pipeline_status is not None
    else True
)

if not ml_evaluated:
    st.info(
        "ML evaluation was skipped because the available history is too "
        "short after lag-feature creation. The future forecast uses the "
        "weekday-average baseline."
    )
elif METRICS_FILE.exists():
    if pipeline_status and pipeline_status.get("evaluation_note"):
        st.warning(pipeline_status["evaluation_note"])
    metrics = pd.read_csv(METRICS_FILE)
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Evaluation results will appear after model evaluation.")

# Predictions
st.header("3. Test-Set Predictions")

if ml_evaluated and PREDICTION_FILE.exists():
    predictions = pd.read_csv(PREDICTION_FILE)
    st.dataframe(predictions.head(100), use_container_width=True)
else:
    st.info(
        "Predictions will appear after model training succeeds."
    )

st.caption("Results are loaded from your saved project files.")

# Future demand forecast
st.header("4. 90-Day Demand Forecast")

if FORECAST_FILE.exists():
    forecast = pd.read_csv(FORECAST_FILE)
    if {"DATE", "FORECAST_BOOKING_COUNT"}.issubset(forecast.columns):
        forecast["DATE"] = pd.to_datetime(forecast["DATE"], errors="coerce")
        forecast["FORECAST_BOOKING_COUNT"] = pd.to_numeric(
            forecast["FORECAST_BOOKING_COUNT"], errors="coerce"
        )
        forecast = forecast.dropna(
            subset=["DATE", "FORECAST_BOOKING_COUNT"]
        ).sort_values("DATE")

        if not forecast.empty:
            method = (
                str(forecast["FORECAST_METHOD"].iloc[0])
                if "FORECAST_METHOD" in forecast.columns
                else "forecast"
            )
            st.caption(f"Forecast method: {method}")
            st.line_chart(
                forecast.set_index("DATE")["FORECAST_BOOKING_COUNT"]
            )
            if MONTHLY_FORECAST_FILE.exists():
                st.subheader("Monthly forecast summary")
                st.dataframe(
                    pd.read_csv(MONTHLY_FORECAST_FILE),
                    use_container_width=True,
                )
            st.dataframe(forecast, use_container_width=True)
        else:
            st.warning("The 90-day forecast file has no valid forecast rows.")
    else:
        st.error(
            "Forecast file must contain DATE and FORECAST_BOOKING_COUNT columns."
        )
else:
    st.info(
        "A future forecast will appear after running "
        "`python -m src.run_pipeline`."
    )

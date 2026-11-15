"""Run data preparation, model evaluation when feasible, and forecasting."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_booking_date_counts
from src.evaluation import evaluate_models
from src.feature_engineering import (
    create_time_features,
    get_feature_columns,
    save_feature_data,
)
from src.forecasting import (
    MODEL_DIR,
    forecast_weekday_baseline,
    forecast_with_model,
    save_forecast_outputs,
    train_models,
)
from src.preprocessing import save_processed_data, prepare_daily_demand

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATUS_PATH = PROJECT_DIR / "outputs" / "pipeline_status.json"
MINIMUM_TRAINING_ROWS = 10
SHORT_HISTORY_FEATURES = [
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_sin",
    "day_cos",
]


def select_training_features(featured_df):
    """Use shorter lags when history cannot support the full feature set."""
    complete_rows = featured_df.dropna(
        subset=get_feature_columns() + ["BOOKING_COUNT"]
    )
    if len(complete_rows) >= MINIMUM_TRAINING_ROWS:
        return get_feature_columns()

    short_rows = featured_df.dropna(
        subset=SHORT_HISTORY_FEATURES + ["BOOKING_COUNT"]
    )
    if len(short_rows) >= MINIMUM_TRAINING_ROWS:
        return SHORT_HISTORY_FEATURES.copy()

    return []


def run_pipeline():
    daily_counts = load_booking_date_counts(expected_month="2026-08")
    daily_df = prepare_daily_demand(
        daily_counts,
        date_col="DATE",
        demand_col="BOOKING_COUNT",
        fill_missing_dates=True,
    )
    save_processed_data(daily_df)

    featured_df = create_time_features(daily_df)
    save_feature_data(featured_df)

    feature_columns = select_training_features(featured_df)
    training_rows = featured_df.dropna(
        subset=feature_columns + ["BOOKING_COUNT"]
    ) if feature_columns else featured_df.iloc[0:0].copy()
    ml_evaluated = len(training_rows) >= MINIMUM_TRAINING_ROWS

    if ml_evaluated:
        print(
            f"Evaluating ML models with {len(training_rows)} usable rows "
            f"and {len(feature_columns)} features."
        )
        training = train_models(
            featured_df,
            feature_columns=feature_columns,
            target_col="BOOKING_COUNT",
            date_col="DATE",
        )
        metrics = evaluate_models(training["predictions"])
        best_model_name = str(metrics.iloc[0]["Model"])
        best_model = training["models"][best_model_name]
        best_model.fit(
            training_rows[feature_columns],
            training_rows["BOOKING_COUNT"],
        )
        model_file = (
            MODEL_DIR
            / f"{best_model_name.lower().replace(' ', '_')}.joblib"
        )
        joblib.dump(best_model, model_file)

        forecast_df = forecast_with_model(
            daily_df,
            best_model,
            feature_columns,
            horizon=90,
        )
        model_name = best_model_name
    else:
        print(
            f"Only {len(training_rows)} usable feature rows are available; "
            f"{MINIMUM_TRAINING_ROWS} are required for ML evaluation. "
            "Producing the labeled weekday-average baseline instead."
        )
        forecast_df = forecast_weekday_baseline(daily_df, horizon=90)
        model_name = None

    save_forecast_outputs(forecast_df)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "forecast_method": forecast_df["FORECAST_METHOD"].iloc[0],
        "history_start": daily_df["DATE"].min().strftime("%Y-%m-%d"),
        "history_end": daily_df["DATE"].max().strftime("%Y-%m-%d"),
        "history_days": int(len(daily_df)),
        "ml_evaluated": ml_evaluated,
        "selected_model": model_name,
        "evaluation_rows": int(len(training_rows)),
        "evaluation_test_rows": (
            int(len(training["test_df"])) if ml_evaluated else 0
        ),
        "evaluation_note": (
            "Exploratory only: a short history makes test metrics unstable."
            if ml_evaluated and len(training["test_df"]) < 20
            else None
        ),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\nPipeline completed.")
    print("Forecast method:", status["forecast_method"])
    print("History:", status["history_start"], "to", status["history_end"])
    if model_name:
        print("Selected model:", model_name)
    return {
        "daily": daily_df,
        "features": featured_df,
        "forecast": forecast_df,
        "status": status,
    }


if __name__ == "__main__":
    run_pipeline()

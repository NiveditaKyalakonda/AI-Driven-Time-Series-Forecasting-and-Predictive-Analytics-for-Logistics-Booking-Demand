
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
FORECAST_DIR = PROJECT_DIR / "outputs" / "forecasts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)


def get_models():
    """Return the forecasting models."""
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }


def _validated_daily_history(
    df,
    date_col="DATE",
    target_col="BOOKING_COUNT"
):
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Required columns: {date_col}, {target_col}"
        )

    history = df[[date_col, target_col]].copy()
    history[date_col] = pd.to_datetime(history[date_col], errors="coerce")
    history[target_col] = pd.to_numeric(
        history[target_col], errors="coerce"
    )
    history = (
        history.dropna(subset=[date_col, target_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if history.empty:
        raise ValueError("No valid daily demand observations were found.")
    if history[date_col].duplicated().any():
        raise ValueError("Daily history contains duplicate dates.")
    if (history[target_col] < 0).any():
        raise ValueError("Booking demand cannot be negative.")
    if len(history) > 1:
        expected_dates = pd.date_range(
            history[date_col].iloc[0],
            history[date_col].iloc[-1],
            freq="D"
        )
        if len(expected_dates) != len(history):
            raise ValueError(
                "Daily history has missing dates. Rebuild it with "
                "fill_missing_dates=True before forecasting."
            )

    return history


def forecast_weekday_baseline(
    daily_df,
    horizon=90,
    lookback_days=28,
    date_col="DATE",
    target_col="BOOKING_COUNT"
):
    """Forecast future demand using recent averages for each weekday."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if lookback_days < 7:
        raise ValueError("lookback_days must be at least 7.")

    history = _validated_daily_history(daily_df, date_col, target_col)
    if len(history) < 7:
        raise ValueError(
            "At least seven consecutive days are required for a "
            "weekday baseline."
        )

    recent = history.tail(min(lookback_days, len(history))).copy()
    weekday_means = recent.groupby(
        recent[date_col].dt.dayofweek
    )[target_col].mean()
    future_dates = pd.date_range(
        history[date_col].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    overall_mean = recent[target_col].mean()
    predictions = [
        max(0, int(np.rint(weekday_means.get(date.dayofweek, overall_mean))))
        for date in future_dates
    ]

    return pd.DataFrame({
        "DATE": future_dates,
        "FORECAST_BOOKING_COUNT": predictions,
        "FORECAST_METHOD": "weekday_mean_baseline"
    })


def forecast_with_model(
    daily_df,
    model,
    feature_columns,
    horizon=90,
    date_col="DATE",
    target_col="BOOKING_COUNT"
):
    """Generate recursive forecasts from a trained feature-based model."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    history = _validated_daily_history(daily_df, date_col, target_col)
    from src.feature_engineering import create_time_features

    engineered = create_time_features(history, date_col, target_col)
    missing = [
        column for column in feature_columns
        if column not in engineered.columns
    ]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    predictions = []
    future_dates = pd.date_range(
        history[date_col].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    for future_date in future_dates:
        candidate = pd.concat(
            [
                history,
                pd.DataFrame([{date_col: future_date, target_col: 0}])
            ],
            ignore_index=True
        )
        row = create_time_features(
            candidate, date_col, target_col
        ).iloc[[-1]]
        value = float(model.predict(row[feature_columns])[0])
        if not np.isfinite(value):
            raise ValueError(
                f"Model produced a non-finite forecast for {future_date:%Y-%m-%d}."
            )
        value = max(0, int(np.rint(value)))
        predictions.append({
            "DATE": future_date,
            "FORECAST_BOOKING_COUNT": value,
            "FORECAST_METHOD": "selected_ml_model"
        })
        history = pd.concat(
            [
                history,
                pd.DataFrame([{date_col: future_date, target_col: value}])
            ],
            ignore_index=True
        )

    return pd.DataFrame(predictions)


def create_monthly_forecast_summary(forecast_df):
    required = {"DATE", "FORECAST_BOOKING_COUNT"}
    missing = required.difference(forecast_df.columns)
    if missing:
        raise ValueError(
            f"Forecast data is missing required columns: {sorted(missing)}"
        )

    forecast = forecast_df.copy()
    forecast["DATE"] = pd.to_datetime(forecast["DATE"], errors="coerce")
    forecast["FORECAST_BOOKING_COUNT"] = pd.to_numeric(
        forecast["FORECAST_BOOKING_COUNT"], errors="coerce"
    )
    forecast = forecast.dropna(
        subset=["DATE", "FORECAST_BOOKING_COUNT"]
    )
    if forecast.empty:
        raise ValueError("No valid forecast values are available to summarize.")

    summary = (
        forecast.assign(MONTH=forecast["DATE"].dt.to_period("M").dt.to_timestamp())
        .groupby("MONTH", as_index=False)
        .agg(
            FORECAST_BOOKINGS=("FORECAST_BOOKING_COUNT", "sum"),
            AVERAGE_DAILY_BOOKINGS=("FORECAST_BOOKING_COUNT", "mean"),
            FORECAST_DAYS=("DATE", "size")
        )
    )
    summary["FORECAST_BOOKINGS"] = (
        summary["FORECAST_BOOKINGS"].round().astype(int)
    )
    summary["AVERAGE_DAILY_BOOKINGS"] = (
        summary["AVERAGE_DAILY_BOOKINGS"].round(2)
    )
    return summary


def save_forecast_outputs(forecast_df, forecast_dir=FORECAST_DIR):
    forecast_dir = Path(forecast_dir)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    daily_path = forecast_dir / "daily_90_day_forecast.csv"
    monthly_path = forecast_dir / "monthly_forecast_summary.csv"

    forecast_df.to_csv(daily_path, index=False)
    create_monthly_forecast_summary(forecast_df).to_csv(
        monthly_path, index=False
    )
    print("Daily forecast saved:", daily_path)
    print("Monthly forecast saved:", monthly_path)
    return daily_path, monthly_path


def train_models(
    df,
    feature_columns=None,
    target_col="BOOKING_COUNT",
    date_col="DATE",
    test_size=0.2
):
    """
    Chronologically split the data, train models, and save predictions.

    df must already contain engineered features.
    """
    df = df.copy()

    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Required columns: {date_col}, {target_col}"
        )

    if feature_columns is None:
        from src.feature_engineering import get_feature_columns
        feature_columns = get_feature_columns()

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = (
        df.dropna(subset=[date_col, target_col] + feature_columns)
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} usable rows remain after feature engineering. "
            "Obtain more historical data or review the lag-feature setup."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(df) * (1 - test_size))

    if split_index < 2 or len(df) - split_index < 2:
        raise ValueError(
            "Not enough records for the requested train/test split."
        )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_test = test_df[feature_columns]
    y_test = test_df[target_col]

    prediction_df = pd.DataFrame({
        date_col: test_df[date_col].to_numpy(),
        "Actual": y_test.to_numpy()
    })

    metrics_rows = []
    trained_models = {}

    for name, model in get_models().items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        prediction_df[name] = predicted
        trained_models[name] = model

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    with open(MODEL_DIR / "feature_columns.json", "w") as file:
        json.dump(feature_columns, file, indent=4)

    prediction_path = (
        FORECAST_DIR / "model_training_predictions.csv"
    )
    prediction_df.to_csv(prediction_path, index=False)

    print("\nTraining completed.")
    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Models saved to:", MODEL_DIR)
    print("Predictions saved to:", prediction_path)

    return {
        "models": trained_models,
        "predictions": prediction_df,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df
    }



if __name__ == "__main__":
    from src.feature_engineering import (
        create_time_features,
        get_feature_columns
    )

    input_path = (
        PROJECT_DIR
        / "data"
        / "processed"
        / "daily_booking_demand.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}\n"
            "Run preprocessing.py first."
        )

    daily_df = pd.read_csv(input_path)

    featured_df = create_time_features(daily_df)

    feature_columns = get_feature_columns()

    results = train_models(
        featured_df,
        feature_columns=feature_columns,
        target_col="BOOKING_COUNT",
        date_col="DATE",
        test_size=0.2
    )

    print("\nTraining predictions:")
    print(results["predictions"].head())

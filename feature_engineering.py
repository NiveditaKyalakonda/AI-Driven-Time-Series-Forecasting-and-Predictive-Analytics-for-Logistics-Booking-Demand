
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def get_feature_columns():
    """Return the feature names in the required order."""
    return [
        "year", "month", "day", "day_of_week",
        "week_of_year", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
        "rolling_mean_3", "rolling_mean_7", "rolling_mean_14",
        "rolling_std_7", "demand_change_1", "demand_change_7",
        "day_sin", "day_cos", "month_sin", "month_cos"
    ]


def create_time_features(
    df,
    date_col="DATE",
    target_col="BOOKING_COUNT"
):
    """Create time-series predictors using past demand only."""
    df = df.copy()

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = (
        df.dropna(subset=[date_col, target_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if df[date_col].duplicated().any():
        raise ValueError(
            "Duplicate dates found. Aggregate demand by date before "
            "creating features."
        )

    # Calendar features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    # Lag features: previous observations
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Shift first to avoid using today's actual demand as a predictor.
    previous_demand = df[target_col].shift(1)

    for window in [3, 7, 14]:
        df[f"rolling_mean_{window}"] = (
            previous_demand.rolling(window=window).mean()
        )

    df["rolling_std_7"] = (
        previous_demand.rolling(window=7).std()
    )

    df["demand_change_1"] = (
        df[target_col].shift(1) - df[target_col].shift(2)
    )

    df["demand_change_7"] = (
        df[target_col].shift(1) - df[target_col].shift(8)
    )

    # Cyclical features
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def save_feature_data(
    featured_df,
    file_name="feature_engineered_demand.csv"
):
    """Save the feature-engineered dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / file_name

    featured_df.to_csv(output_path, index=False)
    print("Feature data saved:", output_path)

    return output_path


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "daily_booking_demand.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Run preprocessing first: {input_path}"
        )

    daily_df = pd.read_csv(input_path)
    featured_df = create_time_features(daily_df)

    save_feature_data(featured_df)

    print("Feature dataset shape:", featured_df.shape)
    print(featured_df.head())

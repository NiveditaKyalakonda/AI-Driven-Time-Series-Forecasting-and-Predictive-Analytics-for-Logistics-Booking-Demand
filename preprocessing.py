
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def prepare_daily_demand(
    df,
    date_col=None,
    demand_col=None,
    transaction_data=False,
    fill_missing_dates=False
):
    """
    Prepare daily booking demand.

    transaction_data=False:
        Sum the demand column for each date.

    transaction_data=True:
        Count valid transaction rows for each date.
    """
    if df.empty:
        raise ValueError("Input dataset is empty.")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if date_col is None:
        candidates = ["DATE", "BKDATE", "Date", "date", "booking_date"]
        date_col = next(
            (col for col in candidates if col in df.columns), None
        )

    if date_col is None or date_col not in df.columns:
        raise ValueError(
            f"Date column not found. Available columns: {df.columns.tolist()}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        raise ValueError("No valid dates remain after cleaning.")

    if transaction_data:
        # Use only when each row represents one booking.
        daily = (
            df.groupby(df[date_col].dt.normalize())
            .size()
            .rename("BOOKING_COUNT")
            .reset_index()
        )
        daily.columns = ["DATE", "BOOKING_COUNT"]

    else:
        if demand_col is None:
            candidates = [
                "BOOKING_COUNT", "booking_count",
                "daily_booking_demand", "demand", "Demand", "bookings"
            ]
            demand_col = next(
                (col for col in candidates if col in df.columns), None
            )

        if demand_col is None or demand_col not in df.columns:
            raise ValueError(
                "Demand column not found. If each row represents a booking, "
                "call prepare_daily_demand(df, date_col='BKDATE', "
                "transaction_data=True)."
            )

        df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
        df = df.dropna(subset=[demand_col])

        daily = (
            df.groupby(df[date_col].dt.normalize())[demand_col]
            .sum()
            .rename("BOOKING_COUNT")
            .reset_index()
        )
        daily.columns = ["DATE", "BOOKING_COUNT"]

    daily["DATE"] = pd.to_datetime(daily["DATE"])
    daily = daily.sort_values("DATE").reset_index(drop=True)

    if fill_missing_dates:
        # Only enable when a missing date genuinely means zero bookings.
        daily = daily.set_index("DATE").asfreq("D")
        daily["BOOKING_COUNT"] = daily["BOOKING_COUNT"].fillna(0)
        daily = daily.reset_index()

    daily["BOOKING_COUNT"] = pd.to_numeric(
        daily["BOOKING_COUNT"], errors="raise"
    )

    if (daily["BOOKING_COUNT"] < 0).any():
        raise ValueError("Booking demand cannot be negative.")

    print("Daily demand prepared.")
    print("Rows:", len(daily))
    print("Date range:", daily["DATE"].min(), "to", daily["DATE"].max())

    return daily


def save_processed_data(
    daily_df,
    file_name="daily_booking_demand.csv"
):
    """Save cleaned data inside data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / file_name

    daily_df.to_csv(output_path, index=False)
    print("Saved:", output_path)

    return output_path


if __name__ == "__main__":
    from src.data_loader import load_booking_date_counts

    daily_counts = load_booking_date_counts(expected_month="2026-08")
    daily_df = prepare_daily_demand(
        daily_counts,
        date_col="DATE",
        demand_col="BOOKING_COUNT",
        fill_missing_dates=True
    )
    save_processed_data(daily_df)

    print(daily_df.head())

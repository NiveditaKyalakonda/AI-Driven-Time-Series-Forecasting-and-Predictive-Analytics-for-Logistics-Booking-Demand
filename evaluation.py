
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PREDICTION_FILE = (
    PROJECT_DIR / "outputs" / "forecasts" / "model_training_predictions.csv"
)
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid actual and predicted values found.")

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # R² requires at least two observations.
    r2 = r2_score(actual, predicted) if len(actual) >= 2 else np.nan

    # MAPE is calculated only where actual values are nonzero.
    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                       / actual[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_Percent": mape,
    }


def evaluate_models(prediction_df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if "Actual" not in prediction_df.columns:
        raise ValueError(
            "The predictions file must contain an 'Actual' column. "
            f"Found columns: {prediction_df.columns.tolist()}"
        )

    # Identify prediction columns, excluding date and actual values.
    prediction_columns = [
        col for col in prediction_df.columns
        if col not in ["DATE", "Actual"]
        and pd.api.types.is_numeric_dtype(prediction_df[col])
    ]

    if not prediction_columns:
        raise ValueError(
            "No model prediction columns found in the predictions file."
        )

    results = []

    for model_name in prediction_columns:
        metrics = calculate_metrics(
            prediction_df["Actual"],
            prediction_df[model_name],
        )
        metrics["Model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "MAE", "RMSE", "R2", "MAPE_Percent"]
    ]

    results_df = results_df.sort_values("MAE", ascending=True)

    metrics_path = REPORT_DIR / "evaluation_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    best_model = results_df.iloc[0]

    report_path = REPORT_DIR / "evaluation_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("LOGISTICS BOOKING DEMAND - MODEL EVALUATION\n")
        file.write("=" * 48 + "\n\n")
        file.write(f"Predictions evaluated: {len(prediction_df)}\n\n")
        file.write("MODEL COMPARISON\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\nBEST MODEL BY LOWEST MAE\n")
        file.write(f"Model: {best_model['Model']}\n")
        file.write(f"MAE: {best_model['MAE']:.4f}\n")
        file.write(f"RMSE: {best_model['RMSE']:.4f}\n")
        file.write(f"R2: {best_model['R2']:.4f}\n")
        file.write(f"MAPE: {best_model['MAPE_Percent']:.2f}%\n")

    # Save the selected model name.
    (PROJECT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "models" / "best_model_name.txt").write_text(
        str(best_model["Model"]), encoding="utf-8"
    )

    print("\nMODEL EVALUATION RESULTS")
    print(results_df.to_string(index=False))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    print(f"Best model by MAE: {best_model['Model']}")

    return results_df


if __name__ == "__main__":
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}\n"
            "Train your models successfully before evaluating them."
        )

    predictions = pd.read_csv(PREDICTION_FILE)
    evaluate_models(predictions)

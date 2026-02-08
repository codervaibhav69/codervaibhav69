"""Train a linear regression model to predict house prices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load house price data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    data = pd.read_csv(csv_path)
    required_columns = {"square_feet", "bedrooms", "bathrooms", "age", "price"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def train_model(data: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series]:
    """Split data, train linear regression model, and return evaluation pieces."""
    features = data[["square_feet", "bedrooms", "bathrooms", "age"]]
    target = data["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model, x_test, y_test, model.predict(x_test)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="House price prediction with linear regression")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/house_prices.csv"),
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()

    data = load_data(args.data)
    model, x_test, y_test, predictions = train_model(data)
    metrics = evaluate(y_test, predictions)

    print("Model coefficients:")
    for name, coef in zip(x_test.columns, model.coef_, strict=True):
        print(f"  {name}: {coef:,.2f}")
    print(f"Intercept: {model.intercept_:,.2f}\n")

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:,.2f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "sales_forecast.csv"
FEATURE_COLUMNS = (
    "advertising_spend",
    "product_price",
    "discount_percent",
    "store_traffic",
    "previous_sales",
    "weekend",
    "holiday_season",
)
TARGET_COLUMN = "sales_amount"
FEATURE_LIMITS = {
    "advertising_spend": (50, 1_200),
    "product_price": (15, 180),
    "discount_percent": (0, 35),
    "store_traffic": (40, 1_000),
    "previous_sales": (500, 12_000),
    "weekend": (0, 1),
    "holiday_season": (0, 1),
}


def generate_sales_dataset(
    path: str | Path = DATASET_PATH,
    row_count: int = 600,
    random_state: int = 42,
) -> Path:
    """Create the reproducible, synthetic sales dataset used by this model."""
    if row_count < 20:
        raise ValueError("The sales dataset must contain at least 20 rows.")

    rng = np.random.default_rng(random_state)
    dataset = pd.DataFrame(
        {
            "advertising_spend": rng.uniform(50, 1_200, row_count).round(2),
            "product_price": rng.uniform(15, 180, row_count).round(2),
            "discount_percent": rng.uniform(0, 35, row_count).round(2),
            "store_traffic": rng.integers(40, 1_001, row_count),
            "previous_sales": rng.uniform(500, 12_000, row_count).round(2),
            "weekend": rng.integers(0, 2, row_count),
            "holiday_season": rng.integers(0, 2, row_count),
        }
    )
    noise = rng.normal(0, 220, row_count)
    dataset[TARGET_COLUMN] = (
        400
        + 2.2 * dataset["advertising_spend"]
        - 18 * dataset["product_price"]
        + 32 * dataset["discount_percent"]
        + 1.8 * dataset["store_traffic"]
        + 0.24 * dataset["previous_sales"]
        + 170 * dataset["weekend"]
        + 260 * dataset["holiday_season"]
        + noise
    ).round(2)
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].clip(lower=1)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


def load_sales_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the expected sales data without silently repairing it."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Sales dataset not found at {dataset_path}. "
            "Generate it with generate_sales_dataset() before starting the app."
        )

    try:
        dataset = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read the sales dataset at {dataset_path}.") from error

    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [name for name in expected_columns if name not in dataset.columns]
    extra_columns = [name for name in dataset.columns if name not in expected_columns]
    if missing_columns or extra_columns:
        raise ValueError(
            f"Sales dataset columns do not match the expected schema. "
            f"Missing: {missing_columns}; unexpected: {extra_columns}."
        )
    dataset = dataset[expected_columns]
    if len(dataset) < 20:
        raise ValueError("Sales dataset must contain at least 20 records.")
    if dataset.isna().any().any():
        raise ValueError("Sales dataset contains missing values.")
    if dataset.duplicated().any():
        raise ValueError("Sales dataset contains duplicate records.")

    for column in expected_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    values = dataset[expected_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sales dataset contains non-numeric or non-finite values.")

    for column, (minimum, maximum) in FEATURE_LIMITS.items():
        if not dataset[column].between(minimum, maximum).all():
            raise ValueError(f"Sales dataset column '{column}' contains out-of-range values.")
    for column in ("weekend", "holiday_season"):
        if not dataset[column].isin((0, 1)).all():
            raise ValueError(f"Sales dataset column '{column}' must contain only 0 or 1.")
    if (dataset[TARGET_COLUMN] <= 0).any():
        raise ValueError("Sales target must contain positive amounts.")

    return dataset


def train_sales_model(
    dataset_path: str | Path = DATASET_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train linear regression and report metrics from a held-out test split."""
    dataset = load_sales_dataset(dataset_path)
    features = dataset.loc[:, list(FEATURE_COLUMNS)]
    target = dataset[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, predictions)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "dataset_size": int(len(dataset)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "test_actual": y_test.tolist(),
        "test_predicted": predictions.tolist(),
        "dataset_preview": dataset.head(5).to_dict(orient="records"),
    }


def predict_sales(model: LinearRegression, features: dict[str, Any]) -> float:
    """Validate one retail scenario and return its predicted sales amount."""
    if not isinstance(features, dict):
        raise ValueError("Prediction input must be a JSON object of numeric features.")

    missing = [column for column in FEATURE_COLUMNS if column not in features]
    unexpected = [column for column in features if column not in FEATURE_COLUMNS]
    if missing or unexpected:
        raise ValueError(
            f"Prediction features do not match the model. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    validated: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = features[column]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"'{column}' must be a finite number.")
        try:
            numeric_value = float(value)
        except OverflowError as error:
            raise ValueError(f"'{column}' must be a finite number.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"'{column}' must be a finite number.")
        minimum, maximum = FEATURE_LIMITS[column]
        if not minimum <= numeric_value <= maximum:
            raise ValueError(f"'{column}' must be between {minimum} and {maximum}.")
        if column in ("weekend", "holiday_season") and numeric_value not in (0, 1):
            raise ValueError(f"'{column}' must be 0 or 1.")
        validated[column] = numeric_value

    input_frame = pd.DataFrame([validated], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(input_frame)[0])
    if prediction < 0:
        raise ValueError(
            "The model estimated a negative sales amount for this combination. "
            "Adjust the inputs; this linear model can extrapolate beyond sensible "
            "business values."
        )
    return prediction

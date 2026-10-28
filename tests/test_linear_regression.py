import math

import numpy as np
import pandas as pd
import pytest

from app import create_app
from ml.linear_regression import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_sales_dataset,
    load_sales_dataset,
    predict_sales,
    train_sales_model,
)


def test_dataset_has_expected_rows_columns_and_valid_values():
    dataset = load_sales_dataset()

    assert len(dataset) >= 100
    assert list(dataset.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert dataset.isna().sum().sum() == 0
    assert dataset.duplicated().sum() == 0


def test_training_uses_a_held_out_test_set_and_real_regression_metrics():
    result = train_sales_model()
    errors = np.asarray(result["test_actual"]) - np.asarray(result["test_predicted"])

    assert result["train_size"] + result["test_size"] == result["dataset_size"]
    assert result["test_size"] > 0
    assert result["train_size"] < result["dataset_size"]
    assert set(result["metrics"]) == {"mae", "rmse", "r2"}
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert result["metrics"]["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert result["metrics"]["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["rmse"] >= result["metrics"]["mae"]


def test_dataset_generation_is_reproducible(tmp_path):
    first_path = generate_sales_dataset(tmp_path / "first.csv", random_state=17)
    second_path = generate_sales_dataset(tmp_path / "second.csv", random_state=17)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_missing_dataset_reports_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Generate it"):
        load_sales_dataset(tmp_path / "missing.csv")


def test_dataset_loader_rejects_missing_values_and_wrong_schema(tmp_path):
    source = load_sales_dataset()
    missing_value_path = tmp_path / "missing-value.csv"
    source_with_missing = source.copy()
    source_with_missing.loc[0, "store_traffic"] = np.nan
    source_with_missing.to_csv(missing_value_path, index=False)
    wrong_schema_path = tmp_path / "wrong-schema.csv"
    source.rename(columns={"sales_amount": "sales"}).to_csv(
        wrong_schema_path,
        index=False,
    )

    with pytest.raises(ValueError, match="missing values"):
        load_sales_dataset(missing_value_path)
    with pytest.raises(ValueError, match="columns do not match"):
        load_sales_dataset(wrong_schema_path)


def test_prediction_returns_finite_sales_amount():
    result = train_sales_model()
    prediction = predict_sales(result["model"], valid_features())

    assert math.isfinite(prediction)
    assert prediction > 0


def test_prediction_rejects_missing_out_of_range_and_nonfinite_values():
    result = train_sales_model()
    incomplete = valid_features()
    incomplete.pop("product_price")
    too_large = valid_features()
    too_large["store_traffic"] = 100_000
    nonfinite = valid_features()
    nonfinite["advertising_spend"] = float("nan")

    for features in (incomplete, too_large, nonfinite):
        try:
            predict_sales(result["model"], features)
        except ValueError:
            continue
        raise AssertionError("Invalid prediction features should raise ValueError.")


def test_prediction_rejects_negative_sales_estimates():
    result = train_sales_model()
    low_sales_scenario = {
        "advertising_spend": 50,
        "product_price": 180,
        "discount_percent": 0,
        "store_traffic": 40,
        "previous_sales": 500,
        "weekend": 0,
        "holiday_season": 0,
    }

    with pytest.raises(ValueError, match="negative sales amount"):
        predict_sales(result["model"], low_sales_scenario)


def test_prediction_endpoint_returns_prediction_and_rejects_invalid_input():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/api/predict/linear-regression",
        json=valid_features(),
    )
    invalid_response = client.post(
        "/api/predict/linear-regression",
        json={"product_price": -1},
    )
    negative_forecast_response = client.post(
        "/api/predict/linear-regression",
        json={
            "advertising_spend": 50,
            "product_price": 180,
            "discount_percent": 0,
            "store_traffic": 40,
            "previous_sales": 500,
            "weekend": 0,
            "holiday_season": 0,
        },
    )

    assert response.status_code == 200
    assert response.json["prediction"]["sales_amount"] > 0
    assert invalid_response.status_code == 400
    assert "error" in invalid_response.json
    assert negative_forecast_response.status_code == 400
    assert "negative sales amount" in negative_forecast_response.json["error"]


def test_prediction_endpoint_rejects_non_json_and_unexpected_features():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    non_json_response = client.post(
        "/api/predict/linear-regression",
        data="not json",
        content_type="text/plain",
    )
    unexpected_feature_response = client.post(
        "/api/predict/linear-regression",
        json={**valid_features(), "customer_name": "not accepted"},
    )

    assert non_json_response.status_code == 400
    assert "JSON object" in non_json_response.json["error"]
    assert unexpected_feature_response.status_code == 400
    assert "unexpected" in unexpected_feature_response.json["error"]


def test_model_page_displays_real_evaluation_values_and_dataset_notice():
    app = create_app()
    app.config.update(TESTING=True)
    response = app.test_client().get("/models/linear-regression")

    assert response.status_code == 200
    assert b"synthetic" in response.data.lower()
    assert b"MAE" in response.data
    assert b"Actual vs. predicted" in response.data
    assert b"Why this data?" in response.data
    assert b"Accepted range: 50" in response.data
    assert b"average in absolute terms" in response.data


def test_company_pages_and_health_report_all_trained_models():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    home = client.get("/")
    about = client.get("/about")
    health = client.get("/health")

    assert home.status_code == 200
    assert home.data.count(b'class="model-card h-100"') == 6
    assert home.data.count(b"status-ready") == 6
    assert b"Our operational objectives" in home.data
    assert about.status_code == 200
    assert b"fictional educational company" in about.data
    assert health.status_code == 200
    assert set(health.json["ready_models"]) == {
        "linear-regression",
        "naive-bayes",
        "knn",
        "svm",
        "decision-tree",
        "neural-network",
    }


def valid_features():
    return {
        "advertising_spend": 450,
        "product_price": 65,
        "discount_percent": 10,
        "store_traffic": 340,
        "previous_sales": 4200,
        "weekend": 1,
        "holiday_season": 0,
    }

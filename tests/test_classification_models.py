import math

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import app as app_module
import ml.datasets as datasets_module
from ml.classification_models import MODEL_CONFIGS, predict_classification, train_classification_model
from ml.datasets import (
    DATASET_COLUMNS,
    DATASET_TARGET_LABELS,
    generate_ann_purchase_dataset,
    generate_churn_dataset,
    generate_customer_similarity_dataset,
    generate_review_dataset,
    generate_svm_purchase_dataset,
    load_classification_dataset,
)


MODEL_SLUGS = tuple(MODEL_CONFIGS)
VALID_INPUTS = {
    "naive-bayes": {"review_text": "The backpack is excellent quality and was a great value."},
    "knn": {
        "age": 34,
        "purchase_frequency": 22,
        "average_spend": 180.5,
        "website_visits": 40,
        "prior_purchases": 20,
    },
    "svm": {
        "product_page_visits": 12,
        "browsing_minutes": 15.5,
        "cart_items": 2,
        "previous_purchase": 1,
        "discount_exposure": 1,
    },
    "decision-tree": {
        "days_since_purchase": 120,
        "purchases_per_year": 8,
        "average_spend": 95,
        "service_complaints": 2,
        "cancelled_orders": 1,
    },
    "neural-network": {
        "product_page_visits": 12,
        "browsing_minutes": 15.5,
        "cart_items": 2,
        "previous_purchase": 1,
        "discount_exposure": 1,
        "email_clicks": 2,
    },
}


@pytest.fixture(scope="module")
def client():
    app = app_module.create_app()
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.mark.parametrize("slug", MODEL_SLUGS)
def test_generated_dataset_matches_schema_and_has_valid_records(slug):
    config = MODEL_CONFIGS[slug]
    dataset = load_classification_dataset(config["dataset"])

    assert tuple(dataset.columns) == DATASET_COLUMNS[config["dataset"]]
    assert len(dataset) >= 40
    assert dataset.isna().sum().sum() == 0
    assert set(dataset[config["target"]].astype(str)) == DATASET_TARGET_LABELS[config["dataset"]]
    assert dataset[config["target"]].value_counts().min() >= 5
    assert dataset.duplicated().sum() == 0


@pytest.mark.parametrize(
    ("generator", "filename"),
    [
        (generate_review_dataset, "reviews.csv"),
        (generate_customer_similarity_dataset, "customers.csv"),
        (generate_svm_purchase_dataset, "svm.csv"),
        (generate_churn_dataset, "churn.csv"),
        (generate_ann_purchase_dataset, "ann.csv"),
    ],
)
def test_synthetic_dataset_generation_is_reproducible(generator, filename, tmp_path):
    first = generator(tmp_path / f"first-{filename}", random_state=91)
    second = generator(tmp_path / f"second-{filename}", random_state=91)

    assert first.read_bytes() == second.read_bytes()


def test_dataset_loader_rejects_unexpected_target_labels(monkeypatch, tmp_path):
    source = load_classification_dataset("customer_similarity.csv").copy()
    source.loc[0, "purchase_category"] = "unknown"
    source.to_csv(tmp_path / "customer_similarity.csv", index=False)
    monkeypatch.setattr(datasets_module, "DATA_DIRECTORY", tmp_path)

    with pytest.raises(ValueError, match="unexpected target labels"):
        load_classification_dataset("customer_similarity.csv")


@pytest.mark.parametrize("slug", MODEL_SLUGS)
def test_model_training_and_metrics_use_held_out_predictions(slug):
    result = train_classification_model(slug)
    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    assert result["train_size"] + result["test_size"] == result["dataset_size"]
    assert result["test_size"] > 0
    assert set(result["metrics"]) == expected_metrics
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in result["metrics"].values())
    assert result["metrics"]["accuracy"] == pytest.approx(
        accuracy_score(result["test_actual"], result["test_predicted"])
    )
    expected_baseline = accuracy_score(
        result["test_actual"],
        np.repeat(result["majority_baseline_label"], result["test_size"]),
    )
    assert result["majority_baseline_accuracy"] == pytest.approx(expected_baseline)
    assert result["metrics"]["precision"] == pytest.approx(
        precision_score(
            result["test_actual"],
            result["test_predicted"],
            labels=result["labels"],
            average="weighted",
            zero_division=0,
        )
    )
    assert result["metrics"]["recall"] == pytest.approx(
        recall_score(
            result["test_actual"],
            result["test_predicted"],
            labels=result["labels"],
            average="weighted",
            zero_division=0,
        )
    )
    assert result["metrics"]["f1"] == pytest.approx(
        f1_score(
            result["test_actual"],
            result["test_predicted"],
            labels=result["labels"],
            average="weighted",
            zero_division=0,
        )
    )
    assert result["confusion_matrix"] == confusion_matrix(
        result["test_actual"],
        result["test_predicted"],
        labels=result["labels"],
    ).tolist()


@pytest.mark.parametrize("slug", MODEL_SLUGS)
def test_each_model_produces_a_valid_prediction(slug):
    result = train_classification_model(slug)
    prediction = predict_classification(result, VALID_INPUTS[slug])

    assert prediction["label"] in result["labels"]
    if "probabilities" in prediction:
        assert set(prediction["probabilities"]) == set(result["labels"])
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    if slug == "knn":
        assert len(prediction["similar_profiles"]) == MODEL_CONFIGS[slug]["neighbors"]
        assert all(profile["distance"] >= 0 for profile in prediction["similar_profiles"])
    if slug == "decision-tree":
        assert "class:" in result["decision_rules"]
    if slug == "neural-network":
        history = result["training_history"]
        assert len(history["loss"]) > 0
        assert len(history["loss"]) == len(history["validation_accuracy"])


@pytest.mark.parametrize("slug", MODEL_SLUGS)
def test_prediction_api_and_invalid_input_handling(slug, client):
    response = client.post(f"/api/predict/{slug}", json=VALID_INPUTS[slug])
    missing_response = client.post(f"/api/predict/{slug}", json={})

    assert response.status_code == 200
    assert response.json["prediction"]["label"]
    assert response.json["target"] == MODEL_CONFIGS[slug]["target"]
    assert response.json["interpretation"]
    assert missing_response.status_code == 400
    assert "error" in missing_response.json


@pytest.mark.parametrize("slug", MODEL_SLUGS)
def test_all_classification_pages_and_navigation_render(slug, client):
    response = client.get(f"/models/{slug}")
    home = client.get("/")
    page_source = response.data.decode("utf-8")

    assert response.status_code == 200
    assert MODEL_CONFIGS[slug]["name"] in page_source
    assert "HELD-OUT TEST RESULTS" in page_source
    assert "Confusion matrix:" in page_source
    assert "majority baseline" in page_source
    assert f"/api/predict/{slug}" in page_source or f'data-model-slug="{slug}"' in page_source
    assert f'href="/models/{slug}"' in home.text
    assert "Synthetic" in page_source


def test_review_and_numeric_prediction_validation(client):
    long_review = client.post(
        "/api/predict/naive-bayes",
        json={"review_text": "x" * 5_001},
    )
    invalid_range = client.post(
        "/api/predict/knn",
        json={**VALID_INPUTS["knn"], "age": 2_000},
    )
    invalid_enum = client.post(
        "/api/predict/svm",
        json={**VALID_INPUTS["svm"], "previous_purchase": 2},
    )

    assert long_review.status_code == 400
    assert invalid_range.status_code == 400
    assert invalid_enum.status_code == 400


def test_comparison_route_reports_all_models_and_metric_warning(client):
    response = client.get("/comparison")
    page = response.data.decode("utf-8")

    assert response.status_code == 200
    for slug in ("linear-regression", *MODEL_SLUGS):
        assert f"/models/{slug}" in page
    assert "must not be ranked across unrelated tasks or datasets" in page
    assert "RMSE" in page
    assert "ACCURACY" in page


def test_initialization_failure_is_visible_without_fake_predictions(monkeypatch):
    def fail_classifier(slug):
        raise RuntimeError(f"simulated {slug} training failure")

    monkeypatch.setattr(app_module, "train_classification_model", fail_classifier)
    app = app_module.create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    health = client.get("/health")
    model_page = client.get("/models/naive-bayes")
    prediction = client.post(
        "/api/predict/naive-bayes",
        json=VALID_INPUTS["naive-bayes"],
    )

    assert health.json["status"] == "degraded"
    assert "naive-bayes" in health.json["unavailable_models"]
    assert model_page.status_code == 503
    assert b"No prediction or evaluation result is being shown" in model_page.data
    assert prediction.status_code == 503
    assert b"unavailable" in prediction.data

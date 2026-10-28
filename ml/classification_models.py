from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from ml.datasets import load_classification_dataset


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "naive-bayes": {
        "slug": "naive-bayes",
        "name": "Naive Bayes",
        "dataset": "customer_reviews.csv",
        "target": "sentiment",
        "features": ("review_text",),
        "labels": ("negative", "positive"),
        "business_problem": "Classify synthetic customer reviews as positive or negative to help retailers spot service themes.",
        "why_model": "TF-IDF converts review words into weighted numeric features; Multinomial Naive Bayes is a fast, interpretable baseline for word-count-like text features.",
        "data_source": "Synthetic English review sentences generated with a fixed seed from varied product and sentiment phrases; no real customer data.",
        "feature_descriptions": {"review_text": "The customer's written product or shopping review."},
        "fields": [
            {"name": "review_text", "label": "Customer review", "type": "textarea", "required": True}
        ],
        "target_description": "Review sentiment: positive or negative.",
    },
    "knn": {
        "slug": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "dataset": "customer_similarity.csv",
        "target": "purchase_category",
        "features": ("age", "purchase_frequency", "average_spend", "website_visits", "prior_purchases"),
        "labels": ("occasional", "regular", "frequent"),
        "business_problem": "Classify a customer into an activity category using similar historical customer profiles.",
        "why_model": "KNN predicts from the labels of nearby examples. Scaling is essential because otherwise large-unit features would dominate distance.",
        "data_source": "Synthetic customer activity records generated with a fixed seed; categories are derived from a noisy activity score.",
        "feature_descriptions": {
            "age": "Customer age in years (synthetic profile attribute).",
            "purchase_frequency": "Purchases during the recent year.",
            "average_spend": "Average purchase amount in USD.",
            "website_visits": "Retail website visits during the recent period.",
            "prior_purchases": "Count of previous completed purchases.",
        },
        "fields": [
            {"name": "age", "label": "Age (years)", "type": "number", "min": 18, "max": 75, "step": 1},
            {"name": "purchase_frequency", "label": "Purchases in recent year", "type": "number", "min": 1, "max": 44, "step": 1},
            {"name": "average_spend", "label": "Average spend (USD)", "type": "number", "min": 10, "max": 500, "step": "any"},
            {"name": "website_visits", "label": "Website visits", "type": "number", "min": 1, "max": 79, "step": 1},
            {"name": "prior_purchases", "label": "Prior purchases", "type": "number", "min": 0, "max": 59, "step": 1},
        ],
        "target_description": "Purchase activity category: occasional, regular, or frequent.",
        "neighbors": 5,
    },
    "svm": {
        "slug": "svm",
        "name": "Support Vector Machine (SVM)",
        "dataset": "purchase_intent_svm.csv",
        "target": "purchase_intent",
        "features": ("product_page_visits", "browsing_minutes", "cart_items", "previous_purchase", "discount_exposure"),
        "labels": ("no_purchase", "purchase"),
        "business_problem": "Estimate purchase intent from a customer's browsing and shopping activity.",
        "why_model": "An RBF-kernel Support Vector Classifier can separate intent classes using nonlinear boundaries; standardization keeps feature scales comparable.",
        "data_source": "Synthetic browsing-session records generated with a fixed seed and a noisy probabilistic label rule.",
        "feature_descriptions": {
            "product_page_visits": "Product pages visited in the session.",
            "browsing_minutes": "Minutes spent browsing.",
            "cart_items": "Items added to the shopping cart.",
            "previous_purchase": "Whether the customer purchased before (0=no, 1=yes).",
            "discount_exposure": "Whether a discount was shown (0=no, 1=yes).",
        },
        "fields": [
            {"name": "product_page_visits", "label": "Product page visits", "type": "number", "min": 1, "max": 29, "step": 1},
            {"name": "browsing_minutes", "label": "Browsing time (minutes)", "type": "number", "min": 0.5, "max": 45, "step": "any"},
            {"name": "cart_items", "label": "Items in cart", "type": "number", "min": 0, "max": 8, "step": 1},
            {"name": "previous_purchase", "label": "Previous purchase", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
            {"name": "discount_exposure", "label": "Saw a discount", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
        ],
        "target_description": "Purchase intent: purchase or no_purchase.",
    },
    "decision-tree": {
        "slug": "decision-tree",
        "name": "Decision Tree",
        "dataset": "customer_churn.csv",
        "target": "churn_risk",
        "features": ("days_since_purchase", "purchases_per_year", "average_spend", "service_complaints", "cancelled_orders"),
        "labels": ("active", "churned"),
        "business_problem": "Flag customer profiles associated with churn so a retailer can consider a retention check-in.",
        "why_model": "A shallow Decision Tree produces human-readable branching rules for a classification task. The rules describe associations, not causes.",
        "data_source": "Synthetic customer activity records generated with a fixed seed and a noisy churn-label rule.",
        "feature_descriptions": {
            "days_since_purchase": "Days since the most recent purchase.",
            "purchases_per_year": "Completed purchases in the past year.",
            "average_spend": "Average purchase amount in USD.",
            "service_complaints": "Number of service complaints.",
            "cancelled_orders": "Number of cancelled orders.",
        },
        "fields": [
            {"name": "days_since_purchase", "label": "Days since last purchase", "type": "number", "min": 1, "max": 365, "step": 1},
            {"name": "purchases_per_year", "label": "Purchases in past year", "type": "number", "min": 1, "max": 39, "step": 1},
            {"name": "average_spend", "label": "Average spend (USD)", "type": "number", "min": 10, "max": 500, "step": "any"},
            {"name": "service_complaints", "label": "Service complaints", "type": "number", "min": 0, "max": 7, "step": 1},
            {"name": "cancelled_orders", "label": "Cancelled orders", "type": "number", "min": 0, "max": 6, "step": 1},
        ],
        "target_description": "Observed synthetic churn status: active or churned.",
        "max_depth": 4,
    },
    "neural-network": {
        "slug": "neural-network",
        "name": "Artificial Neural Network (ANN)",
        "dataset": "purchase_intent_ann.csv",
        "target": "purchase",
        "features": ("product_page_visits", "browsing_minutes", "cart_items", "previous_purchase", "discount_exposure", "email_clicks"),
        "labels": ("no_purchase", "purchase"),
        "business_problem": "Demonstrate whether behavioral signals can help classify purchase likelihood.",
        "why_model": "A feedforward MLP learns layered nonlinear combinations of input features. Standardized inputs support stable gradient-based training.",
        "data_source": "Synthetic browsing-session records generated with a fixed seed and a noisy probabilistic purchase rule.",
        "feature_descriptions": {
            "product_page_visits": "Product pages visited in the session.",
            "browsing_minutes": "Minutes spent browsing.",
            "cart_items": "Items added to the shopping cart.",
            "previous_purchase": "Whether the customer purchased before (0=no, 1=yes).",
            "discount_exposure": "Whether a discount was shown (0=no, 1=yes).",
            "email_clicks": "Retail marketing email links clicked in the recent period.",
        },
        "fields": [
            {"name": "product_page_visits", "label": "Product page visits", "type": "number", "min": 1, "max": 34, "step": 1},
            {"name": "browsing_minutes", "label": "Browsing time (minutes)", "type": "number", "min": 0.5, "max": 60, "step": "any"},
            {"name": "cart_items", "label": "Items in cart", "type": "number", "min": 0, "max": 9, "step": 1},
            {"name": "previous_purchase", "label": "Previous purchase", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
            {"name": "discount_exposure", "label": "Saw a discount", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
            {"name": "email_clicks", "label": "Recent email clicks", "type": "number", "min": 0, "max": 11, "step": 1},
        ],
        "target_description": "Whether the synthetic browsing session resulted in a purchase.",
        "hidden_layers": (16, 8),
    },
}


def train_classification_model(slug: str, random_state: int = 42) -> dict[str, Any]:
    if slug not in MODEL_CONFIGS:
        raise ValueError(f"Unknown classification model: {slug}")
    config = MODEL_CONFIGS[slug]
    dataset = load_classification_dataset(config["dataset"])
    target_column = config["target"]
    feature_columns = config["features"]
    features = dataset.loc[:, list(feature_columns)]
    target = dataset[target_column]
    labels = list(config["labels"])
    counts = target.value_counts()
    if any(int(counts.get(label, 0)) < 5 for label in labels):
        raise ValueError(f"Dataset {config['dataset']} needs at least five examples per target class.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    model = _create_pipeline(slug, random_state)
    if slug == "naive-bayes":
        model.fit(x_train["review_text"], y_train)
        predictions = model.predict(x_test["review_text"])
    else:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)),
    }
    majority_label = y_train.value_counts().idxmax()
    majority_baseline = float(accuracy_score(y_test, np.repeat(majority_label, len(y_test))))

    result: dict[str, Any] = {
        "model": model,
        "slug": slug,
        "config": config,
        "metrics": metrics,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "dataset_size": int(len(dataset)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "majority_baseline_label": str(majority_label),
        "majority_baseline_accuracy": majority_baseline,
        "feature_types": {
            column: ("Text" if slug == "naive-bayes" else str(dataset[column].dtype))
            for column in feature_columns
        },
        "missing_values": int(dataset.isna().sum().sum()),
        "duplicate_records": int(dataset.duplicated().sum()),
        "class_distribution": {
            str(label): int(count) for label, count in target.value_counts().sort_index().items()
        },
        "dataset_preview": dataset.head(5).to_dict(orient="records"),
        "test_actual": y_test.astype(str).tolist(),
        "test_predicted": predictions.astype(str).tolist(),
        "training_examples": x_train.to_dict(orient="records"),
        "training_targets": y_train.astype(str).tolist(),
    }
    if slug == "decision-tree":
        result["decision_rules"] = export_text(
            model.named_steps["classifier"],
            feature_names=list(feature_columns),
            max_depth=config["max_depth"],
        )
    if slug == "neural-network":
        classifier = model.named_steps["classifier"]
        result["training_history"] = {
            "loss": [float(value) for value in classifier.loss_curve_],
            "validation_accuracy": [
                float(value) for value in getattr(classifier, "validation_scores_", [])
            ],
        }
    return result


def predict_classification(
    result: dict[str, Any],
    submitted: dict[str, Any],
) -> dict[str, Any]:
    config = result["config"]
    feature_columns = config["features"]
    if not isinstance(submitted, dict):
        raise ValueError("Prediction input must be a JSON object.")
    missing = [field for field in feature_columns if field not in submitted]
    unexpected = [field for field in submitted if field not in feature_columns]
    if missing or unexpected:
        raise ValueError(f"Input fields do not match this model. Missing: {missing}; unexpected: {unexpected}.")

    if config["dataset"] == "customer_reviews.csv":
        review = submitted["review_text"]
        if not isinstance(review, str) or not review.strip() or len(review) > 5_000:
            raise ValueError("Review text must contain 1 to 5,000 characters.")
        values: dict[str, Any] = {"review_text": review.strip()}
    else:
        values = {}
        field_specs = {field["name"]: field for field in config["fields"]}
        for field in feature_columns:
            value = submitted[field]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"'{field}' must be a numeric value.")
            try:
                number = float(value)
            except OverflowError as error:
                raise ValueError(f"'{field}' must be a finite number.") from error
            if not np.isfinite(number):
                raise ValueError(f"'{field}' must be a finite number.")
            spec = field_specs[field]
            if spec["type"] == "select":
                allowed_values = [float(value) for value, _ in spec["options"]]
                if number not in allowed_values:
                    raise ValueError(f"'{field}' must be one of {allowed_values}.")
                values[field] = int(number)
                continue
            if not spec["min"] <= number <= spec["max"]:
                raise ValueError(f"'{field}' must be between {spec['min']} and {spec['max']}.")
            if spec["step"] == 1 and not number.is_integer():
                raise ValueError(f"'{field}' must be a whole number.")
            values[field] = int(number) if number.is_integer() else number

    model = result["model"]
    if result["slug"] == "naive-bayes":
        frame = pd.Series([values["review_text"]])
    else:
        frame = pd.DataFrame([values], columns=list(feature_columns))
    label = str(model.predict(frame)[0])
    prediction: dict[str, Any] = {"label": label}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = model.named_steps["classifier"].classes_
        prediction["probabilities"] = {
            str(class_name): round(float(probability), 4)
            for class_name, probability in zip(classes, probabilities)
        }
    if result["slug"] == "knn":
        transformed = model.named_steps["scaler"].transform(frame)
        distances, indices = model.named_steps["classifier"].kneighbors(
            transformed,
            n_neighbors=config["neighbors"],
        )
        examples = result["training_examples"]
        targets = result["training_targets"]
        prediction["similar_profiles"] = [
            {
                **examples[int(index)],
                config["target"]: targets[int(index)],
                "distance": round(float(distance), 3),
            }
            for distance, index in zip(distances[0], indices[0])
        ]
    return prediction


def _create_pipeline(slug: str, random_state: int) -> Pipeline:
    if slug == "naive-bayes":
        return Pipeline(
            [
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        )
    if slug == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        )
    if slug == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=random_state)),
            ]
        )
    if slug == "decision-tree":
        return Pipeline(
            [
                ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=random_state)),
            ]
        )
    if slug == "neural-network":
        hidden_layers = MODEL_CONFIGS[slug]["hidden_layers"]
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation="relu",
                        solver="adam",
                        max_iter=220,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown classification model: {slug}")

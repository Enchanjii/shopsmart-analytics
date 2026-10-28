from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.linear_regression import PROJECT_ROOT


DATA_DIRECTORY = PROJECT_ROOT / "data" / "generated"

DATASET_COLUMNS = {
    "customer_reviews.csv": ("review_text", "sentiment"),
    "customer_similarity.csv": (
        "age",
        "purchase_frequency",
        "average_spend",
        "website_visits",
        "prior_purchases",
        "purchase_category",
    ),
    "purchase_intent_svm.csv": (
        "product_page_visits",
        "browsing_minutes",
        "cart_items",
        "previous_purchase",
        "discount_exposure",
        "purchase_intent",
    ),
    "customer_churn.csv": (
        "days_since_purchase",
        "purchases_per_year",
        "average_spend",
        "service_complaints",
        "cancelled_orders",
        "churn_risk",
    ),
    "purchase_intent_ann.csv": (
        "product_page_visits",
        "browsing_minutes",
        "cart_items",
        "previous_purchase",
        "discount_exposure",
        "email_clicks",
        "purchase",
    ),
}
DATASET_TARGET_LABELS = {
    "customer_reviews.csv": {"negative", "positive"},
    "customer_similarity.csv": {"occasional", "regular", "frequent"},
    "purchase_intent_svm.csv": {"no_purchase", "purchase"},
    "customer_churn.csv": {"active", "churned"},
    "purchase_intent_ann.csv": {"no_purchase", "purchase"},
}
NUMERIC_FEATURE_RANGES = {
    "customer_similarity.csv": {
        "age": (18, 75),
        "purchase_frequency": (1, 44),
        "average_spend": (10, 500),
        "website_visits": (1, 79),
        "prior_purchases": (0, 59),
    },
    "purchase_intent_svm.csv": {
        "product_page_visits": (1, 29),
        "browsing_minutes": (0.5, 45),
        "cart_items": (0, 8),
        "previous_purchase": (0, 1),
        "discount_exposure": (0, 1),
    },
    "customer_churn.csv": {
        "days_since_purchase": (1, 365),
        "purchases_per_year": (1, 39),
        "average_spend": (10, 500),
        "service_complaints": (0, 7),
        "cancelled_orders": (0, 6),
    },
    "purchase_intent_ann.csv": {
        "product_page_visits": (1, 34),
        "browsing_minutes": (0.5, 60),
        "cart_items": (0, 9),
        "previous_purchase": (0, 1),
        "discount_exposure": (0, 1),
        "email_clicks": (0, 11),
    },
}

POSITIVE_PHRASES = (
    "arrived earlier than expected",
    "is excellent quality",
    "was easy to use",
    "looks just like the photos",
    "was a great value",
    "made shopping simple",
)
NEGATIVE_PHRASES = (
    "arrived later than expected",
    "is disappointing quality",
    "was difficult to use",
    "looks different from the photos",
    "was poor value",
    "made shopping frustrating",
)
PRODUCTS = ("shirt", "mug", "backpack", "lamp", "headphones", "notebook")
REVIEW_OPENERS = (
    "I bought this {product} last week and it {phrase}.",
    "After trying the {product}, I found it {phrase}.",
    "My new {product} {phrase}, so I wanted to share my experience.",
    "For the price, the {product} {phrase}.",
    "The {product} {phrase} during my first week.",
    "I ordered a {product} online and it {phrase}.",
    "A quick review: this {product} {phrase}.",
    "In my experience, the {product} {phrase}.",
)
REVIEW_ENDINGS = (
    " I noticed this on my first order.",
    " This was my experience with the delivery.",
    " I am sharing this after using it for a week.",
    " The product page matched what I received.",
    " I would mention this to another shopper.",
    " This is feedback from a recent store order.",
    " I paid attention to the fit and finish.",
    " This was my first time trying this item.",
    " The packaging was part of the experience.",
    " I compared it with a similar item.",
    " This is an honest note about the purchase.",
    " I used it during my normal routine.",
)


def generate_review_dataset(
    path: str | Path | None = None,
    row_count: int = 600,
    random_state: int = 43,
) -> Path:
    if row_count < 40:
        raise ValueError("The review dataset must contain at least 40 rows.")
    rng = np.random.default_rng(random_state)
    labels = np.tile(np.array(["positive", "negative"]), row_count // 2 + 1)[:row_count]
    rng.shuffle(labels)
    remaining = {"positive": int(np.count_nonzero(labels == "positive")), "negative": int(np.count_nonzero(labels == "negative"))}
    reviews = []
    selected_labels = []
    seen_reviews: set[str] = set()
    while len(reviews) < row_count:
        available_labels = [label for label, count in remaining.items() if count > 0]
        label = available_labels[int(rng.integers(len(available_labels)))]
        phrases = POSITIVE_PHRASES if label == "positive" else NEGATIVE_PHRASES
        opener = REVIEW_OPENERS[int(rng.integers(len(REVIEW_OPENERS)))]
        review = opener.format(
                product=PRODUCTS[int(rng.integers(len(PRODUCTS)))],
                phrase=phrases[int(rng.integers(len(phrases)))],
            ) + REVIEW_ENDINGS[int(rng.integers(len(REVIEW_ENDINGS)))]
        if review in seen_reviews:
            continue
        seen_reviews.add(review)
        reviews.append(review)
        selected_labels.append(label)
        remaining[label] -= 1
    return _write_dataset(
        pd.DataFrame({"review_text": reviews, "sentiment": selected_labels}),
        path,
        "customer_reviews.csv",
    )


def generate_customer_similarity_dataset(
    path: str | Path | None = None,
    row_count: int = 900,
    random_state: int = 44,
) -> Path:
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 76, row_count)
    frequency = rng.integers(1, 45, row_count)
    average_spend = rng.uniform(10, 500, row_count)
    visits = rng.integers(1, 80, row_count)
    prior_purchases = rng.integers(0, 60, row_count)
    score = (
        0.38 * (frequency / 44)
        + 0.27 * (average_spend / 500)
        + 0.20 * (visits / 79)
        + 0.15 * (prior_purchases / 59)
        + rng.normal(0, 0.035, row_count)
    )
    category = np.where(score < 0.36, "occasional", np.where(score < 0.64, "regular", "frequent"))
    return _write_dataset(
        pd.DataFrame(
            {
                "age": age,
                "purchase_frequency": frequency,
                "average_spend": average_spend.round(2),
                "website_visits": visits,
                "prior_purchases": prior_purchases,
                "purchase_category": category,
            }
        ),
        path,
        "customer_similarity.csv",
    )


def generate_svm_purchase_dataset(
    path: str | Path | None = None,
    row_count: int = 900,
    random_state: int = 45,
) -> Path:
    rng = np.random.default_rng(random_state)
    page_visits = rng.integers(1, 30, row_count)
    browsing_minutes = rng.uniform(0.5, 45, row_count)
    cart_items = rng.integers(0, 9, row_count)
    previous_purchase = rng.integers(0, 2, row_count)
    discount_exposure = rng.integers(0, 2, row_count)
    log_odds = (
        -3.1
        + 0.095 * page_visits
        + 0.055 * browsing_minutes
        + 0.49 * cart_items
        + 0.78 * previous_purchase
        + 0.4 * discount_exposure
        + rng.normal(0, 0.9, row_count)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    intent = np.where(rng.random(row_count) < probability, "purchase", "no_purchase")
    return _write_dataset(
        pd.DataFrame(
            {
                "product_page_visits": page_visits,
                "browsing_minutes": browsing_minutes.round(2),
                "cart_items": cart_items,
                "previous_purchase": previous_purchase,
                "discount_exposure": discount_exposure,
                "purchase_intent": intent,
            }
        ),
        path,
        "purchase_intent_svm.csv",
    )


def generate_churn_dataset(
    path: str | Path | None = None,
    row_count: int = 900,
    random_state: int = 46,
) -> Path:
    rng = np.random.default_rng(random_state)
    days_since_purchase = rng.integers(1, 366, row_count)
    purchases_per_year = rng.integers(1, 40, row_count)
    average_spend = rng.uniform(10, 500, row_count)
    complaints = rng.integers(0, 8, row_count)
    cancelled_orders = rng.integers(0, 7, row_count)
    log_odds = (
        -2.7
        + 0.012 * days_since_purchase
        - 0.07 * purchases_per_year
        - 0.001 * average_spend
        + 0.36 * complaints
        + 0.48 * cancelled_orders
        + rng.normal(0, 0.75, row_count)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    churned = np.where(rng.random(row_count) < probability, "churned", "active")
    return _write_dataset(
        pd.DataFrame(
            {
                "days_since_purchase": days_since_purchase,
                "purchases_per_year": purchases_per_year,
                "average_spend": average_spend.round(2),
                "service_complaints": complaints,
                "cancelled_orders": cancelled_orders,
                "churn_risk": churned,
            }
        ),
        path,
        "customer_churn.csv",
    )


def generate_ann_purchase_dataset(
    path: str | Path | None = None,
    row_count: int = 1_000,
    random_state: int = 47,
) -> Path:
    rng = np.random.default_rng(random_state)
    page_visits = rng.integers(1, 35, row_count)
    browsing_minutes = rng.uniform(0.5, 60, row_count)
    cart_items = rng.integers(0, 10, row_count)
    previous_purchase = rng.integers(0, 2, row_count)
    discount_exposure = rng.integers(0, 2, row_count)
    email_clicks = rng.integers(0, 12, row_count)
    log_odds = (
        -3.5
        + 0.085 * page_visits
        + 0.048 * browsing_minutes
        + 0.46 * cart_items
        + 0.7 * previous_purchase
        + 0.34 * discount_exposure
        + 0.09 * email_clicks
        + rng.normal(0, 1, row_count)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    purchase = np.where(rng.random(row_count) < probability, "purchase", "no_purchase")
    return _write_dataset(
        pd.DataFrame(
            {
                "product_page_visits": page_visits,
                "browsing_minutes": browsing_minutes.round(2),
                "cart_items": cart_items,
                "previous_purchase": previous_purchase,
                "discount_exposure": discount_exposure,
                "email_clicks": email_clicks,
                "purchase": purchase,
            }
        ),
        path,
        "purchase_intent_ann.csv",
    )


def generate_all_datasets() -> list[Path]:
    generators = (
        generate_review_dataset,
        generate_customer_similarity_dataset,
        generate_svm_purchase_dataset,
        generate_churn_dataset,
        generate_ann_purchase_dataset,
    )
    return [generator() for generator in generators]


def load_classification_dataset(name: str) -> pd.DataFrame:
    if name not in DATASET_COLUMNS:
        raise ValueError(f"Unknown classification dataset: {name}")
    path = DATA_DIRECTORY / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m scripts.generate_ml_datasets`."
        )
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read dataset {name}.") from error

    expected_columns = DATASET_COLUMNS[name]
    if tuple(dataset.columns) != expected_columns:
        raise ValueError(f"Dataset {name} has an unexpected column schema.")
    if len(dataset) < 40:
        raise ValueError(f"Dataset {name} must contain at least 40 rows.")
    if dataset.isna().any().any():
        raise ValueError(f"Dataset {name} contains missing values.")
    if dataset.duplicated().any():
        raise ValueError(f"Dataset {name} contains duplicate records.")
    target_column = expected_columns[-1]
    actual_labels = set(dataset[target_column].astype(str).unique())
    if actual_labels != DATASET_TARGET_LABELS[name]:
        raise ValueError(
            f"Dataset {name} has unexpected target labels: "
            f"{sorted(actual_labels)}."
        )
    if any((dataset[target_column].astype(str) == label).sum() < 5 for label in DATASET_TARGET_LABELS[name]):
        raise ValueError(f"Dataset {name} needs at least five records in each target class.")
    if name == "customer_reviews.csv":
        if not dataset["review_text"].map(lambda value: isinstance(value, str) and value.strip()).all():
            raise ValueError("Review dataset contains empty or invalid text.")
    else:
        for column in expected_columns[:-1]:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
        numeric_values = dataset[list(expected_columns[:-1])].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"Dataset {name} contains non-numeric or non-finite features.")
        for column, (minimum, maximum) in NUMERIC_FEATURE_RANGES[name].items():
            if not dataset[column].between(minimum, maximum).all():
                raise ValueError(f"Dataset {name} feature '{column}' contains out-of-range values.")
        for column in ("previous_purchase", "discount_exposure"):
            if column in dataset.columns and not dataset[column].isin((0, 1)).all():
                raise ValueError(f"Dataset {name} feature '{column}' must contain only 0 or 1.")
    return dataset


def _write_dataset(dataset: pd.DataFrame, path: str | Path | None, filename: str) -> Path:
    output_path = Path(path) if path is not None else DATA_DIRECTORY / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path

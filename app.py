from flask import Flask, jsonify, render_template, request

from ml.classification_models import MODEL_CONFIGS, predict_classification, train_classification_model
from ml.linear_regression import (
    FEATURE_LIMITS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    predict_sales,
    train_sales_model,
)


ALGORITHMS = [
    {
        "name": "Linear Regression",
        "slug": "linear-regression",
        "business_problem": "Forecast product sales from retail activity.",
        "status": "Loading",
    },
    {
        "name": "Naive Bayes",
        "slug": "naive-bayes",
        "business_problem": "Classify the sentiment of customer feedback.",
        "status": "Loading",
    },
    {
        "name": "K-Nearest Neighbors (KNN)",
        "slug": "knn",
        "business_problem": "Find customers with similar purchasing patterns.",
        "status": "Loading",
    },
    {
        "name": "Support Vector Machine (SVM)",
        "slug": "svm",
        "business_problem": "Estimate whether a customer intends to purchase.",
        "status": "Loading",
    },
    {
        "name": "Decision Tree",
        "slug": "decision-tree",
        "business_problem": "Identify customers who may be at risk of churn.",
        "status": "Loading",
    },
    {
        "name": "Artificial Neural Network (ANN)",
        "slug": "neural-network",
        "business_problem": "Demonstrate neural-network purchase prediction.",
        "status": "Loading",
    },
]

PREDICTION_INTERPRETATIONS = {
    "naive-bayes": {
        "positive": "This synthetic review was classified as positive; aggregate positive and negative trends can guide human review of feedback.",
        "negative": "This synthetic review was classified as negative; investigate recurring concerns rather than treating one automated label as a full explanation.",
    },
    "knn": {
        "occasional": "This profile is closest to the occasional-purchaser segment in the synthetic history.",
        "regular": "This profile is closest to the regular-purchaser segment in the synthetic history.",
        "frequent": "This profile is closest to the frequent-purchaser segment in the synthetic history.",
    },
    "svm": {
        "purchase": "The browsing profile falls on the purchase-intent side of the SVM boundary for this synthetic task.",
        "no_purchase": "The browsing profile falls on the no-purchase side of the SVM boundary for this synthetic task.",
    },
    "decision-tree": {
        "active": "The profile follows tree rules associated with active customers in this synthetic dataset.",
        "churned": "The profile follows tree rules associated with churned customers in this synthetic dataset; a retention team could review it without assuming a cause.",
    },
    "neural-network": {
        "purchase": "The network classifies this session as purchase for the synthetic task; this is not a calibrated guarantee.",
        "no_purchase": "The network classifies this session as no purchase for the synthetic task; this is not a calibrated guarantee.",
    },
}


def create_app():
    app = Flask(__name__)
    model_results = {}
    model_errors = {}
    trainers = {
        "linear-regression": train_sales_model,
        **{
            slug: lambda slug=slug: train_classification_model(slug)
            for slug in MODEL_CONFIGS
        },
    }
    for slug, trainer in trainers.items():
        try:
            model_results[slug] = trainer()
        except Exception as error:
            app.logger.exception("Model initialization failed for %s", slug)
            model_errors[slug] = str(error)
    app.extensions["model_results"] = model_results
    app.extensions["model_errors"] = model_errors
    app.extensions["algorithm_status"] = {
        slug: "Ready" if slug in model_results else "Unavailable"
        for slug in trainers
    }

    algorithm_cards = []
    for card in ALGORITHMS:
        algorithm_cards.append(
            {**card, "status": app.extensions["algorithm_status"][card["slug"]]}
        )

    @app.get("/")
    def home():
        return render_template("index.html", algorithms=algorithm_cards)

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/models/linear-regression")
    def linear_regression():
        if "linear-regression" not in model_results:
            return render_template(
                "model_unavailable.html",
                algorithm=ALGORITHMS[0],
                error=model_errors.get("linear-regression"),
            ), 503
        return render_template(
            "linear_regression.html",
            model_result=model_results["linear-regression"],
            feature_columns=FEATURE_COLUMNS,
            feature_limits=FEATURE_LIMITS,
            target_column=TARGET_COLUMN,
            feature_descriptions={
                "advertising_spend": "Advertising budget in USD",
                "product_price": "Product price in USD",
                "discount_percent": "Applied discount percentage",
                "store_traffic": "Number of store visits",
                "previous_sales": "Previous sales amount in USD",
                "weekend": "1 for weekend, otherwise 0",
                "holiday_season": "1 during holiday season, otherwise 0",
            },
            feature_types={
                "advertising_spend": "Decimal",
                "product_price": "Decimal",
                "discount_percent": "Decimal",
                "store_traffic": "Integer",
                "previous_sales": "Decimal",
                "weekend": "Binary integer (0/1)",
                "holiday_season": "Binary integer (0/1)",
            },
        )

    @app.get("/models/<slug>")
    def model_page(slug):
        if slug == "linear-regression":
            return linear_regression()
        config = MODEL_CONFIGS.get(slug)
        if config is None:
            return jsonify({"error": "Unknown model."}), 404
        if slug not in model_results:
            return render_template(
                "model_unavailable.html",
                algorithm={
                    "name": config["name"],
                    "slug": slug,
                    "business_problem": config["business_problem"],
                },
                error="Check the Flask terminal output for the model initialization error.",
            ), 503
        result = model_results[slug]
        return render_template(
            "classification_model.html",
            config=config,
            model_result=result,
        )

    @app.post("/api/predict/linear-regression")
    def predict_linear_regression():
        features = request.get_json(silent=True)
        if not isinstance(features, dict):
            return jsonify({"error": "Send the model inputs as a JSON object."}), 400
        try:
            prediction = predict_sales(model_results["linear-regression"]["model"], features)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except KeyError:
            return jsonify({"error": "Linear Regression is currently unavailable."}), 503

        return jsonify(
            {
                "prediction": {TARGET_COLUMN: round(prediction, 2)},
                "unit": "USD",
                "interpretation": (
                    "This is an educational estimate from synthetic data, "
                    "not a guarantee of actual sales."
                ),
            }
        )

    @app.post("/api/predict/<slug>")
    def predict_model(slug):
        config = MODEL_CONFIGS.get(slug)
        if config is None:
            return jsonify({"error": "Unknown model."}), 404
        result = model_results.get(slug)
        if result is None:
            return jsonify({"error": f"{config['name']} is currently unavailable."}), 503
        submitted = request.get_json(silent=True)
        if not isinstance(submitted, dict):
            return jsonify({"error": "Send the model inputs as a JSON object."}), 400
        try:
            prediction = predict_classification(result, submitted)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(
            {
                "prediction": prediction,
                "target": config["target"],
                "interpretation": PREDICTION_INTERPRETATIONS[slug][prediction["label"]],
            }
        )

    @app.get("/comparison")
    def comparison():
        comparison_algorithms = []
        for card in algorithm_cards:
            result = model_results.get(card["slug"])
            if card["slug"] == "linear-regression":
                task_type = "Regression"
                target = TARGET_COLUMN
                dataset = "sales_forecast.csv"
                metrics = result["metrics"] if result else {}
            else:
                config = MODEL_CONFIGS[card["slug"]]
                task_type = "Text classification" if card["slug"] == "naive-bayes" else "Classification"
                target = config["target"]
                dataset = config["dataset"]
                metrics = result["metrics"] if result else {}
            comparison_algorithms.append(
                {
                    **card,
                    "task_type": task_type,
                    "target": target,
                    "dataset": dataset,
                    "metrics": metrics,
                }
            )
        return render_template("comparison.html", algorithms=comparison_algorithms)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok" if not model_errors else "degraded",
                "application": "ShopSmart Analytics",
                "ready_models": sorted(model_results),
                "unavailable_models": sorted(model_errors),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

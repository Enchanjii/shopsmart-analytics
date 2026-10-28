# ShopSmart Analytics presentation and demonstration guide

## Suggested 8–10 minute flow

1. **Company (1 minute):** ShopSmart Analytics is a fictional retail analytics company. Its objectives are sales planning, feedback analysis, customer segmentation, purchase-intent estimation, churn-risk review, and neural-network prediction.
2. **Data and safeguards (1 minute):** Explain that the CSVs are reproducibly generated synthetic examples, not customer data. Show a dataset preview and explain the target label before discussing a score.
3. **Live website (3–4 minutes):** Start Flask, open the home page, show the Model labs navigation and model readiness, open each lab, inspect a held-out evaluation chart/metrics, then submit one example prediction per model. The user prediction is distinct from test-set evaluation.
4. **Compare tasks (1 minute):** Open `/comparison`. Explain that regression's MAE/RMSE/R² and classifiers' accuracy/precision/recall/F1 answer different questions; do not rank their values across unlike tasks.
5. **Limitations and Q&A (2–3 minutes):** Emphasize synthetic labels, small demonstration scope, non-causal associations, and need for validation on representative, consented real-world data before any business use.

## Run the live demo (Windows PowerShell)

From the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.generate_sales_dataset
python -m scripts.generate_ml_datasets
python app.py
```

Open `http://127.0.0.1:5000`. Check `http://127.0.0.1:5000/health` for ready/unavailable model status. If a model is unavailable, check the Flask terminal logs; do not present invented substitute metrics. Stop Flask with `Ctrl+C`.

## Model talking points and observed held-out results

The figures below were measured from the current deterministic datasets and held-out splits using the installed scikit-learn environment. They are educational synthetic-data results and may vary if package versions or code change. The live pages compute and display their metrics from model predictions rather than hardcoding these figures.

| Model / business question | Data and target | Why this model | Observed held-out metrics |
|---|---|---|---|
| Linear Regression: forecast sales | `sales_forecast.csv`; `sales_amount` | Continuous numeric target and an explainable baseline | MAE **$177.88**; RMSE **$226.69**; R² **0.978** |
| Naive Bayes: classify feedback | `customer_reviews.csv`; `sentiment` | TF-IDF word features with MultinomialNB | Accuracy **1.000** (majority baseline **0.500**); weighted precision **1.000**, recall **1.000**, F1 **1.000** |
| KNN: identify similar purchasing profiles | `customer_similarity.csv`; `purchase_category` | Predict from nearby scaled profiles; expose nearest synthetic examples | Accuracy **0.806** (majority baseline **0.611**); weighted precision **0.803**, recall **0.806**, F1 **0.802** |
| SVM: classify purchase intent | `purchase_intent_svm.csv`; `purchase_intent` | RBF decision boundary for behavior-feature classification | Accuracy **0.722** (majority baseline **0.772**); weighted precision **0.783**, recall **0.722**, F1 **0.741** |
| Decision Tree: flag churn risk | `customer_churn.csv`; `churn_risk` | Shallow tree provides readable split rules | Accuracy **0.722** (majority baseline **0.567**); weighted precision **0.721**, recall **0.722**, F1 **0.720** |
| Feedforward ANN: classify purchase | `purchase_intent_ann.csv`; `purchase` | MLP learns layered nonlinear combinations of scaled inputs | Accuracy **0.815** (majority baseline **0.825**); weighted precision **0.679**, recall **0.815**, F1 **0.741** |

For the classification majority baseline, the most frequent label is selected from training data and then scored against the same held-out test records. On this run, SVM and ANN accuracy did not exceed that simple baseline; avoid presenting them as business-ready wins. The Naive Bayes result is perfect on this test split because the synthetic reviews contain clear sentiment phrases; it does not demonstrate performance on ambiguous natural customer feedback. All other results also reflect their generated label rules, not real customer behavior.

## Suggested demonstration inputs

- **Linear Regression:** Use the prefilled scenario and explain MAE as an average absolute dollar error on the held-out synthetic test set.
- **Naive Bayes:** `The backpack is excellent quality and was a great value.`
- **KNN:** age 34, purchase frequency 22, average spend 180.5, website visits 40, prior purchases 20. Point out that feature scaling makes distances meaningful across unlike units.
- **SVM:** 12 product-page visits, 15.5 browsing minutes, 2 cart items, previous purchase Yes, discount exposure Yes.
- **Decision Tree:** 120 days since last purchase, 8 purchases in the past year, average spend 95, 2 complaints, 1 cancelled order. Read the displayed splits as associations, not causes.
- **ANN:** 12 product-page visits, 15.5 browsing minutes, 2 cart items, previous purchase Yes, discount exposure Yes, 2 recent email clicks. Explain that training loss and validation accuracy are drawn from training data; test metrics use a separate held-out split.

## Likely instructor questions

**Why use synthetic data?** It keeps the demonstration reproducible and avoids introducing real customer information. Its limitation is that generated relationships do not establish real-world performance.

**What does a held-out test set do?** It checks performance on records not used to fit the model. Classification preprocessing lives inside the sklearn pipeline and is fitted on training records only.

**Why scale for KNN and SVM?** These methods use distances or margin geometry. Scaling prevents measurements with larger numeric ranges from dominating those calculations.

**Why not scale the Decision Tree?** Tree splits depend on feature ordering and thresholds, so simple monotonic rescaling does not require standardization.

**Why is the ANN not TensorFlow/Keras?** TensorFlow was not installed in the existing native Windows/Python 3.14 environment. The project uses the explicitly permitted feedforward scikit-learn `MLPClassifier`, with a fitted scaling step, two hidden layers, early stopping, and recorded training history. This implementation was actually trained and tested in the project environment.

**Does high accuracy mean the model is ready for retail use?** No. It describes only the generated dataset and selected test split. Real deployment requires representative data, leakage checks, subgroup/fairness analysis, calibration where appropriate, monitoring, privacy controls, and human oversight.

**Are model predictions causal?** No. The models learn associations from generated examples; they do not prove that a discount, complaint, or browsing event causes a purchase or churn.

**Can we compare the scores on `/comparison`?** Only within context. Regression errors and classification scores are different metrics, and models use different datasets and targets. Cross-task score ranking is not meaningful.

**Why include a majority baseline?** It provides a simple reference for imbalanced classification. Accuracy alone can look useful even if a model barely improves—or does worse than—always predicting the most frequent training class.

**What happens if model training fails?** The app logs the initialization exception, marks that model unavailable in `/health` and the overview, and returns an unavailable page/HTTP 503 rather than showing fake metrics.

## Group contribution planning

Assign roles based on actual work and group agreement; do not claim a contribution that was not made. A possible four-person split is: (1) company/site design, (2) datasets and Naive Bayes/KNN, (3) SVM/Decision Tree, and (4) ANN, integration, and testing. Every presenter should be prepared to explain the data, target, evaluation split, limitations, and their own changes.

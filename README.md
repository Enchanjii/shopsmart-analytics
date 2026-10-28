# ShopSmart Analytics

ShopSmart Analytics is a fictional retail analytics company used for a college project about practical machine learning. It demonstrates six trained models against reproducible, synthetic retail data. Predictions and evaluation results are produced by the trained pipelines; the data is for education, not evidence of production performance.

## Implemented features

- Flask application with a reusable application factory.
- Responsive home and company information pages.
- Overview and dedicated interactive lab pages for all six required algorithms.
- Fictional-company and synthetic-data notices.
- Dataset previews, feature explanations, targets, and source/generation notes on each model page.
- Held-out evaluation metrics for regression and classification, confusion charts, and ANN training charts.
- Validated forms and JSON prediction APIs for all six models.
- Similar customer profiles from KNN and readable rules from the Decision Tree.
- Model comparison page that warns against comparing unrelated task metrics.
- `/health` endpoint reporting ready and unavailable models.
- pytest coverage for training, evaluation, input validation, routes, and initialization failures.
- Bootstrap 5 styling and project-specific responsive styles.

## Technology

- Python 3.14.6 is the verified local interpreter for this workspace. The installed Flask, pandas, NumPy, scikit-learn, and pytest versions work in its virtual environment.
- Flask 3.x, pandas, NumPy, scikit-learn, and pytest.
- Bootstrap 5.3.8 and custom CSS.

The ANN is a feedforward scikit-learn `MLPClassifier` with two hidden layers, feature scaling in its pipeline, training loss, and held-out test metrics. TensorFlow was not installed in the existing native Windows/Python 3.14 environment. Rather than add a large framework without verifying its support for this runtime, the project uses the explicitly allowed scikit-learn MLP alternative, which is installed and exercised by the tests.

## Create the Windows environment (PowerShell)

From the project directory, create and activate a virtual environment. If the Python launcher recognizes Python 3.14:

```powershell
py -3.14 --version
py -3.14 -m venv .venv
```

In the current development environment, Python 3.14.6 is installed under uv but is not registered with the Python launcher. Use the discovered interpreter path as a fallback:

```powershell
$python = "$env:APPDATA\uv\python\cpython-3.14.6-windows-x86_64-none\python.exe"
& $python --version
& $python -m venv .venv
```

Then activate the environment and install the project dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, either use a permitted execution policy for your account or run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Datasets and models

All datasets are generated locally with fixed NumPy random seeds. No external dataset, API, or real customer information is used. Regenerate the classification datasets with `python -m scripts.generate_ml_datasets`; the Linear Regression data is generated separately with `python -m scripts.generate_sales_dataset`.

| Model | Dataset / target | Method |
|---|---|---|
| Linear Regression | `sales_forecast.csv`; numeric `sales_amount` | LinearRegression; numeric input validation; MAE, RMSE, R² |
| Naive Bayes | `customer_reviews.csv`; `sentiment` | TF-IDF + MultinomialNB; accuracy, weighted precision/recall/F1, confusion matrix |
| KNN | `customer_similarity.csv`; `purchase_category` | StandardScaler + 7-neighbor KNeighborsClassifier; returns nearest synthetic profiles |
| SVM | `purchase_intent_svm.csv`; `purchase_intent` | StandardScaler + RBF SVC |
| Decision Tree | `customer_churn.csv`; `churn_risk` | DecisionTreeClassifier (max depth 4); readable learned rules |
| ANN | `purchase_intent_ann.csv`; `purchase` | StandardScaler + feedforward MLPClassifier (16, 8 hidden units; early stopping) |

Regression uses a deterministic 80/20 test split. Each classifier uses a deterministic stratified 80/20 split. Text vectorization and numeric scaling are steps in sklearn pipelines, so they are fit on training records only. ANN validation data is selected by the MLP from its training split; the held-out test set is not used for fitting or early stopping. Metrics are computed from predictions on held-out data.

The application validates each dataset's schema, minimum row and target-class counts, missing values, numeric types and finite values before training. Prediction inputs have server-side feature checks and range/enum validation. If an algorithm cannot initialize, the site reports it as unavailable, logs the initialization error, and does not display fabricated metrics or predictions.

## Generate data, run, and test

Create and activate the virtual environment using the setup commands above, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate all five classification datasets:

```powershell
python -m scripts.generate_ml_datasets
```

Generate the sales-forecast dataset:

```powershell
python -m scripts.generate_sales_dataset
```

Start the application:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Choose **Model labs** in the navigation or a model card on the home page. Enter a hypothetical review/profile/session and submit the form. The browser sends the input to its matching Flask JSON endpoint. Test metrics and evaluation charts are separate from the individual prediction and come from that model's held-out split. All examples use synthetic data.

Run the complete test suite from the project root:

```powershell
python -m pytest -q
```

Check application readiness at <http://127.0.0.1:5000/health>. It returns ready/unavailable model slugs. Stop the Flask server with `Ctrl+C`.

Bootstrap and Chart.js are loaded from jsDelivr, so responsive Bootstrap styling and charts require internet access. The pages also include local CSS for the project's brand and layout.

## Project structure

```text
app.py
ml/
  __init__.py
  linear_regression.py
  datasets.py
  classification_models.py
scripts/
  generate_sales_dataset.py
  generate_ml_datasets.py
data/
  generated/
    sales_forecast.csv
    customer_reviews.csv
    customer_similarity.csv
    purchase_intent_svm.csv
    customer_churn.csv
    purchase_intent_ann.csv
requirements.txt
README.md
.gitignore
templates/
  base.html
  index.html
  about.html
  linear_regression.html
  classification_model.html
  comparison.html
static/
  css/
    style.css
  js/
    linear-regression.js
    classification.js
tests/
  test_linear_regression.py
  test_classification_models.py
```

## Limitations

- Synthetic labels are derived from simple phrase templates or generated formulas, so good test scores do not establish that the models will generalize to real customers.
- The Naive Bayes text is deliberately generated with sentiment-related language and is much easier than ambiguous real reviews.
- The KNN categories and churn/purchase labels are simulated; model associations are not causal findings.
- SVC/MLP probability/confidence outputs, where shown, are model estimates, not calibrated business certainty.
- The MLP training/validation curve can vary with software versions and optimization behavior; it is not a production training study.
- The site uses Bootstrap and Chart.js from jsDelivr; visual styling and charts require internet access. Server-rendered information and prediction APIs are local.
- No browser-based cross-device visual test is implied by running pytest. Use the presentation guide's browser steps before a live demo.

See [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for a demonstration outline and instructor Q&A preparation.

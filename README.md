# SalesIQ — Sales Intelligence and Prediction

SalesIQ analyzes Amazon India sales from March–May 2022 and compares machine
learning models for order-amount regression and cancellation classification.
The project is a single Python script that can be run from Spyder/Anaconda on
Windows or from a terminal.

## Project files

- `main.py` — data cleaning, EDA, model training, evaluation, and output saving.
- `Amazon India Sales DataExcel.xlsx` — the original workbook; it is not modified.
- `requirements.txt` — Python packages used by the script.
- `images/` — charts already present in the repository; these are preserved.
- `outputs/` — generated charts and CSV evaluation/analysis tables.

The script reads the **Amazon India Sales Data** worksheet, not the workbook's
pivot/summary sheets. It uses the named columns in that worksheet and ignores
unheaded helper columns.

## Setup and run

Keep the Excel workbook in the same folder as `main.py`. In Anaconda Prompt,
install the dependencies and run the script:

```text
pip install -r requirements.txt
python main.py
```

In Spyder, open `main.py`, select the project folder as the working directory,
and run the file. The script also resolves the workbook relative to its own
location, so its data path does not depend on a user-specific Windows path.

## Analysis

The script reports data shape, missing values, exact duplicate rows, repeated
order IDs, and unparseable dates. Repeated order IDs are retained because an
order can contain multiple line items. Missing feature values are imputed
inside model pipelines; missing `Amount` targets are excluded from regression
rather than being filled with invented target values.

EDA includes descriptive statistics, category-wise sales, monthly trends,
state-wise sales, numeric correlations, order-status counts, and overall and
category-level cancellation rates where the corresponding columns are present.

Regression models predict `Amount` with Linear Regression and Random Forest
Regressor. They are evaluated with MAE, MSE, RMSE, and R².

Cancellation classification uses `Status == "Cancelled"` as its binary target.
It runs Logistic Regression, Decision Tree Classifier, and Random Forest
Classifier only when both target classes are present. It reports confusion
matrices, accuracy, precision, recall, F1, ROC-AUC, and log loss. The inspected
workbook has both cancelled and non-cancelled statuses, so this target is
supported.

Both tasks use a reproducible 80/20 `GroupShuffleSplit` by `Order ID`, keeping
all line items for an order on the same side of the split. This prevents rows
from the same order appearing in both training and test sets. Preprocessing
(median imputation, most-frequent categorical imputation, scaling, and
one-hot encoding) is fitted within each model pipeline using training data
only. For cancellation prediction, fields that can reveal the outcome,
including quantity, amount, modified amount, status-group fields, and shipping
outcomes, are not used as predictors.

## Generated outputs

The script creates `outputs/` when it runs. It saves descriptive statistics,
data quality details, EDA tables and charts, regression metrics, classification
metrics, per-model confusion matrices, and an ROC-curve chart when applicable.
The output files are local analysis artifacts; running the script does not
commit or upload anything.

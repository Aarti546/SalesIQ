# SalesIQ — Sales Intelligence and Prediction

SalesIQ is a machine learning project that analyzes Amazon India sales data and evaluates models for sales amount prediction and order cancellation classification.

## Project Objectives

- Clean and preprocess sales data.
- Explore sales patterns through visualizations.
- Predict sales amount using regression models.
- Classify orders as cancelled or not cancelled.
- Compare machine learning models using appropriate evaluation metrics.

## Dataset

- **Dataset:** Amazon India Sales Data
- **Period:** March–May 2022
- **Original records:** Approximately 65,000
- **Regression target:** `amount`
- **Classification target:** `is_cancelled`

The `is_cancelled` target is derived from the order status. The original `status` column is excluded from classification features to avoid target leakage.

## Data Preprocessing and EDA

The project includes column-name standardization, numeric conversion, removal of unnecessary columns, handling of missing values, and preparation of separate datasets for regression and classification.

Exploratory data analysis includes sales amount distribution, category-wise sales, monthly sales trends, a correlation heatmap, top states by sales, fulfilment analysis, and cancellation counts.

## Machine Learning Models

### Regression

1. Linear Regression
2. Random Forest Regressor

Both models use `qty` as the input feature and `amount` as the target.

| Model | MAE | R² Score |
|---|---:|---:|
| Linear Regression | 212.9430 | 0.0114 |
| Random Forest Regressor | 211.8403 | 0.0355 |

**Finding:** Random Forest Regressor performs slightly better on both metrics. However, both models have low R² scores, indicating that quantity alone is insufficient to explain sales amount variation.

### Classification

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)

The target is `is_cancelled`. The features include quantity, category, size, fulfilment, shipping service level, and B2B status. The classification evaluation uses a stratified 80/20 train-test split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9568 | 0.9953 | 0.6975 | 0.8202 |
| Decision Tree | 0.9568 | 0.9953 | 0.6975 | 0.8202 |
| KNN | 0.9559 | 0.9937 | 0.6921 | 0.8159 |

**Finding:** Logistic Regression and Decision Tree tie on the reported metrics. KNN performs slightly worse. Although precision is high, recall is approximately 69–70%, meaning some cancelled orders are missed.

## Final Conclusion

Among the tested regression models, Random Forest Regressor performs best, although its predictive power is limited with the current feature selection. For cancellation classification, Logistic Regression and Decision Tree produce identical predictions on the test set and tie on the reported metrics.

The project demonstrates data preprocessing, exploratory analysis, regression, classification, and model evaluation. Future improvements could include additional relevant features, feature engineering, and further evaluation.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Git and GitHub

## How to Run

1. Clone or download this repository.
2. Install the required Python packages.
3. Ensure the dataset is available at the path used in the script, or update the path.
4. Run the main Python file.

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Status

The five planned machine learning models and the final model comparison have been implemented. The project documents the observed results and limitations of the current feature selection.

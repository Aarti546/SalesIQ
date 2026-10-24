import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    classification_report
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("C:/Users/Udit ComputerS/Desktop/SalesIQ/Amazon India Sales Data.csv")

# Keeping an untouched copy for reference
df_raw = df.copy()

# Gathering Information related to dataset
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# DATA CLEANING-------------------------
# 1. Fix column names 
df.columns = df.columns.str.strip().str.lower()
# 2. Replace spaces and hyphens with underscores
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("-", "_")


#Ensure Amount is numeric
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')


# 4. REMOVE UNNECESSARY COLUMNS

# Remove extra columns if they exist

if "unnamed:_22" in df.columns:
    df.drop("unnamed:_22", axis=1, inplace=True)

if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

# Keep only columns required for SalesIQ

required_columns = [
    "date",
    "status",
    "fulfilment",
    "ship_service_level",
    "category",
    "size",
    "qty",
    "amount",
    "ship_state",
    "b2b"
]

df = df[required_columns].copy()

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())



# Remove rows where date is missing
df = df.dropna(subset=["date"])

# Fill missing categorical values
df["ship_state"] = df["ship_state"].fillna("Unknown")
df["fulfilment"] = df["fulfilment"].fillna("Unknown")
df["ship_service_level"] = df["ship_service_level"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
df["size"] = df["size"].fillna("Unknown")

# Keeping missing amounts for now.
# Regression models will use rows where amount is available.

# Create cancellation target
df["is_cancelled"] = (df["status"] == "Cancelled").astype(int)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)


# Dataset for regression models
# Amount is the target, so rows with missing amount cannot be used
regression_df = df.dropna(subset=["amount"]).copy()

# Dataset for classification models
classification_df = df.copy()

print("Regression dataset:", regression_df.shape)
print("Classification dataset:", classification_df.shape)

print("\nCancellation distribution:")
print(classification_df["is_cancelled"].value_counts())

print("\nMissing values in regression dataset:")
print(regression_df.isnull().sum())

print("\nCleaned datasets saved successfully!")
#--------------------------------------cleaning done

#--------------------------EDA-------------------
print("Dataset shape:", df.shape)

print("\nSales amount summary:")
print(regression_df["amount"].describe())

print("\nQuantity summary:")
print(df["qty"].describe())

print("\nOrder status counts:")
print(df["status"].value_counts())

print("\nCancellation percentage:")
print(round(df["is_cancelled"].mean() * 100, 2), "%")


#  EDA-Statistical Summary
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())
print("\nCancellation counts:\n", df["is_cancelled"].value_counts())

# VISUALIZATIONS

# 1. Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(regression_df['amount'], bins=30, color='orange', kde=True)
plt.title("Sales Amount Distribution", fontsize=14)
plt.xlabel("Sales Amount")
plt.show()


# 2. Category-wise Sales
plt.figure(figsize=(10,5))
sns.barplot(x='category', y='amount', data=regression_df, estimator=sum, color='purple')
plt.xticks(rotation=45)
plt.title("Category-wise Sales", fontsize=14)
plt.ylabel("Total Sales")
plt.show()


# 3. Monthly Sales Trend
# Convert date column to datetime format
regression_df['date'] = pd.to_datetime(regression_df['date'], errors='coerce')

# Extract month
regression_df['month'] = regression_df['date'].dt.month

# Calculate monthly sales
monthly_sales = regression_df.groupby('month')['amount'].sum()

# Visualize monthly sales trend
plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o', color='red')
plt.title("Monthly Sales Trend", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks([3, 4, 5], ['March', 'April', 'May'])
plt.show()



# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 5. Top 10 States by Sales
top_states = regression_df.groupby('ship_state')['amount'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_states.plot(kind='bar', color='pink')
plt.title("Top 10 States by Sales", fontsize=14)
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()


# 6. Fulfilment Method Analysis
plt.figure(figsize=(8,5))
sns.countplot(x='fulfilment', data=df, color='skyblue')
plt.title("Orders by Fulfilment Method", fontsize=14)
plt.xlabel("Fulfilment Method")
plt.ylabel("Number of Records")
plt.show()


# 7. Order Cancellation Analysis
plt.figure(figsize=(8,5))
sns.countplot(x='is_cancelled', data=df, color='salmon')
plt.title("Order Cancellation Analysis", fontsize=14)
plt.xlabel("Cancellation (0 = Not Cancelled, 1 = Cancelled)")
plt.ylabel("Number of Records")
plt.show()


#-----------------MODALS----------------
# 1.LINEAR REGRESSION: Predicting Sales Amount--------------------

# Select input feature and target
X = regression_df[['qty']]
y = regression_df['amount']

print("Input features shape:", X.shape)
print("Target shape:", y.shape)

print("\nMissing values in X:")
print(X.isnull().sum())

print("\nMissing values in y:")
print(y.isnull().sum())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict sales amount on test data
y_pred = model.predict(X_test)

print("First 10 actual values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted values:")
print(y_pred[:10])

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("R2 Score:", round(r2, 4))


print("Quantity distribution:")
print(regression_df['qty'].value_counts().sort_index())

print("\nNumber of unique quantity values:")
print(regression_df['qty'].nunique())


# Actual vs Predicted Sales Visualization
plt.figure(figsize=(7, 5))

plt.scatter(
    y_test, y_pred,
    alpha=0.4,
    color='green'
)

# Reference line for perfect predictions
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linestyle='--',
    label='Perfect Prediction'
)

plt.xlabel("Actual Sales Amount")
plt.ylabel("Predicted Sales Amount")
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.legend()
plt.show()

# MODEL 2: RANDOM FOREST REGRESSOR--------------------------------

X = regression_df[['qty']]
y = regression_df['amount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train the model
rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_reg.fit(X_train, y_train)

print("Random Forest Regressor trained successfully!")
# Predictions
y_pred_rf = rf_reg.predict(X_test)

print("First 10 actual values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted values:")
print(y_pred_rf[:10])

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Results")
print("MAE:", round(mae_rf, 2))
print("R2 Score:", round(r2_rf, 4))

# Visualize the model
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred_rf, alpha=0.4, color='purple')

plt.xlabel("Actual Sales Amount")
plt.ylabel("Predicted Sales Amount")
plt.title("Random Forest Regressor: Actual vs Predicted Sales")

plt.show()




# MODEL 3: LOGISTIC REGRESSION------------------------

# Select input features
features = [
    'qty',
    'category',
    'size',
    'fulfilment',
    'ship_service_level',
    'b2b'
]

X = classification_df[features].copy()
y = classification_df['is_cancelled'].copy()

print("Input data shape:", X.shape)
print("Target data shape:", y.shape)

print("\nCancellation distribution:")
print(y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert categorical features into numerical columns
categorical_columns = [
    'category',
    'size',
    'fulfilment',
    'ship_service_level',
    'b2b'
]

X_train_encoded = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    dtype=int
)

# Ensure both datasets have the same columns
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

print("Training data:", X_train_encoded.shape)
print("Testing data:", X_test_encoded.shape)
# Train the model
clf = LogisticRegression(max_iter=1000, random_state=42)

clf.fit(X_train_encoded, y_train)

print("Logistic Regression model trained successfully!")
# Predictions
y_pred = clf.predict(X_test_encoded)

print("First 20 actual values:")
print(y_test.head(20).values)

print("\nFirst 20 predicted values:")
print(y_pred[:20])
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

print("\nLOGISTIC REGRESSION RESULTS")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    zero_division=0
))
# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Cancelled', 'Cancelled'],
    yticklabels=['Not Cancelled', 'Cancelled']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# MODEL 4: DECISION TREE CLASSIFIER-----------------------

# Train the model
dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)

dt_clf.fit(X_train_encoded, y_train)

print("Decision Tree Classifier trained successfully!")
# Predictions
y_pred_dt = dt_clf.predict(X_test_encoded)

print("First 20 actual values:")
print(y_test.head(20).values)

print("\nFirst 20 predicted values:")
print(y_pred_dt[:20])
# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)

precision_dt = precision_score(
    y_test, y_pred_dt, zero_division=0
)

recall_dt = recall_score(
    y_test, y_pred_dt, zero_division=0
)

print("\nDECISION TREE CLASSIFIER RESULTS")
print(f"Accuracy:  {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall:    {recall_dt:.4f}")

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("\nConfusion Matrix:")
print(cm_dt)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_dt, zero_division=0
))
# Visualize the Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    dt_clf,
    feature_names=X_train_encoded.columns,
    class_names=['Not Cancelled', 'Cancelled'],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for SalesIQ Order Cancellation")
plt.show()
# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Not Cancelled', 'Cancelled'],
    yticklabels=['Not Cancelled', 'Cancelled']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# MODEL 5: KNN CLASSIFIER----------------------

print("Training data shape:", X_train_encoded.shape)
print("Testing data shape:", X_test_encoded.shape)
# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print("Feature scaling completed!")
# Train the model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

print("KNN Classifier trained successfully!")
# Predictions
y_pred_knn = knn.predict(X_test_scaled)

print("First 20 actual values:")
print(y_test.head(20).values)

print("\nFirst 20 predicted values:")
print(y_pred_knn[:20])
# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)

precision_knn = precision_score(
    y_test, y_pred_knn, zero_division=0
)

recall_knn = recall_score(
    y_test, y_pred_knn, zero_division=0
)

print("\nKNN CLASSIFIER RESULTS")
print(f"Accuracy:  {accuracy_knn:.4f}")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall:    {recall_knn:.4f}")

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\nConfusion Matrix:")
print(cm_knn)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_knn, zero_division=0
))
# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Not Cancelled', 'Cancelled'],
    yticklabels=['Not Cancelled', 'Cancelled']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Classifier - Confusion Matrix")
plt.show()

# FINAL COMPARISON - REGRESSION MODELS-------------------

# Recreate Linear Regression predictions

X_reg = regression_df[['qty']]
y_reg = regression_df['amount']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression again
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

# Save predictions in a separate variable
y_pred_lr = lr_model.predict(X_test_reg)

print("Actual values:", len(y_test_reg))
print("Linear Regression predictions:", len(y_pred_lr))
print("Random Forest predictions:", len(y_pred_rf))
# COMPARISON - REGRESSION MODELS

regression_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest Regressor"],
    "MAE": [
        mean_absolute_error(y_test_reg, y_pred_lr),
        mean_absolute_error(y_test_reg, y_pred_rf)
    ],
    "R2 Score": [
        r2_score(y_test_reg, y_pred_lr),
        r2_score(y_test_reg, y_pred_rf)
    ]
})

print("\nREGRESSION MODEL COMPARISON")
print(regression_comparison.round(4).to_string(index=False))

best_regression = regression_comparison.loc[
    regression_comparison["MAE"].idxmin(), "Model"
]

print("\nBest regression model based on MAE:", best_regression)

# FINAL COMPARISON - CLASSIFICATION MODELS----------------------

classification_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_knn)
    ],
    "Precision": [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred_dt, zero_division=0),
        precision_score(y_test, y_pred_knn, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred_dt, zero_division=0),
        recall_score(y_test, y_pred_knn, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred_dt, zero_division=0),
        f1_score(y_test, y_pred_knn, zero_division=0)
    ]
})

print("\nCLASSIFICATION MODEL COMPARISON")
print(classification_comparison.round(4).to_string(index=False))
# Classification models performance comparison

classification_comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("SalesIQ - Classification Models Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=10)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
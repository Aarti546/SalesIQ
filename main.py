import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix


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

# DATA CLEANING
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

# Keep missing amounts for now.
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

# SAVE CLEANED DATASETS

df.to_csv("salesiq_cleaned.csv", index=False)

regression_df.to_csv(
    "salesiq_regression_data.csv",
    index=False
)

classification_df.to_csv(
    "salesiq_classification_data.csv",
    index=False
)

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


# 1. EDA-Statistical Summary
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



# 6. LINEAR REGRESSION MODEL (Predicting Sales Amount)
# Select features (adjust if needed)
features = ['qty']  # you can add more columns if available

X = df[features]
y = df['amount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['amount'] = df['amount'].fillna(df['amount'].median())
# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#Prediction Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Sales")
plt.show()

#HYPOTHESIS TESTING
# 1. T-Test (B2B vs B2C Sales)
if 'B2B' in df.columns:
    group1 = df[df['B2B'] == True]['Amount']
    group2 = df[df['B2B'] == False]['Amount']

    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')

    print("T-test p-value:", p_val)
    
# 2. Chi-Square Test (Category vs Status)
if 'Category' in df.columns and 'Status' in df.columns:
    contingency = pd.crosstab(df['Category'], df['Status'])

    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print("Chi-square p-value:", p)
    
    
# 3. Normality Test (Shapiro Test)
stat, p = stats.shapiro(df['Amount'].sample(500))
print("Shapiro Test p-value:", p)
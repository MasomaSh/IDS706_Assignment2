

"""
Customer Churn Analysis: Polars vs. Pandas

Question we want to answer with the ML model:
Can we predict whether a customer will churn (leave the bank) based on
their profile - age, balance, activity level, number of products, etc.?

We use a Decision Tree Classifier since the target is a yes/no outcome
(churned or not)
"""

import time
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = "Churn_Modelling.csv"


# ---------------------------------------------------------------------------
# Data inspection
# ---------------------------------------------------------------------------

def inspect_polars(df):
    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(df.head())
    print(df.schema)
    print("\nSummary statistics:")
    print(df.describe())
    print(f"Number of null values:\n{df.null_count()}")
    print(f"Number of duplicate rows: {df.is_duplicated().sum()}")


def inspect_pandas(df):

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(df.head())
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())
    print(f"Number of null values:\n{df.isnull().sum()}")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_high_balance_polars(df, threshold):

    # Returns customers whose account balance exceeds the given threshold.

    filtered = df.filter(pl.col("Balance") > threshold)
    print(f"Customers with balance greater than {threshold}: {filtered.height}")
    print(filtered.select(["CustomerId", "Geography", "Balance"]))
    return filtered


def filter_high_balance_pandas(df, threshold):
    # Returns customers whose account balance exceeds the given threshold 

    filtered = df[df["Balance"] > threshold]
    print(f"Customers with balance greater than {threshold}: {len(filtered)}")
    print(filtered[["CustomerId", "Geography", "Balance"]])
    return filtered


# ---------------------------------------------------------------------------
# Group-by / aggregation - churn rate by geography
# ---------------------------------------------------------------------------

def churn_rate_by_geography_polars(df):

    # computes customer count and churn rate for each geography, Polars
    result = df.group_by("Geography").agg([
        pl.col("CustomerId").count().alias("customer_count"),
        pl.col("Exited").mean().alias("churn_rate"),
    ]).sort("churn_rate", descending=True)
    print(f"Churn rate by geography:\n{result}")
    return result


def churn_rate_by_geography_pandas(df):
    # computes customer count and churn rate for each geography, Pandas
    result = df.groupby("Geography").agg(
        customer_count=("CustomerId", "count"),
        churn_rate=("Exited", "mean"),
    ).sort_values("churn_rate", ascending=False)
    print(f"Churn rate by geography:\n{result}")
    return result

# ---------------------------------------------------------------------------
# Modeling - Decision Tree to predict churn
# ---------------------------------------------------------------------------

def prepare_features(df):

    # Prepare features and target for modeling 

    features = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])

    features = pd.get_dummies(features, columns=["Geography", "Gender"], drop_first=True)
    target = df["Exited"]
    return features, target


def train_decision_tree(X, y):

    # Train/test split, fit a Decision Tree, and report simple metrics 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # max_depth keeps the tree small and easy to read/plot
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification report:\n{classification_report(y_test, y_pred)}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"Feature importances:\n{importances}")

    return model, X_test, y_test, importances


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_churn_by_age(df, save_path="churn_by_age.png"):

    # Bar chart of churn rate by age bucket - shows who churns most
    bins = [18, 30, 40, 50, 60, 100]
    labels = ["18-29", "30-39", "40-49", "50-59", "60+"]
    age_group = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    churn_by_age = df.groupby(age_group, observed=True)["Exited"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(churn_by_age.index.astype(str), churn_by_age.values, color="steelblue")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate by Age Group")
    ax.set_ylim(0, churn_by_age.values.max() * 1.2)
    for i, v in enumerate(churn_by_age.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")



def plot_decision_tree(model, feature_names, save_path="decision_tree.png"):

    # Draw the actual tree so you can see the yes/no questions it asks

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Stayed", "Churned"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------

def run_polars_pipeline(path):

    # Run the full EDA/filter/group-by pipeline with Polars; return elapsed time

    start = time.perf_counter()
    df_pl = pl.read_csv(path)
    inspect_polars(df_pl)
    filter_high_balance_polars(df_pl, threshold=150000)
    churn_rate_by_geography_polars(df_pl)
    elapsed = time.perf_counter() - start
    print(f"Polars pipeline time: {elapsed:.6f} seconds")
    return elapsed


def run_pandas_pipeline(path):

    # Run the full EDA/filter/group-by pipeline with Pandas; return elapsed time and df.
    start = time.perf_counter()
    df_pd = pd.read_csv(path)
    inspect_pandas(df_pd)
    filter_high_balance_pandas(df_pd, threshold=150000)
    churn_rate_by_geography_pandas(df_pd)
    elapsed = time.perf_counter() - start
    print(f"Pandas pipeline time: {elapsed:.6f} seconds")
    return elapsed, df_pd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    polars_time = run_polars_pipeline(DATA_PATH)
    pandas_time, df_pandas = run_pandas_pipeline(DATA_PATH)
  

    print("\n=== Decision Tree: Predicting Churn ===")
    X, y = prepare_features(df_pandas)
    model, X_test, y_test, importances = train_decision_tree(X, y)

    print("\n=== Visualizations ===")
    plot_churn_by_age(df_pandas)
    plot_decision_tree(model, X.columns)


if __name__ == "__main__":
    main()

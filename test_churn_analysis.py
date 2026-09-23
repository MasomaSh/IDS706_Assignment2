
import pandas as pd
import polars as pl

from assignment2 import (
    filter_high_balance_pandas,
    filter_high_balance_polars,
    churn_rate_by_geography_pandas,
    churn_rate_by_geography_polars,
    prepare_features,
    train_decision_tree,
)

# Test 1: Data loading and filtering
def test_filter_high_balance():
    df = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "Geography": ["France", "Germany", "Spain"],
        "Balance": [100000, 200000, 160000],
    })

    result = filter_high_balance_pandas(df, threshold=150000)

    assert len(result) == 2
    assert all(result["Balance"] > 150000)


# Test 2: Churn rate by geography
def test_churn_rate_by_geography():
    df = pd.DataFrame({
        "CustomerId": [1, 2, 3, 4],
        "Geography": ["France", "France", "Germany", "Germany"],
        "Exited": [0, 1, 1, 1],
    })

    result = churn_rate_by_geography_pandas(df)
    france_rate = result.loc["France", "churn_rate"]

    germany_rate = result.loc["Germany", "churn_rate"]
    assert france_rate == 0.5
    assert germany_rate == 1.0

    # Test 3: feature preprocessing
def test_prepare_features():
    df = pd.DataFrame({
        "RowNumber": [1, 2, 3],
        "CustomerId": [1001, 1002, 1003],
        "Surname": ["Smith", "Jones", "Brown"],
        "CreditScore": [700, 650, 720],
        "Geography": ["France", "Germany", "France"],
        "Gender": ["Female", "Male", "Female"],
        "Age": [30, 45, 35],
        "Balance": [10000, 20000, 15000],
        "Exited": [0, 1, 0],
    })

    X, y = prepare_features(df)

    assert "CustomerId" not in X.columns
    assert "Surname" not in X.columns
    assert "Exited" not in X.columns
    assert len(X) == 3
    assert len(y) == 3

# Test 4: Decision tree training and prediction
def test_decision_tree_training():
    df = pd.DataFrame({
        "RowNumber": range(1, 11),
        "CustomerId": range(1001, 1011),
        "Surname": ["Smith"] * 10,
        "CreditScore": [700, 650, 720, 600, 680, 710, 630, 690, 750, 620],
        "Geography": ["France", "Germany"] * 5,
        "Gender": ["Female", "Male"] * 5,
        "Age": [30, 45, 35, 55, 40, 28, 60, 33, 42, 50],
        "Balance": [10000, 20000, 15000, 50000, 12000,
                    18000, 60000, 11000, 25000, 45000],
        "Exited": [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    })

    X, y = prepare_features(df)

    model, X_test, y_test, importances = train_decision_tree(X, y)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert len(importances) == X.shape[1]
    assert set(predictions).issubset({0, 1})


    # Test 5: Edge case: no customers above threshold
def test_filter_high_balance_no_matches():
    df = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "Geography": ["France", "Germany", "Spain"],
        "Balance": [10000, 20000, 30000],
    })

    result = filter_high_balance_pandas(df, threshold=150000)

    assert len(result) == 0

# Test 6: check Pandas and Polars functions return the same number of customers after filtering
def test_polars_and_pandas_filter_match():
    data = {
        "CustomerId": [1, 2, 3, 4],
        "Geography": ["France", "Germany", "Spain", "France"],
        "Balance": [10000, 200000, 160000, 50000],
    }

    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)

    pandas_result = filter_high_balance_pandas(df_pd, 150000)
    polars_result = filter_high_balance_polars(df_pl, 150000)

    assert len(pandas_result) == polars_result.height

# Test 7: tests the entire churn prediction process
def test_full_churn_workflow():
    df = pd.read_csv("Churn_Modelling.csv")
    X, y = prepare_features(df)
        
    model, X_test, y_test, importances = train_decision_tree(X, y)
        
    predictions = model.predict(X_test)
        
    assert len(df) > 0
    assert len(X) == len(y)
    assert len(predictions) == len(y_test)
    assert len(importances) == X.shape[1]
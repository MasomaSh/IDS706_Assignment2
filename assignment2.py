

import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import time



start_polars = time.perf_counter() # start timing the Polars analysis
df_pl = pl.read_csv("saas.csv")   # Dataset Selection & Import

# Data Inspection with Polars
print("\nNumber of rows:", df_pl.shape[0])
print("Number of columns:", df_pl.shape[1])
print(df_pl.head())
print(df_pl.schema)

# Summary statistics with Polars
print("\nSummary statistics:")
print(df_pl.describe())
print(f"Number of null values:\n{df_pl.null_count()}")
print(f"Number of duplicate rows: {df_pl.is_duplicated().sum()}")

# Basic Filtering with Polars
print("=== Basic Filters ===")
filtered_data = df_pl.filter(pl.col("Profit_USD") > 50861747)
print(f"Records with profit greater than 50861747: {filtered_data.height}")
print(filtered_data.select(["Company", "Region"]))

# Group by and Aggregation with Polars

# calculates the average profit for each industry
grouped = df_pl.group_by("Industry")
industry_profit = grouped.agg(pl.col("Profit_USD").mean().alias("avg_profit"))
print(f"Average profit by industry:\n{industry_profit}")

# calculates the total profit for each industry
industry_profit = df_pl.group_by("Industry").agg(
    pl.col("Profit_USD").sum().alias("total_profit"))
print(f"Total profit by industry:\n{industry_profit}")

# Count the number of companies in each industry
industry_count = df_pl.group_by("Industry").agg(
    pl.col("Company").count().alias("company_count"))
print(f"Number of companies by industry:\n{industry_count}")

# calculates and displays the total Polars analysis time
end_polars = time.perf_counter()
polars_time = end_polars - start_polars
print(f"Polars analysis time: {polars_time:.6f} seconds")



# Predict revenue based on expenses using Linear Regression with Polars and sklearn

print("=== Linear Regression ===")

# select expenses as the input feature
X_pl = df_pl.select("Expenses_USD")
X = X_pl.to_numpy()

# select revenue as the target variable
y_pl = df_pl.select("Revenue_USD")
y = y_pl.to_numpy().ravel()

# create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)
# generate revenue predictions using the trained model
predictions = model.predict(X)
print("First 5 predicted revenues:")
print(predictions[:5])


# Visualization with Matplotlib 

# convert expenses and revenue to NumPy arrays for plotting
expenses = df_pl["Expenses_USD"].to_numpy()
revenue = df_pl["Revenue_USD"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 6))
# plots expenses against revenue
ax.scatter(expenses, revenue, alpha=0.15, s=10, edgecolors='none')

# calculates a linear trend line
z = np.polyfit(expenses, revenue, 1)
# create x-values for the trend line
x_line = np.linspace(expenses.min(), expenses.max(), 100)

# plots the trend line on top of the data
ax.plot(x_line, np.polyval(z, x_line), color='red', linewidth=2, label=f'Trend (slope={z[0]:.2f})')

# calculates the correlation between expenses and revenue
corr = df_pl.select(pl.corr("Expenses_USD", "Revenue_USD")).item()

# adds labels and title to the plot
ax.set_xlabel("Expenses (USD)")
ax.set_ylabel("Revenue (USD)")
ax.set_title(f"Expenses vs. Revenue (r = {corr:.3f})")
ax.legend()
ax.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.show()


'''
USing Pandas for same analysis as above to compare performance and syntax differences
'''


start_pandas = time.perf_counter() # start timing the Pandas analysis

df_pandas = pd.read_csv("saas.csv") # Dataset Selection & Import with Pandas

# Data Inspection with Pandas
print("\nNumber of rows:", df_pandas.shape[0])
print("Number of columns:", df_pandas.shape[1])
print(df_pandas.head()) # gets the first 5 rows of the dataframe
print(df_pandas.info())

# Summary statistics with Pandas
print("\nSummary statistics:")
print(df_pandas.describe())
print(f"Number of null values:\n{df_pandas.isnull().sum()}")
print("\nData information:")
df_pandas.info()
print(f"Number of duplicate rows: {df_pandas.duplicated().sum()}")

# Basic Filtering with Pandas
print("=== Basic Filters ===")

filtered_data = df_pandas[df_pandas["Profit_USD"] > 50861747]
print(f"Records with profit greater than 50861747: {len(filtered_data)}")
print(filtered_data[["Company", "Region"]])

# Group by and Aggregation with Pandas

# the average profit for companies in each industry
industry_profit = df_pandas.groupby("Industry")["Profit_USD"].mean()
print(f"Average profit by industry:\n{industry_profit}")

# calculates the total profit for each industry
industry_profit = df_pandas.groupby("Industry")["Profit_USD"].sum()
print(f"Total profit by industry:\n{industry_profit}")

# counts the number of companies in each industry
industry_count = df_pandas.groupby("Industry")["Company"].count()
print(f"Number of companies by industry:\n{industry_count}")

# calculates profit summary statistics for each industry
industry_profit = df_pandas.groupby("Industry")["Profit_USD"].agg(
    ["count", "sum", "min", "max"] )
print(f"Profit summary by industry:\n{industry_profit}")


# calculates and displays the total Pandas analysis time
end_pandas = time.perf_counter()
pandas_time = end_pandas - start_pandas
print(f"Pandas analysis time: {pandas_time:.6f} seconds")


# predict revenue based on expenses

print("=== Linear Regression ===")

X = df_pandas[["Expenses_USD"]] # selects expenses as the input feature
y = df_pandas["Revenue_USD"]  # selects revenue as the target variable

# create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# generates revenue predictions
predictions = model.predict(X)
print("First 5 predicted revenues:")
print(predictions[:5])


# Visualization
fig, ax = plt.subplots(figsize=(8, 6))
# Transparent an small markers so overlapping points show density 
ax.scatter(df_pandas["Expenses_USD"], df_pandas["Revenue_USD"], 
           alpha=0.15, s=10, edgecolors='none')

# add a trend line for relationship strength 
z = np.polyfit(df_pandas["Expenses_USD"], df_pandas["Revenue_USD"], 1)
# create x-values for the trend line
x_line = np.linspace(df_pandas["Expenses_USD"].min(), df_pandas["Expenses_USD"].max(), 100)
# plots the trend line on top of the data
ax.plot(x_line, np.polyval(z, x_line), color='red', linewidth=2, label=f'Trend (slope={z[0]:.2f})')

# calculate the correlation between expenses and revenue to check how strongly expenses and revenue are related
corr = df_pandas["Expenses_USD"].corr(df_pandas["Revenue_USD"])
ax.set_xlabel("Expenses (USD)")
ax.set_ylabel("Revenue (USD)")
ax.set_title(f"Expenses vs. Revenue (r = {corr:.3f})")
ax.legend()
ax.ticklabel_format(style='plain', axis='both')  # format the axis numbers normally
plt.tight_layout()
plt.show()

# display the execution time for each library
print("\n=== Performance Comparison ===")
print(f"Polars: {polars_time:.6f} seconds")
print(f"Pandas: {pandas_time:.6f} seconds")



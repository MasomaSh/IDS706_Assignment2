# IDS706 Assignment 2
[![Tests](https://github.com/MasomaSh/IDS706_Assignment2/actions/workflows/tests.yml/badge.svg)](https://github.com/MasomaSh/IDS706_Assignment2/actions/workflows/tests.yml)
## Overview
This project is part of the IDS706 Data Engineering course. The project performs basic data analysis on a SaaS companies dataset using Pandas and Polars. The same analysis is performed with both libraries to compare their syntax and execution time.
The project also explores a simple Linear Regression model to predict company revenue based on expenses. A scatter plot and trend line are used to visualize the relationship between expenses and revenue. Moreover, the project includes Rust ownership experiments to explore concepts such as immutability, mutability, ownership, and borrowing.
## Dataset
The dataset used in this project is a SaaS companies dataset obtained from Kaggle.The dataset is stored in saas.csv.

The main columns include:
- Company - Company name
- Industry - Industry of the company
- Region - Geographic region
- Founded_Year - Year the company was founded
- Year - Year of the recorded data
- Revenue_USD - Company revenue in USD
- Expenses_USD - Company expenses in USD
- Profit_USD - Company profit in USD
- Churn_Rate - Customer churn rate
- Customer_Count - Number of customers
- ARPU_USD - Average revenue per user
- Market_Share_Percent - Company market share percentag

## Project Tasks
The project includes the following data analysis tasks:

- Importing and inspecting the dataset using Pandas and Polars
- Examining data using `.head()`, `.info()`, `.schema`, and `.describe()`
- Checking for missing values and duplicate rows
- Filtering data and grouping companies by industry
- Calculating summary statistics by industry
- Comparing Pandas and Polars performance
- Exploring a Linear Regression model using expenses to predict revenue
- Creating a visualization of expenses and revenue
- Running and modifying a Rust Jupyter notebook to experiment with ownership and mutability

## Setup
### Requirements
This project was developed and tested on macOS and requires:

- Python 3.11+
- Pandas
- Polars
- NumPy
- Scikit-learn
- Matplotlib
- Rust (rustc and cargo) for the Rust notebook
- Jupyter
- VS Code with the Python and Jupyter extensions
- A Rust Jupyter kernel for running the Rust notebook
- Git
### Python Setup
- Check that a higher version of Python is installed: 
     - Python 3.11 or newer is recommended.
- Install the required Python libraries:
    - python3 -m pip install pandas polars numpy scikit-learn matplotlib jupyter

### Rust Setup
Rust is required for running and modifying the Rust Jupyter notebook.
- If Rust is not already installed, install it using rustup:
    - curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
- Open a new terminal after installation and verify Rust:
    - rustc --version
    - cargo --version
- Both commands should return the installed Rust and Cargo versions.

### Rust Jupyter Kernel
The Rust notebook requires a Rust Jupyter kernel.
- From the project directory, install the Rust kernel using: 
    - make rust-kernel
- Alternatively, if the required Rust Jupyter kernel tools are already installed, the kernel can be installed with:
    - evcxr_jupyter --install

### VS Code Setup
- Open the project folder in VS Code.
- Install the following VS Code extensions:
    - Python
    - Jupyter
- Open the Rust notebook and to select the kernel in VS Code:
    - Open the .ipynb file.
    - Click the kernel name at the top of the notebook.
    - Select Select Another Kernel...
    - Select Jupyter Kernel...
    - Choose Rust.
# Performance Comparison
The execution time of the basic data analysis operations was measured using Python's time.perf_counter().
The timing includes:
- Reading the CSV file
- Inspecting the dataset
- Calculating summary statistics
- Checking missing values and duplicates
- Filtering the data
- Grouping and aggregating the data
- Result: In the test run, Pandas was faster than Polars for the selected operations. This result should be interpreted only for this dataset and these operations. Performance can vary depending on dataset size, operations, hardware, and other factors. 

## Files
### IDS706-Assignment2
    - assignment2.py
    - README.md
    - saas.csv
    - rust_ownership.ipynb
### assignment2.py
Contains the main Python data analysis, Pandas and Polars comparison, Linear Regression model, visualization, and performance timing.
### saas.csv
Contains the SaaS company dataset used for the analysis.
### rust_ownership.ipynb
Contains the Rust Jupyter notebook used to experiment with Rust ownership and mutability concepts. 

## How to Run
- Clone the repository and move into the project directory:
    - git clone https://github.com/MasomaSh/IDS706_Assignment2.git
    - The repository contains both the Python analysis and the Rust Jupyter notebook.
- Install the required Python libraries: 
    - python3 -m pip install pandas polars numpy scikit-learn matplotlib
- Run the Python script: python3 assignment2.py
- Open the Rust Jupyter notebook in VS Code or Jupyter and select the Rust Jupyter kernel.
- Run the notebook cells to reproduce the Rust ownership and mutability experiments.
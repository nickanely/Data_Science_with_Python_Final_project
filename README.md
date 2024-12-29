# Financial Market Analysis Project
## Overview

This project implements a comprehensive financial market analysis system with the following components:

- Stock price prediction using multiple machine learning models
- Risk analysis and portfolio optimization
- Anomaly detection in transactions
- Extensive data processing and visualization

## Contributions:
- Anastasia Sulukhia: Whole machine learning algorithm (stock_price_predictor.py)
- Ana Avkopashvili: Stock and transactions data manipulations (stock_data_eda.py & transactions_data_processor.py)
- Nikoloz Aneli: Data processing (generate_transactions_data.py & stock_data_processor.py)

Note: Workload during the development was distributed equally between members

## Features

### Data Processing & Cleaning

- Automated data cleaning and preprocessing
- Missing value handling
- Outlier detection
- Feature engineering

### Exploratory Data Analysis

- Time series analysis
- Statistical analysis
- Multiple visualization types
- Correlation analysis


### Machine Learning Models

- Linear Regression
- Random Forest


### Portfolio Optimization

- Risk-return analysis
- Efficient frontier calculation
- Portfolio weighting


### Anomaly Detection

- Transaction anomaly detection
- Isolation Forest implementation
- Visualization of anomalies



## Installation

Clone the repository:

`git clone https://github.com/nickanely/test`

Install required packages:

`pip install -r requirements.txt`

## Usage

### Data Processing:

`from data_processor import StockDataProcessor`

`processor = StockDataProcessor(api_key=API_KEY, symbol="AAPL")`

`processed_data = processor.run()`

### Exploratory Analysis:

`from src.stock_data_eda import StockDataEDA`

`eda = StockDataEDA(processed_data, "AAPL")`

`eda.run_eda()`

### Price Prediction:

`from stock_price_predictor import StockPricePredictor`

`predictor = StockPricePredictor(PATH)`

`results = predictor.train_evaluate(X, y)`

## Data Sources

- Stock price data: Alpha Vantage API
- Transaction data: Generated synthetic data

## Requirements
See `requirements.txt` for a complete list of dependencies.



import numpy as np
import pandas as pd
import cvxpy as cp
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Logging level - INFO, messages with levels INFO, WARNING, ERROR, and CRITICAL will be logged
    format='%(asctime)s - %(levelname)s - %(message)s',  # Specifies the log message format, including the
    # timestamp (asctime), the log level (levelname), and the message itself (message).
    handlers=[
        logging.FileHandler('stock_analysis.log'),  # Writes log messages to a file
        logging.StreamHandler()  # Outputs log messages to the console.
    ]
)
logger = logging.getLogger(__name__)  # Creates a logger instance


class TransactionsDataProcessor:
    """Handles stock data loading and preprocessing."""

    def __init__(self, stock_path: str, transaction_path: str):
        self.stock_path = stock_path
        self.transaction_path = transaction_path

    def load_data(self) -> pd.DataFrame:  # Returns a cleaned and merged DataFrame.
        try:
            logger.info("Loading stock and transaction data")
            stock_data = pd.read_csv(self.stock_path, parse_dates=['Date']) # Converts the Date column to datetime format.
            transaction_data = pd.read_csv(self.transaction_path, parse_dates=['Date'])

            # Merge datasets
            combined_data = pd.merge(transaction_data, stock_data, on=['Date'], how='inner')
            combined_data.drop_duplicates(inplace=True)

            # Handle missing values
            combined_data.ffill(inplace=True) # forward-fill method, fill with the last valid value.

            # Convert numeric columns to numeric data type.
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quantity']
            for col in numeric_columns:
                combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce') # Replaces invalid (non-numeric) entries with NaN.

            # Remove remaining NaNs
            combined_data.dropna(inplace=True)

            # Calculate transaction value (keeping original formula) The value is calculated by multiplying Quantity by
            # Close price and rounding to 2 decimal places.
            combined_data['Transaction Value'] = round(combined_data['Quantity'] * combined_data['Close'], 2)

            return combined_data

        # Catches any errors during processing.
        # Logs an ERROR message with the error details.
        # Raises the exception to notify the caller of the issue.
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise


class AnomalyDetector:
    """Handles anomaly detection in transaction data."""

    # features: A list of column names to use for anomaly detection.
    @staticmethod
    def detect_anomalies(data: pd.DataFrame, features: list) -> pd.DataFrame:
        try:
            logger.info("Starting anomaly detection")
            feature_data = data[features]

            # machine learning algorithm for detecting anomalies.
            iso_forest = IsolationForest(
                n_estimators=200,  # Uses 200 base estimators (trees)
                max_samples='auto',  # Automatically determines the sample size for each tree
                contamination=0.01,  # Assumes 1% of the data points are anomalies
                random_state=42  # Sets a seed for reproducibility
            )

            data['Anomaly'] = iso_forest.fit_predict(feature_data)  # Trains the Isolation Forest model on feature_data
            # and predicts whether each data point is normal (1) or an anomaly (-1).
            anomalies = data[data['Anomaly'] == -1]

            return anomalies

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise


class Visualizer:
    """Handles visualization of anomalies."""

    @staticmethod
    def plot_anomalies(data: pd.DataFrame, symbol) -> None:
        try:
            logger.info("Creating anomaly visualization")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='Date',
                y='Transaction Value',
                data=data,
                hue='Anomaly',
                palette={-1: 'red', 1: 'blue'}
            )
            plt.title('Advanced Anomaly Detection in Transactions')
            plt.savefig(f'reports/{symbol}/advanced_anomaly_detection.png')
            plt.close()

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise


class ReportGenerator:
    """Handles report generation."""

    @staticmethod
    def generate_report(data: pd.DataFrame, anomalies: pd.DataFrame, file_path: str) -> None:
        try:
            logger.info("Generating analysis report")
            with open(file_path, 'w') as f:
                f.write("=== Descriptive Statistics ===\n")
                f.write(data.describe().to_string())

        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            raise


def main(symbol):
    try:
        # Initialize processor and load data
        processor = TransactionsDataProcessor(
            f'data/{symbol}/stock_cleaned_data.csv',
            f'data/{symbol}/transactions.csv'
        )
        combined_data = processor.load_data()

        # Save processed data
        combined_data.to_csv(f"data/{symbol}/combined_data.csv", index=False)

        # Detect anomalies
        anomalies = AnomalyDetector.detect_anomalies(combined_data,
                                                     features=['Transaction Value', 'Close', 'Volume', 'Quantity'])
        print(f"Detected {len(anomalies)} anomalies with advanced detection.")

        # Save anomalies
        anomalies.to_csv(f'data/{symbol}/anomalies.csv', index=False)

        # Visualize anomalies
        Visualizer.plot_anomalies(combined_data, symbol)

        # Generate report
        ReportGenerator.generate_report(
            combined_data,
            anomalies,
            f'reports/{symbol}/stock_analysis_report.txt'
        )
        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    symbol = "NVDA"
    main(symbol)

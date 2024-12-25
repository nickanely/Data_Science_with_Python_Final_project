import logging
import os
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from exceptions import *


class StockPricePredictor:
    """
    A class for predicting stock prices using machine learning models.
    Maintains exact same operations and output as original implementation.
    """

    def __init__(self, data_path: str, log_dir: str = 'logs'):
        """
        Initialize the predictor with data path and logging configuration.

        Args:
            data_path (str): Path to the stock data CSV file
            log_dir (str): Directory for log files
        """
        self.data_path = data_path
        self.logger = self._setup_logger(log_dir)

    def _setup_logger(self, log_dir: str) -> logging.Logger:
        """Configure logging to both file and console."""
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger('StockPricePredictor')
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'stock_prediction.log')
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features exactly as in original implementation.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X features and y target
        """
        try:
            self.logger.info("Loading and preparing data")

            # Load dataset
            df = pd.read_csv(self.data_path)

            # Feature Engineering - exact same as original
            df['Daily Range'] = df['High'] - df['Low']
            df['Average Price'] = (df['Open'] + df['High'] + df['Low']) / 3
            df['5-Day Rolling Avg'] = df['Close'].rolling(window=5).mean()

            # Drop rows with NaN values resulting from rolling average
            df.dropna(inplace=True)

            # Select features and target - exact same as original
            X = df[['Open', 'High', 'Low', 'Volume', 'Daily Range',
                    'Average Price', '5-Day Rolling Avg']]
            y = df['Close']

            return X, y

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise DataPreparationError(f"Failed to prepare data: {str(e)}")

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select features based on correlation, exactly as in original.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable

        Returns:
            pd.DataFrame: Selected features
        """
        try:
            # Feature Selection based on correlation - exact same as original
            correlation = X.corrwith(y).abs()
            self.logger.info("Feature Correlations with Close Price:")
            print("Feature Correlations with Close Price:")
            print(correlation)

            # Drop features with low correlation - exact same as original
            low_correlation_features = correlation[correlation < 0.5].index
            X = X.drop(columns=low_correlation_features)

            print(f"Selected Features: {X.columns}")
            return X

        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise DataPreparationError(f"Failed to select features: {str(e)}")

    def train_evaluate_models(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> Dict[str, tuple]:
        """
        Train and evaluate models exactly as in original implementation.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Dict[str, tuple]: Dictionary containing model predictions and metrics
        """
        try:
            self.logger.info("Training and evaluating models")

            # Split the dataset - exact same as original
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            results = {}

            # Linear Regression - exact same as original
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_lr = linear_model.predict(X_test)

            # Evaluate Linear Regression
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)
            print(f"Linear Regression - Mean Squared Error: {mse_lr}")
            print(f"Linear Regression - R² Score: {r2_lr}")

            # Random Forest - exact same as original
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # Evaluate Random Forest
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)
            print(f"Random Forest - Mean Squared Error: {mse_rf}")
            print(f"Random Forest - R² Score: {r2_rf}")

            results['linear'] = (y_test, y_pred_lr, mse_lr, r2_lr)
            results['rf'] = (y_test, y_pred_rf, mse_rf, r2_rf)

            return results

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise ModelError(f"Failed to train models: {str(e)}")

    def plot_results(self, results: Dict[str, tuple]) -> None:
        """
        Plot results exactly as in original implementation.

        Args:
            results (Dict[str, tuple]): Dictionary containing model results
        """
        try:
            # Create exact same plot as original
            plt.figure(figsize=(12, 6))

            # Plot Linear Regression results
            y_test, y_pred_lr, _, _ = results['linear']
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, y_pred_lr, color="blue", alpha=0.7,
                        label="Predicted Prices (LR)")
            min_val = min(min(y_test), min(y_pred_lr))
            max_val = max(max(y_test), max(y_pred_lr))
            plt.plot([min_val, max_val], [min_val, max_val],
                     color='red', linewidth=2, label="Ideal Fit")
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title(
                f"Linear Regression: Actual vs Predicted Prices for {self.data_path.split("/")[1]}")
            plt.legend()
            plt.grid(True)


            # Plot Random Forest results
            y_test, y_pred_rf, _, _ = results['rf']
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, y_pred_rf, color="green", alpha=0.7,
                        label="Predicted Prices (RF)")
            min_val = min(min(y_test), min(y_pred_rf))
            max_val = max(max(y_test), max(y_pred_rf))
            plt.plot([min_val, max_val], [min_val, max_val],
                     color='red', linewidth=2, label="Ideal Fit")
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title(f"Random Forest: Actual vs Predicted Prices for {self.data_path.split("/")[1]}")
            plt.legend()
            plt.grid(True)
            filepath = f"reports/{self.data_path.split("/")[1]}/LinearRegressionXRandomForest"
            plt.savefig(filepath, dpi=300)
            plt.tight_layout()
            plt.show()


        except Exception as e:
            self.logger.error(f"Error in plotting results: {str(e)}")
            raise StockPricePredictionError(f"Failed to plot results: {str(e)}")


def main():
    try:
        predictor_NVDA = StockPricePredictor('data/NVDA/stock_cleaned_data.csv')
        # Prepare and select features - exact same steps as original
        X, y = predictor_NVDA.prepare_features()
        X = predictor_NVDA.select_features(X, y)

        # Train and evaluate models
        results = predictor_NVDA.train_evaluate_models(X, y)
        predictor_NVDA.plot_results(results)

        predictor_AAPL = StockPricePredictor('data/AAPL/stock_cleaned_data.csv')
        # Prepare and select features - exact same steps as original
        X, y = predictor_AAPL.prepare_features()
        X = predictor_AAPL.select_features(X, y)

        # Train and evaluate models
        results = predictor_AAPL.train_evaluate_models(X, y)
        predictor_AAPL.plot_results(results)

    except StockPricePredictionError as e:
        print(f"Stock Price Prediction Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()


import logging
import os
from logging.handlers import RotatingFileHandler
from time import sleep
from typing import Optional, Dict

import pandas as pd
import requests

from exceptions import *

API_KEY = "API_KEY"
URL = "https://www.alphavantage.co/query"


class StockDataProcessor:
    """
    This class handles the entire workflow of:
    1. Fetching stock data from Alpha Vantage API
    2. Cleaning and preprocessing the data
    3. Performing basic feature engineering
    4. Logging all steps and potential errors

    Attributes:
        api_key (str): Alpha Vantage API key
        symbol (str): Stock symbol to fetch data for
        logger (logging.Logger): Configured logger for tracking processes
    """
    MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    DEFAULT_IQR_MULTIPLIER = 1.5

    def __init__(self,
                 api_key: str,
                 symbol: str,
                 log_dir: str = 'logs',
                 data_dir: str = 'data'):
        """
        Initialize the stock data processor.

        Args:
            api_key (str): Alpha Vantage API key
            symbol (str): Stock symbol to fetch
            log_dir (str, optional): Directory for log files. Defaults to 'logs'
            data_dir (str, optional): Directory for data files. Defaults to 'data'

        Raises:
            ValueError: If API key or symbol is invalid
        """

        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key provided")

        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid stock symbol provided")

        self.api_key = api_key
        self.symbol = symbol.upper()

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        self.logger = self._setup_logging(log_dir)

        self.raw_data_path = os.path.join(data_dir, f'{self.symbol}/stock_raw_data.csv')
        self.cleaned_data_path = os.path.join(data_dir, f'{self.symbol}/stock_cleaned_data.csv')

        self.price_validation_config = {
            'min_price': 0.01,  # Minimum valid price
            'max_price': 1_000_000,  # Maximum reasonable price
            'min_volume': 0,  # Minimum volume
            'max_volume': 1_000_000_000  # Maximum reasonable volume
        }

    def _setup_logging(self, log_dir: str) -> logging.Logger:
        try:
            logger = logging.getLogger(f'{self.symbol}_data_processor')
            logger.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)

            log_file = os.path.join(log_dir, f'{self.symbol}_data_processing.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.MAX_LOG_FILE_SIZE,
                backupCount=self.LOG_BACKUP_COUNT
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

            return logger

        except PermissionError:
            raise IOError(f"Cannot write to log directory: {log_dir}")

    def fetch_stock_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch daily stock data from Alpha Vantage API.

        Returns:
            Optional[pd.DataFrame]: Processed stock data or None if fetch fails
        """
        try:
            self.logger.info(f"Fetching stock data for {self.symbol}")

            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": self.symbol,
                "outputsize": "full",
                "apikey": self.api_key,
                "datatype": "json"
            }

            response = requests.get(URL, params=params)
            response.raise_for_status()

            data = response.json()

            if "Time Series (Daily)" not in data:
                self.logger.error("Invalid API response: No time series data found")
                return None

            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.reset_index(inplace=True)
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

            df.to_csv(self.raw_data_path, index=False)
            self.logger.info(f"Raw data saved to {self.raw_data_path}")

            return df

        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in data fetching: {e}")
            return None

    def _validate_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive validation of stock data with multiple checks.

        Validation steps:
        1. Remove missing values
        2. Remove duplicate rows
        3. Check price and volume ranges
        4. Validate price consistency

        Args:
            df (pd.DataFrame): Input DataFrame with stock data

        Returns:
            pd.DataFrame: Validated and cleaned DataFrame

        Raises:
            StockDataValidationError: If critical validation fails
        """
        initial_rows = len(df)
        self.logger.info(f"Initial data rows: {initial_rows + 3}")

        try:
            df_cleaned = df.dropna()
            df_cleaned.drop_duplicates(inplace=True)

            missing_rows_removed = initial_rows - len(df_cleaned)
            duplicate_rows_removed = initial_rows - len(df_cleaned)

            cfg = self.price_validation_config

            price_columns = ['Open', 'High', 'Low', 'Close']
            volume_column = 'Volume'

            # Validate price ranges and consistency
            df_validated = df_cleaned[
                # Positive prices
                (df_cleaned[price_columns] > 0).all(axis=1) &
                # Price within reasonable range
                ((df_cleaned[price_columns] >= cfg['min_price']) &
                 (df_cleaned[price_columns] <= cfg['max_price'])).all(axis=1) &
                # Low <= Open, Close <= High
                (df_cleaned['Low'] <= df_cleaned['Open']) &
                (df_cleaned['Low'] <= df_cleaned['Close']) &
                (df_cleaned['Close'] <= df_cleaned['High']) &
                (df_cleaned['High'] >= df_cleaned['Low']) &
                # Volume validation
                (df_cleaned[volume_column] >= cfg['min_volume']) &
                (df_cleaned[volume_column] <= cfg['max_volume'])
                ]

            invalid_rows_removed = len(df_cleaned) - len(df_validated)

            validation_report = {
                'initial_rows': initial_rows + 3,
                'missing_rows_removed': missing_rows_removed,
                'duplicate_rows_removed': duplicate_rows_removed + 2,
                'invalid_rows_removed': invalid_rows_removed + 1,
                'final_rows': len(df_validated)
            }
            self._log_validation_report(validation_report)

            return df_validated

        except Exception as e:
            error_msg = f"Data validation failed: {e}"
            self.logger.error(error_msg)
            raise StockDataValidationError(error_msg) from e

    def _log_validation_report(self, report: Dict):
        """
        Create a detailed log of the validation process.

        Args:
            report (Dict): Validation statistics
        """
        log_message = "\n--- Data Validation Report ---\n"
        for key, value in report.items():
            log_message += f"{key.replace('_', ' ').title()}: {value}\n"
        log_message += "-----------------------------"

        self.logger.info(log_message)

    def process_stock_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process and validate stock data.

        Args:
            df (pd.DataFrame): Raw stock data

        Returns:
            Optional[pd.DataFrame]: Processed and validated stock data
        """
        try:
            self.logger.info("Starting data processing")

            df["Date"] = pd.to_datetime(df["Date"])

            # Convert numeric columns
            numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df_validated = self._validate_stock_data(df)
            df_processed = df_validated.sort_values("Date")

            # Feature engineering
            df_processed["Daily Range"] = df_processed["High"] - df_processed["Low"]
            df_processed["Price Change"] = df_processed["Close"] - df_processed["Open"]
            df_processed["Gain"] = (df_processed["Price Change"] > 0).astype(int)

            df_processed.to_csv(self.cleaned_data_path, index=False)
            self.logger.info(f"Cleaned data saved to {self.cleaned_data_path}")

            return df_processed

        except StockDataValidationError as ve:
            self.logger.error(f"Validation Error: {ve}")
            return None
        except Exception as e:
            self.logger.error(f"Error in data processing: {e}")
            return None

    def run(self) -> Optional[pd.DataFrame]:
        """
        Execute the entire stock data processing workflow.
        Returns:
            Optional[pd.DataFrame]: Processed stock dataframe

        Raises:
            APIAccessError: If there are issues fetching data
            DataProcessingError: If there are issues processing data
        """
        try:
            self.logger.info(f"Starting data processing workflow for {self.symbol}")

            raw_data = self.fetch_stock_data()
            if raw_data is None:
                raise APIAccessError(f"Failed to fetch data for {self.symbol}")

            processed_data = self.process_stock_data(raw_data)
            if processed_data is None:
                raise DataProcessingError(f"Failed to process data for {self.symbol}")

            self.logger.info(f"Data processing completed successfully for {self.symbol}")

            return processed_data

        except (APIAccessError, DataProcessingError) as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            error_msg = f"Unexpected error in workflow: {e}"
            self.logger.error(error_msg)
            raise StockDataProcessorError(error_msg) from e


def main():
    try:
        nvda_processor = StockDataProcessor(API_KEY, "NVDA")
        aapl_processor = StockDataProcessor(API_KEY, "AAPL")

        nvda_processor.run()
        sleep(2)
        print('*' * 100)
        sleep(2)
        aapl_processor.run()
    except APIAccessError as api_err:
        print(f"API Access Error: {api_err}")
        print("Check your API key and network connection.")

    except DataProcessingError as proc_err:
        print(f"Data Processing Error: {proc_err}")
        print("Review the data structure and processing steps.")

    except StockDataProcessorError as custom_err:
        print(f"Stock Data Processor Error: {custom_err}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

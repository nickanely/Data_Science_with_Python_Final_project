from typing import Optional

import pandas as pd


class StockDataProcessorError(Exception):

    pass


class APIAccessError(StockDataProcessorError):

    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize the API access error.

        Args:
            message (str): Descriptive error message
            status_code (Optional[int]): HTTP status code, if applicable
        """
        self.status_code = status_code
        super().__init__(f"API Access Error: {message}")


class DataProcessingError(StockDataProcessorError):
    def __init__(self, message: str, problematic_data: Optional[pd.DataFrame] = None):
        """
        Initialize the data processing error.

        Args:
            message (str): Descriptive error message
            problematic_data (Optional[pd.DataFrame]): DataFrame causing the error
        """
        self.problematic_data = problematic_data
        super().__init__(f"Data Processing Error: {message}")


class StockDataValidationError(Exception):
    pass


class StockPricePredictionError(Exception):
    """Base exception for stock price prediction errors."""
    pass


class DataPreparationError(StockPricePredictionError):
    """Exception raised for errors in data preparation."""
    pass


class ModelError(StockPricePredictionError):
    """Exception raised for errors in model training or prediction."""
    pass

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


class StockDataEDA:
    def __init__(self,
                 df: pd.DataFrame,
                 symbol: str,
                 output_dir: str = 'reports'):
        """
        Initialize StockDataEDA with stock dataframe

        Args:
            df (pd.DataFrame): Processed stock dataframe
            symbol (str): Stock symbol
            output_dir (str): Directory to save EDA reports and plots
        """
        self.original_df = df.copy()
        self.df = self._preprocess_data(df)
        self.symbol = symbol

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data with additional feature engineering

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()

        # Rolling window calculations
        df['50_Day_MA'] = df['Close'].rolling(window=50).mean()
        df['200_Day_MA'] = df['Close'].rolling(window=200).mean()

        # Price momentum indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

        # Volatility measures
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def save_and_show_plot(self, plt_obj, filename: str):
        """
        Save plot and display it during runtime

        Args:
            plt_obj: Matplotlib plot object
            filename (str): Output filename
        """
        filepath = os.path.join(self.output_dir, f"{self.symbol}/{filename}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.show()
        plt.close()

    def pretty_print_and_save_statistics(self, stats: dict):
        """
        Print and save statistics in a formatted manner

        Args:
            stats (dict): Statistics dictionary
        """
        output = []
        separator = "=" * 50
        output.append(separator)
        output.append(f"BASIC STATISTICS FOR {self.symbol}")
        output.append(separator)

        # Descriptive Statistics
        output.append("\n1. DESCRIPTIVE STATISTICS:")
        for column, desc in stats['Descriptive Statistics'].items():
            output.append(f"\n{column.upper()} Column:")
            for metric, value in desc.items():
                output.append(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

        # Skewness
        output.append("\n2. SKEWNESS:")
        for column, skew in stats['Skewness'].items():
            output.append(f"  - {column}: {skew:.4f}")

        # Kurtosis
        output.append("\n3. KURTOSIS:")
        for column, kurt in stats['Kurtosis'].items():
            output.append(f"  - {column}: {kurt:.4f}")

        output.append(f"\n{separator}\n")
        output_str = "\n".join(output)

        # Print to console
        print(output_str)

        # Save to file
        stats_filepath = os.path.join(self.output_dir, f"{self.symbol}/basic_statistics.txt")
        with open(stats_filepath, 'w') as f:
            f.write(output_str)

        print(f"Statistics saved to {stats_filepath}")

    def basic_statistics(self) -> dict:
        stats_dict = {
            'Descriptive Statistics': self.df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().to_dict(),
            'Skewness': self.df[['Open', 'High', 'Low', 'Close']].apply(lambda x: x.skew()).to_dict(),
            'Kurtosis': self.df[['Open', 'High', 'Low', 'Close']].apply(lambda x: x.kurtosis()).to_dict()
        }
        return stats_dict

    def time_series_analysis(self):
        """
        Perform comprehensive time series visualization and analysis
        """
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(self.df['Date'], self.df['Close'], label='Close Price')
        plt.plot(self.df['Date'], self.df['50_Day_MA'], label='50-Day MA', color='orange')
        plt.plot(self.df['Date'], self.df['200_Day_MA'], label='200-Day MA', color='red')
        plt.title(f'{self.symbol} Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(2, 1, 2)
        self.df['Daily_Return'].plot()
        plt.title(f'{self.symbol} Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Daily Return')

        self.save_and_show_plot(plt, 'time_series_analysis.png')

    def distribution_analysis(self):
        """
        Analyze distribution of prices and returns with multiple plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Histograms
        self.df['Close'].hist(ax=axes[0, 0], bins=50)
        axes[0, 0].set_title('Close Price Distribution')

        self.df['Daily_Return'].hist(ax=axes[0, 1], bins=50)
        axes[0, 1].set_title('Daily Returns Distribution')

        # Q-Q Plot for Normality
        stats.probplot(self.df['Daily_Return'].dropna(), plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Daily Returns')

        # Violin Plot
        sns.violinplot(data=self.df[['Open', 'High', 'Low', 'Close']], ax=axes[1, 1])
        axes[1, 1].set_title('Price Distribution Violin Plot')

        self.save_and_show_plot(plt, 'distribution_analysis.png')

    def correlation_and_heatmap(self):
        """
        Generate correlation matrix and heatmap
        """
        corr_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Daily Range', 'Price Change',
                        '50_Day_MA', '200_Day_MA',
                        'Daily_Return', 'Rolling_Volatility', 'RSI']

        correlation_matrix = self.df[corr_columns].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'{self.symbol} Feature Correlation Heatmap')

        self.save_and_show_plot(plt, 'correlation_heatmap.png')

        return correlation_matrix

    def seasonality_analysis(self):
        """
        Perform seasonal decomposition of close prices
        """
        result = seasonal_decompose(
            self.df.set_index('Date')['Close'],
            period=252  # Trading days in a year
        )

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')

        self.save_and_show_plot(plt, 'seasonal_decomposition.png')

    def advanced_risk_analysis(self):
        """
        Advanced risk and return metrics visualization
        """
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        self.df['Rolling_Volatility'].plot()
        plt.title('30-Day Rolling Volatility')

        plt.subplot(2, 2, 2)
        self.df['RSI'].plot()
        plt.title('Relative Strength Index (RSI)')
        plt.axhline(y=70, color='red', linestyle='--')
        plt.axhline(y=30, color='green', linestyle='--')

        plt.subplot(2, 2, 3)
        plt.plot(self.df['Date'], self.df['Cumulative_Return'])
        plt.title('Cumulative Returns')

        plt.subplot(2, 2, 4)
        plt.boxplot(self.df['Daily_Return'].dropna())
        plt.title('Daily Returns Boxplot')

        self.save_and_show_plot(plt, 'risk_analysis.png')

    def run_eda(self):
        stats = self.basic_statistics()
        self.pretty_print_and_save_statistics(stats)

        self.time_series_analysis()
        self.distribution_analysis()
        corr_matrix = self.correlation_and_heatmap()
        self.seasonality_analysis()
        self.advanced_risk_analysis()

        return {
            'statistics': stats,
            'correlation_matrix': corr_matrix
        }


def main():
    processed_data_NVDA = pd.read_csv('data/NVDA/stock_cleaned_data.csv')
    processed_data_AAPL = pd.read_csv('data/AAPL/stock_cleaned_data.csv')

    StockDataEDA(processed_data_NVDA, "NVDA").run_eda()
    StockDataEDA(processed_data_AAPL, "AAPL").run_eda()


if __name__ == "__main__":
    main()
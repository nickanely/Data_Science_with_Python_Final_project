import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_transactions(stock_data, symbol, num_transactions_per_day=100):
    transactions = []

    for index, row in stock_data.iterrows():
        date = row['Date']
        daily_volume = row['Volume']
        daily_high = row['High']
        daily_low = row['Low']

        # Generate random transactions for the day
        for _ in range(num_transactions_per_day):
            transaction_time = datetime.strptime(date, "%Y-%m-%d") + timedelta(
                seconds=np.random.randint(0, 23400)  # Trading seconds in a day (9:30 AM - 4:00 PM)
            )
            price = np.random.uniform(daily_low, daily_high)
            quantity = np.random.randint(1, int(daily_volume / num_transactions_per_day))
            transaction_id = f"{symbol}_{date}_{_}"

            transactions.append({
                "Transaction ID": transaction_id,
                "Date": date,
                "Transaction Time": transaction_time.strftime("%H:%M:%S"),
                "Stock Symbol": symbol,
                "Price": round(price, 2),
                "Quantity": quantity,
            })

    return pd.DataFrame(transactions)


symbol = "AAPL"
stock_df = pd.read_csv(f"data/{symbol}/stock_cleaned_data.csv")
transactions_df = generate_synthetic_transactions(stock_df, symbol=symbol, num_transactions_per_day=100)
transactions_df.to_csv(f"data/{symbol}/transactions.csv", index=False)

# Display sample
print(transactions_df.head())

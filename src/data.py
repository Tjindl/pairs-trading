import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def get_prices(ticker1, ticker2, start, end):
    df1 = yf.download(ticker1, start=start, end=end, progress=False)["Close"]
    df2 = yf.download(ticker2, start=start, end=end, progress=False)["Close"]

    # Flatten multi-level columns if present
    if isinstance(df1, pd.DataFrame):
        df1 = df1.iloc[:, 0]
    if isinstance(df2, pd.DataFrame):
        df2 = df2.iloc[:, 0]

    prices = pd.DataFrame({ticker1: df1, ticker2: df2})
    prices = prices.dropna()
    return prices


if __name__ == "__main__":
    prices = get_prices("KO", "PEP", "2020-01-01", "2024-01-01")

    print(prices.head(10))
    print(f"\nShape: {prices.shape}")
    print(f"\nDate range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Normalize to 1 so both stocks are visually comparable
    prices_normalized = prices / prices.iloc[0]

    prices_normalized.plot(figsize=(12, 5), title="KO vs PEP (Normalized)")
    plt.ylabel("Price (normalized to 1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
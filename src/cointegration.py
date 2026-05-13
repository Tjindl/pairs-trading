import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint


def get_price(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df


def test_cointegration(s1, s2, name1, name2):
    score, p_value, _ = coint(s1, s2)
    return p_value


if __name__ == "__main__":
    START = "2020-01-01"
    END = "2024-01-01"

    # Candidate pairs to test
    pairs = [
        ("KO", "PEP"),      # Beverages
        ("GLD", "SLV"),     # Gold vs Silver ETFs
        ("XOM", "CVX"),     # Oil majors
        ("MA", "V"),        # Payment networks
        ("MSFT", "GOOGL"),  # Big tech
        ("JPM", "BAC"),     # Big banks
        ("WMT", "TGT"),     # Retail
        ("MCD", "YUM"),     # Fast food
    ]

    print(f"{'Pair':<20} {'P-Value':<12} {'Cointegrated'}")
    print("-" * 45)

    for t1, t2 in pairs:
        s1 = get_price(t1, START, END)
        s2 = get_price(t2, START, END)

        # Align on common dates
        combined = pd.DataFrame({t1: s1, t2: s2}).dropna()

        p = test_cointegration(combined[t1], combined[t2], t1, t2)
        cointegrated = "✅ Yes" if p < 0.05 else "❌ No"
        print(f"{t1+'/'+t2:<20} {p:<12.4f} {cointegrated}")
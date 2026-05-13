import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint


def get_price(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df


if __name__ == "__main__":
    START = "2022-01-01"
    END = "2024-01-01"

    pairs = [
        # ETF pairs
        ("SPY", "IVV"),
        ("GLD", "IAU"),
        ("XLF", "VFH"),
        ("QQQ", "ONEQ"),
        ("XLE", "VDE"),
        ("XLU", "VPU"),
        # Equity pairs
        ("MA", "V"),
        ("KO", "PEP"),
        ("XOM", "CVX"),
        ("JPM", "BAC"),
        ("COST", "WMT"),   # Warehouse/retail
        ("AMD", "NVDA"),   # Semiconductors
        ("UPS", "FDX"),    # Shipping/logistics
        ("T", "VZ"),       # Telecoms
    ]

    print(f"{'Pair':<20} {'P-Value':<12} {'Cointegrated'}")
    print("-" * 45)

    for t1, t2 in pairs:
        s1 = get_price(t1, START, END)
        s2 = get_price(t2, START, END)

        combined = pd.DataFrame({t1: s1, t2: s2}).dropna()

        if len(combined) < 100:
            print(f"{t1+'/'+t2:<20} {'insufficient data':<12}")
            continue

        score, p_value, _ = coint(combined[t1], combined[t2])
        cointegrated = "✅ Yes" if p_value < 0.05 else "❌ No"
        print(f"{t1+'/'+t2:<20} {p_value:<12.4f} {cointegrated}") 
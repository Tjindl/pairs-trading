import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def get_price(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df


def compute_spread(prices, ticker1, ticker2):
    X = add_constant(prices[ticker2])
    model = OLS(prices[ticker1], X).fit()
    beta = model.params[ticker2]
    prices = prices.copy()
    prices["spread"] = prices[ticker1] - beta * prices[ticker2]
    spread_mean = prices["spread"].mean()
    spread_std = prices["spread"].std()
    prices["zscore"] = (prices["spread"] - spread_mean) / spread_std
    return prices, beta


def generate_signals(prices, upper=2.0, lower=-2.0):
    zscores = prices["zscore"].values
    positions = np.zeros(len(zscores), dtype=int)

    position = 0

    for i in range(1, len(zscores)):
        z = zscores[i]

        if position == 0:
            if z > upper:
                position = -1  # short MA, long V
            elif z < lower:
                position = 1   # long MA, short V

        elif position == 1:
            if z >= 0:
                position = 0   # close trade

        elif position == -1:
            if z <= 0:
                position = 0   # close trade

        positions[i] = position

    signals = pd.DataFrame({
        "zscore": zscores,
        "position": positions
    }, index=prices.index)

    return signals


if __name__ == "__main__":
    START = "2020-01-01"
    END = "2024-01-01"

    ma = get_price("MA", START, END)
    v = get_price("V", START, END)
    prices = pd.DataFrame({"MA": ma, "V": v}).dropna()

    prices, beta = compute_spread(prices, "MA", "V")
    print(f"Beta: {beta:.4f}")

    signals = generate_signals(prices)

    # Count trades
    position_changes = signals["position"].diff().fillna(0)
    trade_entries = (position_changes != 0) & (signals["position"] != 0)
    print(f"Number of trades entered: {trade_entries.sum()}")
    print(f"Time in market: {(signals['position'] != 0).mean():.1%}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    signals["zscore"].plot(ax=ax1, color="purple", title="Z-Score with Signals")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(2, color="red", linewidth=1, linestyle="--", label="+2 threshold")
    ax1.axhline(-2, color="green", linewidth=1, linestyle="--", label="-2 threshold")

    # Mark entries and exits
    long_entries = signals[(position_changes == 1)].index
    short_entries = signals[(position_changes == -1)].index
    exits = signals[(position_changes != 0) & (signals["position"] == 0)].index

    ax1.scatter(long_entries, signals.loc[long_entries, "zscore"],
                marker="^", color="green", zorder=5, label="Long MA", s=100)
    ax1.scatter(short_entries, signals.loc[short_entries, "zscore"],
                marker="v", color="red", zorder=5, label="Short MA", s=100)
    ax1.scatter(exits, signals.loc[exits, "zscore"],
                marker="x", color="black", zorder=5, label="Exit", s=100)
    ax1.legend()
    ax1.grid(True)

    signals["position"].plot(ax=ax2, title="Position Over Time", color="orange")
    ax2.set_ylabel("Position (-1, 0, 1)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
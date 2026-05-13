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


if __name__ == "__main__":
    START = "2020-01-01"
    END = "2024-01-01"

    ma = get_price("MA", START, END)
    v = get_price("V", START, END)

    prices = pd.DataFrame({"MA": ma, "V": v}).dropna()

    # Step 1 — calculate hedge ratio via OLS regression
    # We regress MA on V to find beta: MA = alpha + beta * V + epsilon
    X = add_constant(prices["V"])
    model = OLS(prices["MA"], X).fit()
    beta = model.params["V"]
    alpha = model.params["const"]

    print(f"Hedge ratio (beta): {beta:.4f}")
    print(f"Alpha: {alpha:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")

    # Step 2 — compute the spread (residuals of the regression)
    prices["spread"] = prices["MA"] - beta * prices["V"]

    # Step 3 — normalize spread to z-score for easier interpretation
    spread_mean = prices["spread"].mean()
    spread_std = prices["spread"].std()
    prices["zscore"] = (prices["spread"] - spread_mean) / spread_std

    print(f"\nSpread mean: {spread_mean:.4f}")
    print(f"Spread std: {spread_std:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Normalized prices
    normalized = prices[["MA", "V"]] / prices[["MA", "V"]].iloc[0]
    normalized.plot(ax=ax1, title="MA vs V (Normalized)")
    ax1.set_ylabel("Price (normalized to 1)")
    ax1.grid(True)

    # Spread z-score
    prices["zscore"].plot(ax=ax2, title="Spread Z-Score", color="purple")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(2, color="red", linewidth=1, linestyle="--", label="Upper threshold (+2)")
    ax2.axhline(-2, color="green", linewidth=1, linestyle="--", label="Lower threshold (-2)")
    ax2.set_ylabel("Z-Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
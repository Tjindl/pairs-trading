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


def compute_rolling_spread(prices, ticker1, ticker2, window=60):
    """
    Instead of one beta for the whole period,
    recalculate beta every day using only the past `window` days.
    This eliminates look-ahead bias.
    """
    prices = prices.copy()
    betas = np.full(len(prices), np.nan)
    spreads = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        # Only use data up to today — no future information
        window_slice = prices.iloc[i - window:i]

        X = add_constant(window_slice[ticker2])
        model = OLS(window_slice[ticker1], X).fit()
        beta = model.params[ticker2]

        # Apply today's beta to today's prices only
        betas[i] = beta
        spreads[i] = prices[ticker1].iloc[i] - beta * prices[ticker2].iloc[i]

    prices["beta"] = betas
    prices["spread"] = spreads

    # Rolling z-score — also uses only past window data
    rolling_mean = prices["spread"].rolling(window).mean()
    rolling_std = prices["spread"].rolling(window).std()
    prices["zscore"] = (prices["spread"] - rolling_mean) / rolling_std

    return prices


def generate_signals(prices, upper=2.0, lower=-2.0):
    zscores = prices["zscore"].values
    positions = np.zeros(len(zscores), dtype=int)
    position = 0

    for i in range(1, len(zscores)):
        z = zscores[i]

        # Skip if z-score not yet calculated (first 60 days)
        if np.isnan(z):
            continue

        if position == 0:
            if z > upper:
                position = -1
            elif z < lower:
                position = 1

        elif position == 1:
            if z >= 0:
                position = 0

        elif position == -1:
            if z <= 0:
                position = 0

        positions[i] = position

    return pd.Series(positions, index=prices.index, name="position")


def backtest(prices, positions, transaction_cost=0.001):
    prices = prices.copy()

    prices["ma_ret"] = prices["MA"].pct_change()
    prices["v_ret"] = prices["V"].pct_change()

    # Use the rolling beta for each day's return calculation
    strategy_rets = np.zeros(len(prices))

    for i in range(1, len(prices)):
        pos = positions.iloc[i - 1]  # previous day's position
        beta = prices["beta"].iloc[i]

        if pos == 0 or np.isnan(beta):
            strategy_rets[i] = 0
        else:
            strategy_rets[i] = pos * (
                prices["ma_ret"].iloc[i] - beta * prices["v_ret"].iloc[i]
            )

    prices["strategy_ret"] = strategy_rets

    # Transaction costs on position changes
    position_changes = positions.diff().fillna(0) != 0
    prices.loc[position_changes, "strategy_ret"] -= transaction_cost

    # Cumulative returns
    prices["cum_strategy"] = (1 + prices["strategy_ret"]).cumprod()
    prices["cum_ma"] = (1 + prices["ma_ret"]).cumprod()
    prices["cum_v"] = (1 + prices["v_ret"]).cumprod()

    return prices


def compute_metrics(prices, positions):
    ret = prices["strategy_ret"].dropna()

    total_return = prices["cum_strategy"].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(ret)) - 1
    annualized_vol = ret.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol
    rolling_max = prices["cum_strategy"].cummax()
    drawdown = (prices["cum_strategy"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    trade_entries = (positions.diff().fillna(0) != 0) & (positions != 0)

    print("\n--- Rolling Window Strategy ---")
    print(f"Window:              60 days")
    print(f"Number of trades:    {trade_entries.sum()}")
    print(f"Time in market:      {(positions != 0).mean():.1%}")
    print(f"Total Return:        {total_return:.2%}")
    print(f"Annualized Return:   {annualized_return:.2%}")
    print(f"Annualized Vol:      {annualized_vol:.2%}")
    print(f"Sharpe Ratio:        {sharpe:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2%}")

    print("\n--- Benchmark (Buy & Hold) ---")
    print(f"MA Total Return:     {prices['cum_ma'].iloc[-1] - 1:.2%}")
    print(f"V Total Return:      {prices['cum_v'].iloc[-1] - 1:.2%}")


if __name__ == "__main__":
    START = "2020-01-01"
    END = "2024-01-01"

    ma = get_price("MA", START, END)
    v = get_price("V", START, END)
    prices = pd.DataFrame({"MA": ma, "V": v}).dropna()

    print("Computing rolling betas... (this takes a moment)")
    prices = compute_rolling_spread(prices, "MA", "V", window=252)
    positions = generate_signals(prices)
    results = backtest(prices, positions)
    compute_metrics(results, positions)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11))

    # Rolling beta over time
    prices["beta"].plot(ax=ax1, color="steelblue", title=f"Rolling Beta (252-day window)")
    ax1.axhline(1.7294, color="red", linestyle="--", linewidth=1, label="Static beta (old)")
    ax1.set_ylabel("Beta")
    ax1.legend()
    ax1.grid(True)

    # Z-score with signals
    prices["zscore"].plot(ax=ax2, color="purple", title="Rolling Z-Score with Signals")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(2, color="red", linewidth=1, linestyle="--")
    ax2.axhline(-2, color="green", linewidth=1, linestyle="--")
    ax2.grid(True)

    # Cumulative returns
    results[["cum_strategy", "cum_ma", "cum_v"]].plot(
        ax=ax3,
        title="Cumulative Returns: Rolling Strategy vs Benchmarks"
    )
    ax3.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
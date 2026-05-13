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
    prices = prices.copy()
    X = add_constant(prices[ticker2])
    model = OLS(prices[ticker1], X).fit()
    beta = model.params[ticker2]
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


def backtest(prices, positions, beta, transaction_cost=0.001):
    """
    Simulate the strategy returns.
    Position = 1  → long MA, short V (scaled by beta)
    Position = -1 → short MA, long V (scaled by beta)
    transaction_cost = 0.1% per trade per leg (realistic estimate)
    """
    prices = prices.copy()

    # Daily returns of each stock
    prices["ma_ret"] = prices["MA"].pct_change()
    prices["v_ret"] = prices["V"].pct_change()

    # Strategy return:
    # Long MA / short V  → +ma_ret - beta * v_ret
    # Short MA / long V  → -ma_ret + beta * v_ret
    prices["strategy_ret"] = positions.shift(1) * (prices["ma_ret"] - beta * prices["v_ret"])

    # Subtract transaction costs on position changes
    position_changes = positions.diff().fillna(0) != 0
    prices.loc[position_changes, "strategy_ret"] -= transaction_cost

    # Cumulative returns
    prices["cum_strategy"] = (1 + prices["strategy_ret"]).cumprod()
    prices["cum_ma"] = (1 + prices["ma_ret"]).cumprod()
    prices["cum_v"] = (1 + prices["v_ret"]).cumprod()

    return prices


def compute_metrics(prices):
    ret = prices["strategy_ret"].dropna()

    total_return = prices["cum_strategy"].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(ret)) - 1
    annualized_vol = ret.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol
    rolling_max = prices["cum_strategy"].cummax()
    drawdown = (prices["cum_strategy"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    print("\n--- Strategy Performance ---")
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

    prices, beta = compute_spread(prices, "MA", "V")
    positions = generate_signals(prices)
    results = backtest(prices, positions, beta)

    compute_metrics(results)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    results[["cum_strategy", "cum_ma", "cum_v"]].plot(
        ax=ax1,
        title="Cumulative Returns: Strategy vs Benchmarks",
        label=["Pairs Strategy", "Buy & Hold MA", "Buy & Hold V"]
    )
    ax1.set_ylabel("Cumulative Return (1 = starting value)")
    ax1.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax1.grid(True)

    # Drawdown
    rolling_max = results["cum_strategy"].cummax()
    drawdown = (results["cum_strategy"] - rolling_max) / rolling_max
    drawdown.plot(ax=ax2, title="Strategy Drawdown", color="red")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
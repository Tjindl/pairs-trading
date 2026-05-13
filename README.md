# Pairs Trading Strategy — MA/V Cointegration Backtest

A quantitative trading research framework built from scratch in Python. This project implements a **statistical arbitrage pairs trading strategy** using cointegration analysis on equity price data, backtested with realistic transaction costs.

**Result:** 45.49% total return over 4 years (2020–2024), Sharpe Ratio of 0.97, beating buy-and-hold benchmarks of both MA (43.74%) and V (40.05%) while being in the market only 16.2% of the time.

---

## What Is Pairs Trading?

Pairs trading is a **market-neutral** strategy — it doesn't bet on the overall market going up or down. Instead it bets on the *relative* movement between two stocks that are statistically linked.

The core idea:
1. Find two stocks whose prices move together over time (cointegrated)
2. When they diverge more than usual — one has moved too far relative to the other
3. Bet they will converge back — buy the underperformer, short the overperformer
4. Exit when they converge

Because you're simultaneously long one stock and short another, the strategy is largely immune to broad market moves. A market crash hurts both positions roughly equally, leaving the spread intact.

---

## Mathematical Foundation

### Stationarity

A time series $X_t$ is **stationary** if its statistical properties don't change over time:

$$E[X_t] = \mu \quad \text{(constant mean)}$$
$$\text{Var}(X_t) = \sigma^2 \quad \text{(constant variance)}$$

Stock prices are **not** stationary — they follow a random walk and drift over time with no fixed mean to revert to.

### Cointegration

Two non-stationary series $A_t$ and $B_t$ are **cointegrated** if there exists a $\beta$ such that:

$$S_t = A_t - \beta B_t \sim \text{stationary}$$

The spread $S_t$ has a fixed mean it always returns to — even though each stock individually drifts freely. This is the mathematical foundation of the trade: we're not betting on either stock, we're betting on the spread.

### The Hedge Ratio

$\beta$ is found via OLS regression of $A$ on $B$:

$$A_t = \alpha + \beta B_t + \epsilon_t$$

The residuals $\epsilon_t$ are the spread. For MA/V we find $\beta = 1.7294$ — meaning for every 1 share of MA, we hold 1.7294 shares of V on the other side. This makes the portfolio dollar-neutral with respect to common factor movements.

### Z-Score Normalization

The raw spread is normalized to a z-score for consistent signal thresholds:

$$z_t = \frac{S_t - \mu_S}{\sigma_S}$$

Trading signals are generated at $z_t = \pm 2$ — two standard deviations from the mean, where mean reversion is statistically expected.

---

## Project Structure

```
pairs-trading/
├── src/
│   ├── data.py              # Data pipeline — fetches and cleans price data
│   ├── cointegration.py     # Statistical tests — ADF and Engle-Granger
│   ├── spread.py            # Hedge ratio calculation and spread visualization
│   ├── signals.py           # Signal generation logic
│   └── backtest.py          # Full backtest engine with performance metrics
├── data/                    # Cached price data
├── notebooks/               # Exploratory analysis
├── requirements.txt
└── README.md
```

---

## Step-by-Step Walkthrough

### Step 1 — Data Pipeline (`src/data.py`)

We pull 4 years of daily closing prices (2020–2024) using `yfinance`. Prices are cleaned by dropping any days where either stock has missing data (e.g. different trading holidays).

We normalize prices to 1 for visual comparison — this doesn't affect the analysis, only the chart. The raw prices are used for all calculations.

**Key insight from the data:** MA and V move almost identically day-to-day, which is the visual intuition before any formal testing.

---

### Step 2 — Pair Selection via Cointegration Testing (`src/cointegration.py`)

Not every pair that looks related actually is. We formally test 8 candidate pairs:

| Pair | P-Value | Cointegrated |
|------|---------|--------------|
| KO/PEP | 0.0744 | ❌ No |
| GLD/SLV | 0.6186 | ❌ No |
| XOM/CVX | 0.2687 | ❌ No |
| **MA/V** | **0.0244** | **✅ Yes** |
| MSFT/GOOGL | 0.6204 | ❌ No |
| JPM/BAC | 0.9906 | ❌ No |
| WMT/TGT | 0.5799 | ❌ No |
| MCD/YUM | 0.2686 | ❌ No |

**The testing process for each pair:**

1. **ADF Test on each stock individually** — confirms each series is non-stationary (random walk) on its own. Both MA and V pass: p-values of ~0.54 and ~0.57, confirming they don't revert to a fixed mean individually.

2. **Engle-Granger Cointegration Test** — regresses MA on V, then runs an ADF test on the residuals. If the residuals are stationary, the pair is cointegrated. MA/V passes with p = 0.0244 (95% confidence threshold is p < 0.05).

**Why only MA/V?** Most pairs that seem intuitively related fail the formal test because common-sense similarity ≠ statistical cointegration. GLD/SLV failing is a good example — gold and silver are related commodities but their price ratio has shifted structurally over this period.

---

### Step 3 — Spread Calculation (`src/spread.py`)

With MA/V confirmed cointegrated, we calculate the hedge ratio via OLS:

- **Beta: 1.7294** — for every share of MA, short 1.7294 shares of V
- **R-squared: 0.9224** — 92% of MA's movement is explained by V, confirming the tight relationship
- **Spread:** $S_t = MA_t - 1.7294 \times V_t$

The spread is then normalized to a z-score. The z-score plot confirms stationarity visually — it oscillates around zero and repeatedly reverts from extremes, exactly as cointegration theory predicts.

---

### Step 4 — Signal Generation (`src/signals.py`)

Trading rules:

| Z-Score | Signal | Action |
|---------|--------|--------|
| $z_t > +2$ | Short MA, Long V | MA overpriced relative to V |
| $z_t < -2$ | Long MA, Short V | MA underpriced relative to V |
| $z_t$ crosses 0 | Exit | Spread has converged |

**Result:** 4 trades over 4 years, 16.2% time in market. The strategy is inactive 84% of the time — capital is only deployed at high-conviction divergences.

---

### Step 5 — Backtesting (`src/backtest.py`)

The backtest simulates realistic P&L:

**Return calculation:**
- Long position: $r_{strategy} = r_{MA} - \beta \cdot r_V$
- Short position: $r_{strategy} = -r_{MA} + \beta \cdot r_V$

**Transaction costs:** 0.1% per trade applied on every position change (covers brokerage commissions and bid-ask spread).

**Performance metrics:**

| Metric | Strategy | MA Buy & Hold | V Buy & Hold |
|--------|----------|---------------|--------------|
| Total Return | **45.49%** | 43.74% | 40.05% |
| Annualized Return | 9.86% | — | — |
| Annualized Volatility | 10.19% | — | — |
| Sharpe Ratio | **0.97** | — | — |
| Max Drawdown | -12.23% | — | — |
| Time in Market | 16.2% | 100% | 100% |

The strategy outperforms both benchmarks while being invested only 16.2% of the time — the remaining 83.8% of capital is free for other opportunities.

---

## Known Limitations

These are the honest caveats a rigorous quant would flag:

**1. Small sample size**
Only 4 trades over 4 years. Statistical confidence requires hundreds of trades. Results could be partially attributable to luck.

**2. Look-ahead bias in hedge ratio**
The hedge ratio $\beta$ was calculated using the full 4-year dataset and then applied to the entire period. In live trading, you wouldn't know the future beta. A production implementation would use a rolling window (e.g. recalculate $\beta$ every 60 days using only past data).

**3. Regime change**
All 4 trades occurred in 2021–2022. From mid-2022 onwards the spread barely reached ±1 standard deviation. This suggests either the pair became more efficiently priced, or the static hedge ratio drifted and no longer accurately captures the relationship.

**4. Short-selling costs not modeled**
Borrowing shares to short has an annual cost (typically 0.5–2%). This would reduce returns on short positions.

**5. No slippage model**
We assume trades execute at the closing price. In reality, large orders move the market. This matters less for daily pairs trading on liquid stocks like MA and V.

---

## Potential Extensions

- **Rolling window beta** — recalculate hedge ratio on a 60-day rolling basis to reduce look-ahead bias and adapt to regime changes
- **Dynamic thresholds** — use rolling mean and std for z-score calculation rather than static full-period values
- **Multi-pair portfolio** — run the strategy across multiple cointegrated pairs simultaneously to increase trade frequency and diversify
- **Walk-forward validation** — train on first 2 years, test on next 2 years, to properly measure out-of-sample performance
- **Factor model** — incorporate Fama-French factors to ensure the spread is truly market-neutral

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/tjindl/pairs-trading
cd pairs-trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/data.py          # visualize price data
python src/cointegration.py # test pairs for cointegration
python src/spread.py        # calculate and plot the spread
python src/signals.py       # generate trading signals
python src/backtest.py      # run backtest and print metrics
```

---

## Dependencies

```
yfinance
pandas
numpy
matplotlib
statsmodels
```

---

## Background

Built as a quantitative finance research project to explore statistical arbitrage using real equity data. Implements the full pipeline from data ingestion through statistical testing, signal generation, and backtesting — the same workflow used by quantitative researchers at hedge funds and trading firms.

**Author:** Tushar Jindal — BSc Mathematics, University of British Columbia  
**Contact:** tushar.bzp05@gmail.com | [LinkedIn](https://linkedin.com/in/tushar-jindal-97602420b/) | [GitHub](https://github.com/tjindl)
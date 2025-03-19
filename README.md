```markdown
# Genetic Algorithm-Based Forex Trading System

This repository demonstrates an **algorithmic forex trading system** that combines rule-based technical signals with a **genetic algorithm (GA)** optimizer. The GA selects the optimal combination of trading rules to maximize performance metrics (e.g., risk-adjusted returns).

---

## 1. Purpose

1. **Technical Signals**: We generate trading signals from multiple **rule-based indicators** (moving averages, RSI, etc.).  
2. **Genetic Optimization**: We use a **genetic algorithm** to weight or combine these signals to find an optimal trading strategy on historical data.  
3. **Evaluation**: We backtest on both training and testing periods, plotting performance metrics, trade executions, and candlestick charts with entry/exit markers.

---

## 2. Requirements

- **Python 3.8+**  
- **Pandas** (for data handling)  
- **NumPy** (for numerical operations)  
- **matplotlib** (for plotting)  
- **mplfinance** (for candlestick charts)  
- **Jupyter Notebook** (optional, for interactive development)  

Typical installation:
```bash
pip install numpy pandas matplotlib mplfinance
```

---

## 3. Logic Overview

1. **Data Loading**  
   - Load historical OHLC data (e.g., `AUDUSD_M5.csv`).  
   - Convert or create a `DateTimeIndex`.

2. **Rule-Based Signals**  
   - We have multiple indicators (e.g., SMA, EMA, RSI, Bollinger Bands, etc.).  
   - Each indicator outputs a trading signal (+1 for long, -1 for short, 0 for flat).

3. **Genetic Algorithm**  
   - **Population** of candidate solutions (weight vectors).  
   - **Fitness**: risk-adjusted return (Sharpe-like ratio) or custom SSR metric.  
   - **Selection**, **crossover**, and **mutation** produce new generations.  
   - Convergence yields a “best” weight vector.

4. **Evaluation**  
   - Apply best weights to signals => final positions.  
   - Compute PnL, Sharpe ratio, max drawdown, etc.  
   - Plot candlesticks + trade markers.

---

## 4. Genetic Algorithm Flow (ASCII Diagram)

Below is a **simplified** text flow diagram of the GA:

```
       ┌───────────────────────┐
       │  Initialize Population│
       │ (random weight vectors) 
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │   Evaluate Fitness     │
       │ (apply weights to      │
       │ signals & measure SSR) │
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │   Select Parents       │
       │ (top N by fitness)     │
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │    Crossover +         │
       │    Mutation            │
       │ (create offspring)     │
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │  New Generation        │
       │(combine parents+offspr.)  
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │   Convergence Check    │
       │ (repeat or stop)       │
       └───────────────────────┘
```

1. **Initialize Population**: random solutions (weight vectors).  
2. **Evaluate Fitness**: each solution’s risk-adjusted performance.  
3. **Selection**: pick the best solutions as parents.  
4. **Crossover**: mix genes (weights) from two parents.  
5. **Mutation**: small random changes.  
6. **New Generation**: combine parents & offspring.  
7. **Repeat** until convergence or max generations.

---

## 5. Usage

1. **Clone** this repo:
   ```bash
   git clone https://github.com/your-username/ga-forex-trading.git
   cd ga-forex-trading
   ```
2. **Install Requirements**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   python Main_notebook.py
   ```
4. **Outputs**:
   - Plots of training & testing performance.  
   - Candlestick charts with trade markers.  
   - Printed metrics (Sharpe ratio, max drawdown, etc.).

---

**Enjoy experimenting with the Genetic Algorithm-based Forex Trading System!**
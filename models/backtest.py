"""
Module de backtesting — simule un portefeuille en suivant les predictions des modeles.
Repond a la question : "Si j'avais suivi les predictions, combien j'aurais gagne ?"
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgboost_model import train_xgboost, FEATURE_COLS
from sklearn.preprocessing import StandardScaler


def backtest_strategy(
    df: pd.DataFrame,
    ticker: str,
    initial_capital: float = 10000.0,
    model_type: str = "xgboost",
    train_ratio: float = 0.6,
) -> dict:
    """
    Simule un portefeuille base sur les predictions du modele.

    Strategie:
    - Si le modele predit une HAUSSE -> acheter (ou garder)
    - Si le modele predit une BAISSE -> vendre (ou ne pas acheter)

    Retourne les resultats detailles du backtest.
    """
    ticker_data = df[df["Ticker"] == ticker].sort_index().copy()
    ticker_data = ticker_data.dropna(subset=FEATURE_COLS + ["Close"])

    if len(ticker_data) < 100:
        return None

    # Split : entrainement | backtest
    split_idx = int(len(ticker_data) * train_ratio)
    train_data = ticker_data.iloc[:split_idx]
    test_data = ticker_data.iloc[split_idx:]

    # Entrainer le modele sur la partie train
    X_train = train_data[FEATURE_COLS].values
    y_train = train_data["Close"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_train_s, y_train)

    # Simuler le trading sur la partie test
    capital = initial_capital
    shares = 0
    portfolio_values = []
    buy_hold_values = []
    trades = []
    position = "cash"  # "cash" ou "invested"

    first_price = float(test_data["Close"].iloc[0])
    buy_hold_shares = initial_capital / first_price

    for i in range(len(test_data) - 1):
        current_price = float(test_data["Close"].iloc[i])
        next_actual_price = float(test_data["Close"].iloc[i + 1])

        # Prediction du modele
        features = test_data[FEATURE_COLS].iloc[i:i+1].values
        features_scaled = scaler.transform(features)
        predicted_price = float(model.predict(features_scaled)[0])

        # Decision
        predicted_change = (predicted_price - current_price) / current_price

        if predicted_change > 0.001 and position == "cash":
            # Acheter
            shares = capital / current_price
            capital = 0
            position = "invested"
            trades.append({"day": i, "action": "BUY", "price": current_price})

        elif predicted_change < -0.001 and position == "invested":
            # Vendre
            capital = shares * current_price
            shares = 0
            position = "cash"
            trades.append({"day": i, "action": "SELL", "price": current_price})

        # Valeur du portefeuille
        if position == "invested":
            portfolio_value = shares * current_price
        else:
            portfolio_value = capital

        portfolio_values.append(portfolio_value)
        buy_hold_values.append(buy_hold_shares * current_price)

    # Valeur finale
    final_price = float(test_data["Close"].iloc[-1])
    if position == "invested":
        final_value = shares * final_price
    else:
        final_value = capital

    buy_hold_final = buy_hold_shares * final_price

    # Metriques
    strategy_return = (final_value - initial_capital) / initial_capital * 100
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital * 100

    # Drawdown max
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Sharpe ratio (simplifie)
    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Nombre de trades gagnants
    winning_trades = 0
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            if trades[i]["action"] == "BUY" and trades[i+1]["action"] == "SELL":
                if trades[i+1]["price"] > trades[i]["price"]:
                    winning_trades += 1

    total_pairs = len(trades) // 2

    return {
        "ticker": ticker,
        "model": model_type,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "strategy_return": round(strategy_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "outperformance": round(strategy_return - buy_hold_return, 2),
        "total_trades": len(trades),
        "winning_trades": winning_trades,
        "win_rate": round(winning_trades / total_pairs * 100, 1) if total_pairs > 0 else 0,
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "test_days": len(test_data),
        "portfolio_values": portfolio_values,
        "buy_hold_values": buy_hold_values,
        "trades": trades,
        "dates": test_data.index[:-1].tolist(),
    }


def backtest_multiple(
    df: pd.DataFrame,
    tickers: list[str] = None,
    initial_capital: float = 10000.0,
) -> list[dict]:
    """Backtest sur plusieurs tickers."""
    if tickers is None:
        tickers = df["Ticker"].unique().tolist()

    results = []
    for ticker in tickers:
        print(f"  Backtesting {ticker}...")
        result = backtest_strategy(df, ticker, initial_capital)
        if result:
            results.append(result)
            gain = "+" if result["strategy_return"] > 0 else ""
            print(f"    Strategie: {gain}{result['strategy_return']:.1f}% | "
                  f"Buy&Hold: {result['buy_hold_return']:.1f}% | "
                  f"Trades: {result['total_trades']}")
    return results


if __name__ == "__main__":
    from pathlib import Path
    DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "all_tickers_clean.csv"
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    print("=" * 60)
    print("BACKTESTING — Simulation de portefeuille")
    print("=" * 60)
    print(f"Capital initial: $10,000\n")

    results = backtest_multiple(df, initial_capital=10000)

    print(f"\n{'='*60}")
    print(f"{'Ticker':<10} {'Strategie':<12} {'Buy&Hold':<12} {'Surperf.':<12} {'Win Rate':<10}")
    print("-" * 56)
    for r in sorted(results, key=lambda x: x["outperformance"], reverse=True):
        print(f"{r['ticker']:<10} {r['strategy_return']:>+.1f}%{'':<5} "
              f"{r['buy_hold_return']:>+.1f}%{'':<5} "
              f"{r['outperformance']:>+.1f}%{'':<5} "
              f"{r['win_rate']:.0f}%")

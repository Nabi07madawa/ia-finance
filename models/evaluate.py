"""
Evaluation des modeles — mesure la precision de chaque modele de prediction.
Metriques: RMSE, MAE, MAPE, R2
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgboost_model import FEATURE_COLS


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_xgboost(df: pd.DataFrame, ticker: str, train_ratio: float = 0.8) -> dict:
    """Evalue le modele XGBoost sur un ticker."""
    from xgboost import XGBRegressor

    ticker_data = df[df["Ticker"] == ticker].sort_index().copy()
    ticker_data = ticker_data.dropna(subset=FEATURE_COLS + ["Close"])

    if len(ticker_data) < 100:
        return None

    split_idx = int(len(ticker_data) * train_ratio)
    train = ticker_data.iloc[:split_idx]
    test = ticker_data.iloc[split_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train["Close"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["Close"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return {
        "model": "XGBoost",
        "ticker": ticker,
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mape": round(float(mape(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "test_size": len(test),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "dates": test.index.tolist(),
    }


def evaluate_prophet(df: pd.DataFrame, ticker: str, train_ratio: float = 0.8) -> dict:
    """Evalue le modele Prophet sur un ticker."""
    try:
        from prophet import Prophet
    except ImportError:
        return None

    ticker_data = df[df["Ticker"] == ticker].sort_index().copy()
    ticker_data = ticker_data.dropna(subset=["Close"])

    if len(ticker_data) < 100:
        return None

    split_idx = int(len(ticker_data) * train_ratio)
    train = ticker_data.iloc[:split_idx]
    test = ticker_data.iloc[split_idx:]

    prophet_train = pd.DataFrame({
        "ds": train.index,
        "y": train["Close"].values,
    })

    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test.index})
    forecast = model.predict(future)
    y_pred = forecast["yhat"].values
    y_test = test["Close"].values

    return {
        "model": "Prophet",
        "ticker": ticker,
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mape": round(float(mape(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "test_size": len(test),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "dates": test.index.tolist(),
    }


def evaluate_lstm(df: pd.DataFrame, ticker: str, train_ratio: float = 0.8) -> dict:
    """Evalue le modele LSTM sur un ticker."""
    try:
        import torch
        from models.lstm_model import LSTMNetwork, SEQUENCE_LENGTH
    except ImportError:
        return None

    ticker_data = df[df["Ticker"] == ticker].sort_index().copy()
    ticker_data = ticker_data.dropna(subset=["Close"])

    if len(ticker_data) < SEQUENCE_LENGTH + 50:
        return None

    prices = ticker_data["Close"].values.reshape(-1, 1)
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices)

    split_idx = int(len(prices_scaled) * train_ratio)
    train_scaled = prices_scaled[:split_idx]
    test_scaled = prices_scaled[split_idx:]

    # Sequences pour le train
    X_train, y_train = [], []
    for i in range(SEQUENCE_LENGTH, len(train_scaled)):
        X_train.append(train_scaled[i - SEQUENCE_LENGTH:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train = torch.FloatTensor(np.array(X_train)).unsqueeze(-1)
    y_train = torch.FloatTensor(np.array(y_train))

    # Entrainer
    model = LSTMNetwork(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train).squeeze()
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Predire sur le test
    full_scaled = prices_scaled
    test_start = split_idx
    y_pred_list = []

    model.eval()
    with torch.no_grad():
        for i in range(test_start, len(full_scaled)):
            seq = full_scaled[i - SEQUENCE_LENGTH:i, 0]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
            pred = model(seq_tensor).item()
            y_pred_list.append(pred)

    y_pred_scaled = np.array(y_pred_list).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_test = ticker_data["Close"].values[test_start:]

    dates = ticker_data.index[test_start:].tolist()

    return {
        "model": "LSTM",
        "ticker": ticker,
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mape": round(float(mape(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "test_size": len(y_test),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "dates": dates,
    }


def evaluate_all_models(df: pd.DataFrame, ticker: str) -> list[dict]:
    """Evalue les 3 modeles pour un ticker et retourne les resultats."""
    results = []

    print(f"  Evaluation XGBoost pour {ticker}...")
    r = evaluate_xgboost(df, ticker)
    if r:
        results.append(r)

    print(f"  Evaluation Prophet pour {ticker}...")
    r = evaluate_prophet(df, ticker)
    if r:
        results.append(r)

    print(f"  Evaluation LSTM pour {ticker}...")
    r = evaluate_lstm(df, ticker)
    if r:
        results.append(r)

    return results


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "all_tickers_clean.csv"
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    print("=" * 60)
    print("EVALUATION DES MODELES")
    print("=" * 60)

    tickers = df["Ticker"].unique().tolist()[:3]

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        results = evaluate_all_models(df, ticker)
        print(f"\n  {'Modele':<10} {'RMSE':<12} {'MAE':<12} {'MAPE(%)':<12} {'R2':<10}")
        print(f"  {'-'*46}")
        for r in results:
            print(f"  {r['model']:<10} {r['rmse']:<12} {r['mae']:<12} {r['mape']:<12} {r['r2']:<10}")

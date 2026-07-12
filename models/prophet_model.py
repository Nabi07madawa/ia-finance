"""
Modele Prophet pour la prediction de series temporelles financieres.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent / "saved"


def prepare_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Prepare les donnees au format Prophet (ds, y)."""
    data = df[df["Ticker"] == ticker].copy()
    data = data.reset_index()
    data = data.rename(columns={"Date": "ds", "Close": "y"})
    data["ds"] = pd.to_datetime(data["ds"])
    return data[["ds", "y"]].dropna()


def train_prophet(df: pd.DataFrame, ticker: str, forecast_days: int = 30) -> dict:
    """Entraine un modele Prophet sur un ticker donne."""
    data = prepare_data(df, ticker)

    if len(data) < 60:
        print(f"  [!] Pas assez de donnees pour {ticker} ({len(data)} lignes)")
        return None

    # Split train/test (80/20)
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    # Entrainement
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(train)

    # Predictions sur la periode de test
    future_test = model.make_future_dataframe(periods=len(test))
    forecast_test = model.predict(future_test)

    # Calcul des metriques sur le test set
    pred_test = forecast_test.iloc[split_idx:]["yhat"].values
    actual_test = test["y"].values
    mae = np.mean(np.abs(pred_test - actual_test))
    mape = np.mean(np.abs((actual_test - pred_test) / actual_test)) * 100

    # Prevision future
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    return {
        "model": model,
        "ticker": ticker,
        "forecast": forecast,
        "metrics": {"mae": mae, "mape": mape},
        "train_size": len(train),
        "test_size": len(test),
    }


def save_model(result: dict) -> Path:
    """Sauvegarde le modele Prophet."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    filepath = MODELS_DIR / f"prophet_{ticker.replace('^', 'IDX_')}.pkl"
    joblib.dump(result["model"], filepath)
    return filepath


def load_model(ticker: str) -> Prophet:
    """Charge un modele Prophet sauvegarde."""
    filepath = MODELS_DIR / f"prophet_{ticker.replace('^', 'IDX_')}.pkl"
    return joblib.load(filepath)

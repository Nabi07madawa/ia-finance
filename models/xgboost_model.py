"""
Modele XGBoost pour la prediction du prix de cloture.
Utilise les features techniques comme predicteurs.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent / "saved"

FEATURE_COLS = ["Open", "High", "Low", "Volume", "MA_7", "MA_20", "MA_50", "Volatility_20d", "RSI_14"]
TARGET_COL = "Close"


def prepare_data(df: pd.DataFrame, ticker: str) -> tuple:
    """Prepare les donnees pour XGBoost avec features techniques."""
    data = df[df["Ticker"] == ticker].copy()
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])

    if len(data) < 100:
        return None, None, None, None, None

    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    # Split temporel (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_xgboost(df: pd.DataFrame, ticker: str) -> dict:
    """Entraine un modele XGBoost sur un ticker."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, ticker)

    if X_train is None:
        print(f"  [!] Pas assez de donnees pour {ticker}")
        return None

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metriques
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Importance des features
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))

    return {
        "model": model,
        "scaler": scaler,
        "ticker": ticker,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
        "feature_importance": importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def save_model(result: dict) -> Path:
    """Sauvegarde le modele XGBoost et son scaler."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    base_name = f"xgboost_{ticker.replace('^', 'IDX_')}"

    model_path = MODELS_DIR / f"{base_name}.pkl"
    scaler_path = MODELS_DIR / f"{base_name}_scaler.pkl"

    joblib.dump(result["model"], model_path)
    joblib.dump(result["scaler"], scaler_path)
    return model_path


def load_model(ticker: str) -> tuple:
    """Charge un modele XGBoost et son scaler."""
    base_name = f"xgboost_{ticker.replace('^', 'IDX_')}"
    model = joblib.load(MODELS_DIR / f"{base_name}.pkl")
    scaler = joblib.load(MODELS_DIR / f"{base_name}_scaler.pkl")
    return model, scaler


def predict(ticker: str, features: np.ndarray) -> float:
    """Fait une prediction avec le modele sauvegarde."""
    model, scaler = load_model(ticker)
    features_scaled = scaler.transform(features.reshape(1, -1))
    return model.predict(features_scaled)[0]

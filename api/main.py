import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from models.prophet_model import load_model as load_prophet, train_prophet
from models.xgboost_model import load_model as load_xgboost, FEATURE_COLS
from models.lstm_model import load_model as load_lstm, LSTMNetwork, SEQUENCE_LENGTH

import torch
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="IA Finance API",
    description="API de prediction et d'analyse financiere",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "all_tickers_clean.csv"
MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"


def get_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


def get_available_tickers() -> list[str]:
    df = get_data()
    return sorted(df["Ticker"].unique().tolist())


# --- Schemas ---

class PredictionResponse(BaseModel):
    ticker: str
    model: str
    prediction: float
    currency: str = "USD"


class ForecastPoint(BaseModel):
    date: str
    predicted: float
    lower: float | None = None
    upper: float | None = None


class ForecastResponse(BaseModel):
    ticker: str
    model: str
    horizon_days: int
    forecast: list[ForecastPoint]


class TickerAnalysis(BaseModel):
    ticker: str
    current_price: float
    change_1d: float | None
    change_7d: float | None
    change_30d: float | None
    ma_7: float | None
    ma_20: float | None
    ma_50: float | None
    volatility_20d: float | None
    rsi_14: float | None
    volume_avg_20d: float | None


class ModelMetrics(BaseModel):
    ticker: str
    model: str
    mape: float
    mae: float


# --- Routes de base ---

@app.get("/")
def root():
    return {"status": "ok", "message": "IA Finance API operationnelle", "version": "0.2.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tickers", summary="Liste des tickers disponibles")
def list_tickers():
    return {"tickers": get_available_tickers()}


# --- Routes de prediction ---

@app.get("/predict/{ticker}", response_model=PredictionResponse, summary="Prediction du prochain prix de cloture")
def predict(
    ticker: str,
    model: str = Query(default="xgboost", description="Modele: prophet, xgboost, lstm"),
):
    ticker = ticker.upper()
    df = get_data()

    if ticker not in df["Ticker"].unique():
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' non disponible")

    ticker_data = df[df["Ticker"] == ticker].sort_index()

    if model == "xgboost":
        prediction = _predict_xgboost(ticker, ticker_data)
    elif model == "lstm":
        prediction = _predict_lstm(ticker, ticker_data)
    elif model == "prophet":
        prediction = _predict_prophet(ticker, ticker_data)
    else:
        raise HTTPException(status_code=400, detail=f"Modele '{model}' non supporte. Choix: prophet, xgboost, lstm")

    return PredictionResponse(ticker=ticker, model=model, prediction=round(prediction, 2))


@app.get("/forecast/{ticker}", response_model=ForecastResponse, summary="Prevision sur N jours (Prophet)")
def forecast(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365, description="Nombre de jours a predire"),
):
    ticker = ticker.upper()
    df = get_data()

    if ticker not in df["Ticker"].unique():
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' non disponible")

    try:
        model = load_prophet(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele Prophet non trouve pour '{ticker}'")

    future = model.make_future_dataframe(periods=days)
    pred = model.predict(future)

    forecast_data = pred.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    points = [
        ForecastPoint(
            date=row["ds"].strftime("%Y-%m-%d"),
            predicted=round(row["yhat"], 2),
            lower=round(row["yhat_lower"], 2),
            upper=round(row["yhat_upper"], 2),
        )
        for _, row in forecast_data.iterrows()
    ]

    return ForecastResponse(ticker=ticker, model="prophet", horizon_days=days, forecast=points)


@app.get("/predict/ensemble/{ticker}", summary="Prediction ensemble (moyenne des 3 modeles)")
def predict_ensemble(ticker: str):
    ticker = ticker.upper()
    df = get_data()

    if ticker not in df["Ticker"].unique():
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' non disponible")

    ticker_data = df[df["Ticker"] == ticker].sort_index()
    predictions = {}
    errors = {}

    for model_name, predict_fn in [
        ("prophet", _predict_prophet),
        ("xgboost", _predict_xgboost),
        ("lstm", _predict_lstm),
    ]:
        try:
            predictions[model_name] = predict_fn(ticker, ticker_data)
        except Exception as e:
            errors[model_name] = str(e)

    if not predictions:
        raise HTTPException(status_code=500, detail=f"Aucun modele disponible: {errors}")

    ensemble = np.mean(list(predictions.values()))

    return {
        "ticker": ticker,
        "ensemble_prediction": round(float(ensemble), 2),
        "individual_predictions": {k: round(v, 2) for k, v in predictions.items()},
        "models_used": len(predictions),
        "errors": errors if errors else None,
    }


# --- Routes d'analyse ---

@app.get("/compare", summary="Comparaison des performances de plusieurs tickers")
def compare(
    tickers: str = Query(description="Tickers separes par des virgules (ex: AAPL,MSFT,TSLA)"),
    days: int = Query(default=30, ge=1, le=365),
):
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    df = get_data()
    available = df["Ticker"].unique().tolist()

    results = []
    for ticker in ticker_list:
        if ticker not in available:
            continue
        ticker_data = df[df["Ticker"] == ticker].sort_index().tail(days)
        if len(ticker_data) < 2:
            continue

        start_price = float(ticker_data.iloc[0]["Close"])
        end_price = float(ticker_data.iloc[-1]["Close"])
        performance = (end_price - start_price) / start_price

        results.append({
            "ticker": ticker,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "performance": round(performance * 100, 2),
            "volatility": round(float(ticker_data["Close"].pct_change().std()), 4),
        })

    results.sort(key=lambda x: x["performance"], reverse=True)
    return {"period_days": days, "comparison": results}


@app.get("/analysis/{ticker}", response_model=TickerAnalysis, summary="Analyse technique d'un ticker")
def analysis(ticker: str):
    ticker = ticker.upper()
    df = get_data()

    if ticker not in df["Ticker"].unique():
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' non disponible")

    ticker_data = df[df["Ticker"] == ticker].sort_index()
    latest = ticker_data.iloc[-1]

    current_price = float(latest["Close"])
    change_1d = _safe_pct_change(ticker_data, 1)
    change_7d = _safe_pct_change(ticker_data, 5)
    change_30d = _safe_pct_change(ticker_data, 22)

    volume_avg = float(ticker_data["Volume"].tail(20).mean()) if "Volume" in ticker_data.columns else None

    return TickerAnalysis(
        ticker=ticker,
        current_price=round(current_price, 2),
        change_1d=round(change_1d, 4) if change_1d else None,
        change_7d=round(change_7d, 4) if change_7d else None,
        change_30d=round(change_30d, 4) if change_30d else None,
        ma_7=round(float(latest["MA_7"]), 2) if pd.notna(latest.get("MA_7")) else None,
        ma_20=round(float(latest["MA_20"]), 2) if pd.notna(latest.get("MA_20")) else None,
        ma_50=round(float(latest["MA_50"]), 2) if pd.notna(latest.get("MA_50")) else None,
        volatility_20d=round(float(latest["Volatility_20d"]), 4) if pd.notna(latest.get("Volatility_20d")) else None,
        rsi_14=round(float(latest["RSI_14"]), 2) if pd.notna(latest.get("RSI_14")) else None,
        volume_avg_20d=round(volume_avg, 0) if volume_avg else None,
    )


@app.get("/analysis/{ticker}/history", summary="Historique des prix")
def history(
    ticker: str,
    days: int = Query(default=90, ge=1, le=1825, description="Nombre de jours d'historique"),
):
    ticker = ticker.upper()
    df = get_data()

    if ticker not in df["Ticker"].unique():
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' non disponible")

    ticker_data = df[df["Ticker"] == ticker].sort_index().tail(days)
    records = [
        {
            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        for idx, row in ticker_data.iterrows()
    ]

    return {"ticker": ticker, "days": len(records), "history": records}


@app.get("/models/status", summary="Statut des modeles disponibles")
def models_status():
    tickers = get_available_tickers()
    status = []

    for ticker in tickers:
        ticker_file = ticker.replace("^", "IDX_")
        status.append({
            "ticker": ticker,
            "prophet": (MODELS_DIR / f"prophet_{ticker_file}.pkl").exists(),
            "xgboost": (MODELS_DIR / f"xgboost_{ticker_file}.pkl").exists(),
            "lstm": (MODELS_DIR / f"lstm_{ticker_file}.pt").exists(),
        })

    return {"models": status}


# --- Fonctions internes de prediction ---

def _predict_xgboost(ticker: str, ticker_data: pd.DataFrame) -> float:
    try:
        model, scaler = load_xgboost(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele XGBoost non trouve pour '{ticker}'")

    latest = ticker_data[FEATURE_COLS].iloc[-1:].values
    features_scaled = scaler.transform(latest)
    return float(model.predict(features_scaled)[0])


def _predict_lstm(ticker: str, ticker_data: pd.DataFrame) -> float:
    try:
        model, scaler = load_lstm(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele LSTM non trouve pour '{ticker}'")

    close_prices = ticker_data["Close"].values[-SEQUENCE_LENGTH:].reshape(-1, 1)
    close_scaled = scaler.transform(close_prices)
    input_tensor = torch.FloatTensor(close_scaled).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()

    return float(scaler.inverse_transform(pred_scaled)[0][0])


def _predict_prophet(ticker: str, ticker_data: pd.DataFrame) -> float:
    try:
        model = load_prophet(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele Prophet non trouve pour '{ticker}'")

    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    return float(forecast.iloc[-1]["yhat"])


def _safe_pct_change(data: pd.DataFrame, periods: int) -> float | None:
    if len(data) <= periods:
        return None
    current = float(data["Close"].iloc[-1])
    past = float(data["Close"].iloc[-1 - periods])
    if past == 0:
        return None
    return (current - past) / past

"""
Recommandations automatiques — combine tous les signaux pour generer
des recommandations d'achat/vente pour chaque ticker.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_signal_rsi(rsi: float) -> tuple[str, float]:
    """Signal base sur le RSI."""
    if rsi < 30:
        return "ACHAT", 0.8
    elif rsi < 40:
        return "ACHAT", 0.4
    elif rsi > 70:
        return "VENTE", 0.8
    elif rsi > 60:
        return "VENTE", 0.4
    return "NEUTRE", 0.0


def get_signal_macd(macd: float, macd_signal: float, macd_hist: float) -> tuple[str, float]:
    """Signal base sur le MACD."""
    if macd > macd_signal and macd_hist > 0:
        strength = min(abs(macd_hist) / abs(macd_signal) if macd_signal != 0 else 0.5, 1.0)
        return "ACHAT", strength * 0.7
    elif macd < macd_signal and macd_hist < 0:
        strength = min(abs(macd_hist) / abs(macd_signal) if macd_signal != 0 else 0.5, 1.0)
        return "VENTE", strength * 0.7
    return "NEUTRE", 0.0


def get_signal_bollinger(bb_position: float) -> tuple[str, float]:
    """Signal base sur la position dans les Bollinger Bands."""
    if bb_position < 0.1:
        return "ACHAT", 0.7
    elif bb_position < 0.25:
        return "ACHAT", 0.4
    elif bb_position > 0.9:
        return "VENTE", 0.7
    elif bb_position > 0.75:
        return "VENTE", 0.4
    return "NEUTRE", 0.0


def get_signal_stochastic(stoch_k: float, stoch_d: float) -> tuple[str, float]:
    """Signal base sur le Stochastic Oscillator."""
    if stoch_k < 20 and stoch_d < 20:
        return "ACHAT", 0.7
    elif stoch_k < 30 and stoch_k > stoch_d:
        return "ACHAT", 0.4
    elif stoch_k > 80 and stoch_d > 80:
        return "VENTE", 0.7
    elif stoch_k > 70 and stoch_k < stoch_d:
        return "VENTE", 0.4
    return "NEUTRE", 0.0


def get_signal_ma_trend(close: float, ma_7: float, ma_20: float, ma_50: float) -> tuple[str, float]:
    """Signal base sur les moyennes mobiles (tendance)."""
    above_all = close > ma_7 > ma_20 > ma_50
    below_all = close < ma_7 < ma_20 < ma_50

    if above_all:
        return "ACHAT", 0.6
    elif close > ma_7 and ma_7 > ma_20:
        return "ACHAT", 0.3
    elif below_all:
        return "VENTE", 0.6
    elif close < ma_7 and ma_7 < ma_20:
        return "VENTE", 0.3
    return "NEUTRE", 0.0


def get_signal_momentum(return_1d: float, volatility: float) -> tuple[str, float]:
    """Signal base sur le momentum et la volatilite."""
    if volatility == 0 or np.isnan(volatility):
        return "NEUTRE", 0.0

    normalized = return_1d / volatility if volatility > 0 else 0
    if normalized > 1.5:
        return "ACHAT", 0.5
    elif normalized > 0.5:
        return "ACHAT", 0.3
    elif normalized < -1.5:
        return "VENTE", 0.5
    elif normalized < -0.5:
        return "VENTE", 0.3
    return "NEUTRE", 0.0


def analyze_ticker_signals(df: pd.DataFrame, ticker: str) -> dict:
    """Analyse tous les signaux pour un ticker et genere une recommandation."""
    ticker_data = df[df["Ticker"] == ticker].sort_index()

    if ticker_data.empty or len(ticker_data) < 50:
        return {"ticker": ticker, "recommendation": "DONNEES INSUFFISANTES", "signals": []}

    latest = ticker_data.iloc[-1]
    signals = []

    # RSI
    if "RSI_14" in latest.index and not np.isnan(latest["RSI_14"]):
        direction, strength = get_signal_rsi(latest["RSI_14"])
        signals.append({"indicator": "RSI (14)", "value": round(float(latest["RSI_14"]), 1),
                        "direction": direction, "strength": strength})

    # MACD
    if all(col in latest.index for col in ["MACD", "MACD_Signal", "MACD_Hist"]):
        if not any(np.isnan(latest[c]) for c in ["MACD", "MACD_Signal", "MACD_Hist"]):
            direction, strength = get_signal_macd(latest["MACD"], latest["MACD_Signal"], latest["MACD_Hist"])
            signals.append({"indicator": "MACD", "value": round(float(latest["MACD"]), 4),
                            "direction": direction, "strength": strength})

    # Bollinger Bands
    if "BB_Position" in latest.index and not np.isnan(latest["BB_Position"]):
        direction, strength = get_signal_bollinger(latest["BB_Position"])
        signals.append({"indicator": "Bollinger Bands", "value": round(float(latest["BB_Position"]), 3),
                        "direction": direction, "strength": strength})

    # Stochastic
    if all(col in latest.index for col in ["Stoch_K", "Stoch_D"]):
        if not any(np.isnan(latest[c]) for c in ["Stoch_K", "Stoch_D"]):
            direction, strength = get_signal_stochastic(latest["Stoch_K"], latest["Stoch_D"])
            signals.append({"indicator": "Stochastic", "value": round(float(latest["Stoch_K"]), 1),
                            "direction": direction, "strength": strength})

    # Moyennes mobiles
    if all(col in latest.index for col in ["MA_7", "MA_20", "MA_50"]):
        if not any(np.isnan(latest[c]) for c in ["MA_7", "MA_20", "MA_50"]):
            direction, strength = get_signal_ma_trend(latest["Close"], latest["MA_7"], latest["MA_20"], latest["MA_50"])
            signals.append({"indicator": "Tendance MA", "value": f"Prix vs MA",
                            "direction": direction, "strength": strength})

    # Momentum
    if all(col in latest.index for col in ["Return_1d", "Volatility_20d"]):
        if not any(np.isnan(latest[c]) for c in ["Return_1d", "Volatility_20d"]):
            direction, strength = get_signal_momentum(latest["Return_1d"], latest["Volatility_20d"])
            signals.append({"indicator": "Momentum", "value": round(float(latest["Return_1d"]) * 100, 2),
                            "direction": direction, "strength": strength})

    # Score global
    buy_score = sum(s["strength"] for s in signals if s["direction"] == "ACHAT")
    sell_score = sum(s["strength"] for s in signals if s["direction"] == "VENTE")
    total_possible = len(signals) * 0.8 if signals else 1

    net_score = (buy_score - sell_score) / total_possible

    if net_score > 0.4:
        recommendation = "ACHAT FORT"
        confidence = min(net_score / 0.6, 1.0)
    elif net_score > 0.15:
        recommendation = "ACHAT"
        confidence = net_score / 0.4
    elif net_score < -0.4:
        recommendation = "VENTE FORT"
        confidence = min(abs(net_score) / 0.6, 1.0)
    elif net_score < -0.15:
        recommendation = "VENTE"
        confidence = abs(net_score) / 0.4
    else:
        recommendation = "NEUTRE"
        confidence = 1.0 - abs(net_score) / 0.15

    # Info prix
    current_price = float(latest["Close"])
    price_change_1d = float(latest["Return_1d"]) * 100 if not np.isnan(latest["Return_1d"]) else 0

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "change_1d": round(price_change_1d, 2),
        "recommendation": recommendation,
        "confidence": round(confidence * 100, 1),
        "net_score": round(net_score, 3),
        "buy_score": round(buy_score, 3),
        "sell_score": round(sell_score, 3),
        "signals": signals,
        "n_signals": len(signals),
    }


def get_all_recommendations(df: pd.DataFrame, tickers: list[str] = None) -> list[dict]:
    """Genere des recommandations pour tous les tickers."""
    if tickers is None:
        tickers = df["Ticker"].unique().tolist()

    results = []
    for ticker in tickers:
        result = analyze_ticker_signals(df, ticker)
        results.append(result)

    # Trier par score (meilleurs achats en premier)
    results.sort(key=lambda x: x.get("net_score", 0), reverse=True)
    return results

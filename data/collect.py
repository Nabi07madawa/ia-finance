"""
Collecte de données financières via yfinance.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parent / "raw"

TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "^GSPC",  # S&P 500
    "^FCHI",  # CAC 40
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "SOL-USD",  # Solana
]

DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"


def collect_ticker(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Télécharge l'historique d'un ticker et retourne un DataFrame."""
    print(f"  Telechargement de {ticker}...")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"  [ERREUR] {ticker}: {e}")
        return pd.DataFrame()
    if data.empty:
        print(f"  [!] Aucune donnee pour {ticker}")
        return pd.DataFrame()
    # yfinance retourne un MultiIndex — on aplatit les colonnes
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data["Ticker"] = ticker
    return data


def collect_all(tickers: list[str] = None, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> None:
    """Collecte les données pour tous les tickers et les sauvegarde en CSV."""
    tickers = tickers or TICKERS
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")

    for ticker in tickers:
        df = collect_ticker(ticker, period, interval)
        if df.empty:
            continue

        filename = f"{ticker.replace('^', 'IDX_')}_{timestamp}.csv"
        filepath = RAW_DIR / filename
        df.to_csv(filepath)
        print(f"  [OK] {ticker} -> {filepath.name} ({len(df)} lignes)")

    print(f"\nCollecte terminee. Fichiers dans : {RAW_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("COLLECTE DE DONNÉES FINANCIÈRES")
    print("=" * 50)
    collect_all()

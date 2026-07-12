"""
Script principal d'entrainement de tous les modeles.
Entraine Prophet, XGBoost et LSTM sur chaque ticker disponible.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prophet_model import train_prophet, save_model as save_prophet
from models.xgboost_model import train_xgboost, save_model as save_xgboost
from models.lstm_model import train_lstm, save_model as save_lstm

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "all_tickers_clean.csv"


def load_data() -> pd.DataFrame:
    """Charge le dataset nettoye."""
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Donnees chargees : {len(df)} lignes, {df['Ticker'].nunique()} tickers")
    return df


def train_all_models(df: pd.DataFrame, tickers: list[str] = None) -> dict:
    """Entraine les 3 modeles sur chaque ticker."""
    if tickers is None:
        tickers = df["Ticker"].unique().tolist()

    results = {"prophet": {}, "xgboost": {}, "lstm": {}}

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"TICKER : {ticker}")
        print(f"{'='*50}")

        # Prophet
        print(f"\n  [Prophet] Entrainement...")
        result = train_prophet(df, ticker)
        if result:
            path = save_prophet(result)
            results["prophet"][ticker] = result["metrics"]
            print(f"  [Prophet] MAE={result['metrics']['mae']:.2f}, MAPE={result['metrics']['mape']:.2f}%")
            print(f"  [Prophet] Sauvegarde -> {path.name}")

        # XGBoost
        print(f"\n  [XGBoost] Entrainement...")
        result = train_xgboost(df, ticker)
        if result:
            path = save_xgboost(result)
            results["xgboost"][ticker] = result["metrics"]
            print(f"  [XGBoost] MAE={result['metrics']['mae']:.2f}, RMSE={result['metrics']['rmse']:.2f}, MAPE={result['metrics']['mape']:.2f}%")
            print(f"  [XGBoost] Top features: {sorted(result['feature_importance'].items(), key=lambda x: -x[1])[:3]}")
            print(f"  [XGBoost] Sauvegarde -> {path.name}")

        # LSTM
        print(f"\n  [LSTM] Entrainement...")
        result = train_lstm(df, ticker, epochs=50)
        if result:
            path = save_lstm(result)
            results["lstm"][ticker] = result["metrics"]
            print(f"  [LSTM] MAE={result['metrics']['mae']:.2f}, MAPE={result['metrics']['mape']:.2f}%")
            print(f"  [LSTM] Device: {result['device']}")
            print(f"  [LSTM] Sauvegarde -> {path.name}")

    return results


def print_summary(results: dict) -> None:
    """Affiche un resume comparatif des performances."""
    print(f"\n{'='*60}")
    print("RESUME DES PERFORMANCES (MAPE %)")
    print(f"{'='*60}")
    print(f"{'Ticker':<10} {'Prophet':<12} {'XGBoost':<12} {'LSTM':<12}")
    print("-" * 46)

    for ticker in set(
        list(results["prophet"].keys()) +
        list(results["xgboost"].keys()) +
        list(results["lstm"].keys())
    ):
        prophet_mape = results["prophet"].get(ticker, {}).get("mape", "-")
        xgboost_mape = results["xgboost"].get(ticker, {}).get("mape", "-")
        lstm_mape = results["lstm"].get(ticker, {}).get("mape", "-")

        p = f"{prophet_mape:.2f}%" if isinstance(prophet_mape, float) else "-"
        x = f"{xgboost_mape:.2f}%" if isinstance(xgboost_mape, float) else "-"
        l = f"{lstm_mape:.2f}%" if isinstance(lstm_mape, float) else "-"
        print(f"{ticker:<10} {p:<12} {x:<12} {l:<12}")


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRAINEMENT DES MODELES DE PREDICTION")
    print("=" * 60)

    df = load_data()
    results = train_all_models(df)
    print_summary(results)

    print(f"\nModeles sauvegardes dans : {Path(__file__).parent / 'saved'}")
    print("[OK] Entrainement termine.")

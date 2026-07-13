"""
Nettoyage et prétraitement des données financières brutes.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"


def load_raw_files() -> pd.DataFrame:
    """Charge tous les CSV bruts et les combine en un seul DataFrame."""
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {RAW_DIR}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        frames.append(df)
        print(f"  Chargé : {f.name} ({len(df)} lignes)")

    combined = pd.concat(frames, ignore_index=False)
    print(f"\n  Total combiné : {len(combined)} lignes")
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les étapes de nettoyage sur le DataFrame."""
    initial_rows = len(df)

    # 1. Supprimer les doublons
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    if dupes_removed:
        print(f"  ->{dupes_removed} doublons supprimés")

    # 2. Traiter les valeurs manquantes
    missing_before = df.isnull().sum().sum()
    # Forward fill pour les données de marché (le dernier prix connu est pertinent)
    df = df.ffill()
    # Backward fill pour les premières lignes restantes
    df = df.bfill()
    missing_after = df.isnull().sum().sum()
    print(f"  ->Valeurs manquantes : {missing_before} -> {missing_after}")

    # 3. Supprimer les lignes avec des prix négatifs ou nuls (anomalies)
    price_cols = ["Open", "High", "Low", "Close"]
    existing_price_cols = [c for c in price_cols if c in df.columns]
    if existing_price_cols:
        mask = (df[existing_price_cols] > 0).all(axis=1)
        anomalies = (~mask).sum()
        df = df[mask]
        if anomalies:
            print(f"  ->{anomalies} lignes avec prix <= 0 supprimées")

    # 4. Supprimer les volumes négatifs
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        df = df[df["Volume"] >= 0]
        if neg_vol:
            print(f"  ->{neg_vol} lignes avec volume négatif supprimées")

    # 5. Vérifier la cohérence High >= Low
    if "High" in df.columns and "Low" in df.columns:
        incoherent = (df["High"] < df["Low"]).sum()
        if incoherent:
            df = df[df["High"] >= df["Low"]]
            print(f"  ->{incoherent} lignes incohérentes (High < Low) supprimées")

    print(f"  ->Lignes finales : {len(df)} (sur {initial_rows} initiales)")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features techniques utiles pour le ML."""
    if "Close" not in df.columns:
        return df

    # Rendements journaliers
    df["Return_1d"] = df.groupby("Ticker")["Close"].pct_change()

    # Moyennes mobiles
    for window in [7, 20, 50]:
        df[f"MA_{window}"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.rolling(window=window).mean()
        )

    # Volatilité (écart-type glissant sur 20 jours)
    df["Volatility_20d"] = df.groupby("Ticker")["Return_1d"].transform(
        lambda x: x.rolling(window=20).std()
    )

    # RSI (Relative Strength Index) sur 14 jours
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["RSI_14"] = df.groupby("Ticker")["Close"].transform(compute_rsi)

    # MACD (Moving Average Convergence Divergence)
    df["EMA_12"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df["EMA_26"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df.groupby("Ticker")["MACD"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df.drop(columns=["EMA_12", "EMA_26"], inplace=True)

    # Bollinger Bands (20 jours)
    df["BB_Middle"] = df["MA_20"]
    df["BB_Std"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window=20).std()
    )
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df.drop(columns=["BB_Std"], inplace=True)

    # ATR (Average True Range) sur 14 jours
    if "High" in df.columns and "Low" in df.columns:
        prev_close = df.groupby("Ticker")["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR_14"] = df.groupby("Ticker")["TR"].transform(
            lambda x: x.rolling(window=14).mean()
        )
        df.drop(columns=["TR"], inplace=True)

    # Stochastic Oscillator (%K et %D)
    if "High" in df.columns and "Low" in df.columns:
        low_14 = df.groupby("Ticker")["Low"].transform(lambda x: x.rolling(14).min())
        high_14 = df.groupby("Ticker")["High"].transform(lambda x: x.rolling(14).max())
        df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
        df["Stoch_D"] = df.groupby("Ticker")["Stoch_K"].transform(
            lambda x: x.rolling(window=3).mean()
        )

    print(f"  ->Features ajoutees : Return_1d, MA_7/20/50, Volatility_20d, RSI_14,")
    print(f"    MACD, Bollinger Bands, ATR_14, Stochastic")
    return df


def save_processed(df: pd.DataFrame) -> None:
    """Sauvegarde les données nettoyées par ticker."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if "Ticker" in df.columns:
        for ticker, group in df.groupby("Ticker"):
            filename = f"{ticker.replace('^', 'IDX_')}_clean.csv"
            filepath = PROCESSED_DIR / filename
            group.to_csv(filepath)
            print(f"  [OK]{ticker} -> {filepath.name} ({len(group)} lignes)")
    else:
        filepath = PROCESSED_DIR / "all_clean.csv"
        df.to_csv(filepath)
        print(f"  [OK]Sauvegardé -> {filepath.name} ({len(df)} lignes)")

    # Aussi sauvegarder le dataset complet
    all_path = PROCESSED_DIR / "all_tickers_clean.csv"
    df.to_csv(all_path)
    print(f"  [OK]Dataset complet -> {all_path.name}")


def run_pipeline() -> pd.DataFrame:
    """Exécute la pipeline complète de nettoyage."""
    print("\n[1/4] Chargement des données brutes...")
    df = load_raw_files()

    print("\n[2/4] Nettoyage...")
    df = clean_data(df)

    print("\n[3/4] Ajout de features techniques...")
    df = add_features(df)

    print("\n[4/4] Sauvegarde...")
    save_processed(df)

    return df


if __name__ == "__main__":
    print("=" * 50)
    print("NETTOYAGE & PRÉTRAITEMENT DES DONNÉES")
    print("=" * 50)
    run_pipeline()
    print("\n[OK] Pipeline terminée avec succès.")

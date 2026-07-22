"""
Migration des donnees CSV vers Supabase PostgreSQL.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import get_client

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "all_tickers_clean.csv"


def migrate_prices():
    """Migre les prix historiques vers Supabase."""
    print("Chargement des donnees CSV...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    client = get_client()
    tickers = df["Ticker"].unique().tolist()

    total = 0
    for ticker in tickers:
        ticker_data = df[df["Ticker"] == ticker].copy()
        ticker_data = ticker_data.reset_index()
        # L'index s'appelle "Date" apres reset
        date_col = "Date" if "Date" in ticker_data.columns else "index"

        # Preparer les donnees par batch de 500
        records = []
        for _, row in ticker_data.iterrows():
            records.append({
                "ticker": ticker,
                "date": str(row[date_col].date()),
                "open": round(float(row["Open"]), 4) if pd.notna(row["Open"]) else None,
                "high": round(float(row["High"]), 4) if pd.notna(row["High"]) else None,
                "low": round(float(row["Low"]), 4) if pd.notna(row["Low"]) else None,
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
            })

        # Insert par batch de 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                client.table("prices").upsert(batch, on_conflict="ticker,date").execute()
            except Exception as e:
                print(f"  [ERREUR] {ticker} batch {i}: {e}")
                continue

        total += len(records)
        print(f"  [OK] {ticker}: {len(records)} lignes migrees")

    print(f"\nTotal: {total} lignes migrees vers Supabase")


def migrate_portfolio():
    """Migre le portefeuille JSON vers Supabase."""
    import json

    portfolio_path = Path(__file__).parent.parent / "data" / "portfolio.json"
    transactions_path = Path(__file__).parent.parent / "data" / "transactions.json"

    client = get_client()

    # Portfolio
    if portfolio_path.exists():
        portfolio = json.loads(portfolio_path.read_text())

        # Cash
        client.table("cash").upsert({"id": 1, "amount": portfolio["cash"]}).execute()
        print(f"  [OK] Cash: ${portfolio['cash']:.2f}")

        # Holdings
        for ticker, holding in portfolio.get("holdings", {}).items():
            client.table("portfolio").upsert({
                "ticker": ticker,
                "quantity": holding["quantity"],
                "avg_price": holding["avg_price"],
            }, on_conflict="ticker").execute()
            print(f"  [OK] Position: {ticker} x{holding['quantity']}")
    else:
        # Cash initial
        client.table("cash").upsert({"id": 1, "amount": 10000.0}).execute()
        print("  [OK] Cash initial: $10,000")

    # Transactions
    if transactions_path.exists():
        transactions = json.loads(transactions_path.read_text())
        if transactions:
            records = []
            for tx in transactions:
                records.append({
                    "type": tx["type"],
                    "ticker": tx["ticker"],
                    "quantity": tx["quantity"],
                    "price": tx["price"],
                    "total": tx["total"],
                })
            client.table("transactions").insert(records).execute()
            print(f"  [OK] {len(records)} transactions migrees")


def migrate_alerts():
    """Migre les alertes JSON vers Supabase."""
    import json

    alerts_path = Path(__file__).parent.parent / "data" / "alerts.json"
    client = get_client()

    if alerts_path.exists():
        alerts = json.loads(alerts_path.read_text())
        if alerts:
            records = []
            for alert in alerts:
                records.append({
                    "ticker": alert["ticker"],
                    "condition": alert["condition"],
                    "threshold": alert["threshold"],
                    "active": alert.get("active", True),
                })
            client.table("alerts").insert(records).execute()
            print(f"  [OK] {len(records)} alertes migrees")
    else:
        print("  Aucune alerte a migrer")


if __name__ == "__main__":
    print("=" * 50)
    print("MIGRATION VERS SUPABASE")
    print("=" * 50)

    print("\n[1/3] Migration des prix...")
    migrate_prices()

    print("\n[2/3] Migration du portefeuille...")
    migrate_portfolio()

    print("\n[3/3] Migration des alertes...")
    migrate_alerts()

    print("\n[OK] Migration terminee!")

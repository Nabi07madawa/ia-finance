"""
Portfolio Manager — gestion d'un portefeuille virtuel.
Permet d'acheter/vendre des actifs et suivre les gains/pertes.
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PORTFOLIO_FILE = Path(__file__).parent.parent / "data" / "portfolio.json"
TRANSACTIONS_FILE = Path(__file__).parent.parent / "data" / "transactions.json"


def load_portfolio() -> dict:
    """Charge le portefeuille."""
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"cash": 10000.0, "holdings": {}, "created_at": datetime.now().isoformat()}


def save_portfolio(portfolio: dict) -> None:
    """Sauvegarde le portefeuille."""
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2))


def load_transactions() -> list[dict]:
    """Charge l'historique des transactions."""
    if TRANSACTIONS_FILE.exists():
        return json.loads(TRANSACTIONS_FILE.read_text())
    return []


def save_transactions(transactions: list[dict]) -> None:
    """Sauvegarde les transactions."""
    TRANSACTIONS_FILE.write_text(json.dumps(transactions, indent=2))


def buy(ticker: str, quantity: float, price: float) -> dict:
    """Acheter des actions/crypto."""
    portfolio = load_portfolio()
    cost = quantity * price

    if cost > portfolio["cash"]:
        return {"error": f"Fonds insuffisants. Cash: ${portfolio['cash']:.2f}, Cout: ${cost:.2f}"}

    portfolio["cash"] -= cost

    if ticker in portfolio["holdings"]:
        # Moyenne le prix d'achat
        existing = portfolio["holdings"][ticker]
        total_qty = existing["quantity"] + quantity
        avg_price = (existing["quantity"] * existing["avg_price"] + cost) / total_qty
        portfolio["holdings"][ticker] = {"quantity": total_qty, "avg_price": round(avg_price, 4)}
    else:
        portfolio["holdings"][ticker] = {"quantity": quantity, "avg_price": price}

    save_portfolio(portfolio)

    # Enregistrer la transaction
    transaction = {
        "type": "BUY",
        "ticker": ticker,
        "quantity": quantity,
        "price": price,
        "total": round(cost, 2),
        "date": datetime.now().isoformat(),
    }
    transactions = load_transactions()
    transactions.append(transaction)
    save_transactions(transactions)

    return {"message": f"Achat de {quantity} {ticker} a ${price:.2f}", "transaction": transaction}


def sell(ticker: str, quantity: float, price: float) -> dict:
    """Vendre des actions/crypto."""
    portfolio = load_portfolio()

    if ticker not in portfolio["holdings"]:
        return {"error": f"Vous ne possedez pas de {ticker}"}

    holding = portfolio["holdings"][ticker]
    if quantity > holding["quantity"]:
        return {"error": f"Quantite insuffisante. Vous avez {holding['quantity']} {ticker}"}

    revenue = quantity * price
    portfolio["cash"] += revenue

    holding["quantity"] -= quantity
    if holding["quantity"] <= 0.0001:
        del portfolio["holdings"][ticker]
    else:
        portfolio["holdings"][ticker] = holding

    save_portfolio(portfolio)

    # Enregistrer la transaction
    transaction = {
        "type": "SELL",
        "ticker": ticker,
        "quantity": quantity,
        "price": price,
        "total": round(revenue, 2),
        "date": datetime.now().isoformat(),
    }
    transactions = load_transactions()
    transactions.append(transaction)
    save_transactions(transactions)

    return {"message": f"Vente de {quantity} {ticker} a ${price:.2f}", "transaction": transaction}


def get_portfolio_value(df: pd.DataFrame) -> dict:
    """Calcule la valeur actuelle du portefeuille."""
    portfolio = load_portfolio()
    holdings_value = []
    total_invested = 0
    total_current = 0

    for ticker, holding in portfolio["holdings"].items():
        ticker_data = df[df["Ticker"] == ticker]
        if ticker_data.empty:
            current_price = holding["avg_price"]
        else:
            current_price = float(ticker_data.sort_index().iloc[-1]["Close"])

        invested = holding["quantity"] * holding["avg_price"]
        current = holding["quantity"] * current_price
        pnl = current - invested
        pnl_pct = (pnl / invested) * 100 if invested > 0 else 0

        holdings_value.append({
            "ticker": ticker,
            "quantity": holding["quantity"],
            "avg_price": round(holding["avg_price"], 2),
            "current_price": round(current_price, 2),
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

        total_invested += invested
        total_current += current

    total_value = portfolio["cash"] + total_current
    total_pnl = total_current - total_invested

    return {
        "cash": round(portfolio["cash"], 2),
        "holdings": holdings_value,
        "total_invested": round(total_invested, 2),
        "total_holdings_value": round(total_current, 2),
        "total_portfolio_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / total_invested * 100) if total_invested > 0 else 0, 2),
    }


def reset_portfolio(initial_cash: float = 10000.0) -> dict:
    """Reinitialise le portefeuille."""
    portfolio = {"cash": initial_cash, "holdings": {}, "created_at": datetime.now().isoformat()}
    save_portfolio(portfolio)
    save_transactions([])
    return portfolio

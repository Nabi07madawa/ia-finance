"""
Portfolio Manager — version Supabase.
Lit et ecrit dans PostgreSQL au lieu des fichiers JSON.
"""
import pandas as pd
from datetime import datetime
from database.db import get_client


def load_portfolio_db() -> dict:
    """Charge le portefeuille depuis Supabase."""
    client = get_client()

    # Cash
    cash_result = client.table("cash").select("amount").eq("id", 1).execute()
    cash = float(cash_result.data[0]["amount"]) if cash_result.data else 10000.0

    # Holdings
    holdings_result = client.table("portfolio").select("*").execute()
    holdings = {}
    for h in holdings_result.data:
        holdings[h["ticker"]] = {
            "quantity": float(h["quantity"]),
            "avg_price": float(h["avg_price"]),
        }

    return {"cash": cash, "holdings": holdings}


def buy_db(ticker: str, quantity: float, price: float) -> dict:
    """Acheter des actions/crypto (Supabase)."""
    client = get_client()
    portfolio = load_portfolio_db()
    cost = quantity * price

    if cost > portfolio["cash"]:
        return {"error": f"Fonds insuffisants. Cash: ${portfolio['cash']:.2f}, Cout: ${cost:.2f}"}

    new_cash = portfolio["cash"] - cost

    # Mettre a jour le cash
    client.table("cash").update({"amount": round(new_cash, 4)}).eq("id", 1).execute()

    # Mettre a jour ou creer la position
    if ticker in portfolio["holdings"]:
        existing = portfolio["holdings"][ticker]
        total_qty = existing["quantity"] + quantity
        avg_price = (existing["quantity"] * existing["avg_price"] + cost) / total_qty
        client.table("portfolio").update({
            "quantity": round(total_qty, 6),
            "avg_price": round(avg_price, 4),
        }).eq("ticker", ticker).execute()
    else:
        client.table("portfolio").insert({
            "ticker": ticker,
            "quantity": round(quantity, 6),
            "avg_price": round(price, 4),
        }).execute()

    # Enregistrer la transaction
    client.table("transactions").insert({
        "type": "BUY",
        "ticker": ticker,
        "quantity": round(quantity, 6),
        "price": round(price, 4),
        "total": round(cost, 4),
    }).execute()

    return {"message": f"Achat de {quantity} {ticker} a ${price:.2f}"}


def sell_db(ticker: str, quantity: float, price: float) -> dict:
    """Vendre des actions/crypto (Supabase)."""
    client = get_client()
    portfolio = load_portfolio_db()

    if ticker not in portfolio["holdings"]:
        return {"error": f"Vous ne possedez pas de {ticker}"}

    holding = portfolio["holdings"][ticker]
    if quantity > holding["quantity"]:
        return {"error": f"Quantite insuffisante. Vous avez {holding['quantity']} {ticker}"}

    revenue = quantity * price
    new_cash = portfolio["cash"] + revenue

    # Mettre a jour le cash
    client.table("cash").update({"amount": round(new_cash, 4)}).eq("id", 1).execute()

    # Mettre a jour ou supprimer la position
    remaining = holding["quantity"] - quantity
    if remaining <= 0.0001:
        client.table("portfolio").delete().eq("ticker", ticker).execute()
    else:
        client.table("portfolio").update({
            "quantity": round(remaining, 6),
        }).eq("ticker", ticker).execute()

    # Enregistrer la transaction
    client.table("transactions").insert({
        "type": "SELL",
        "ticker": ticker,
        "quantity": round(quantity, 6),
        "price": round(price, 4),
        "total": round(revenue, 4),
    }).execute()

    return {"message": f"Vente de {quantity} {ticker} a ${price:.2f}"}


def get_portfolio_value_db(df: pd.DataFrame) -> dict:
    """Calcule la valeur actuelle du portefeuille (Supabase)."""
    portfolio = load_portfolio_db()
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


def load_transactions_db() -> list[dict]:
    """Charge l'historique des transactions depuis Supabase."""
    client = get_client()
    result = client.table("transactions").select("*").order("created_at", desc=True).execute()
    return result.data if result.data else []


def reset_portfolio_db() -> dict:
    """Reinitialise le portefeuille."""
    client = get_client()
    client.table("cash").update({"amount": 10000.0}).eq("id", 1).execute()
    client.table("portfolio").delete().neq("id", 0).execute()
    client.table("transactions").delete().neq("id", 0).execute()
    return {"cash": 10000.0, "holdings": {}}

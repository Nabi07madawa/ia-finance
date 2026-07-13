"""
Dashboard IA Finance — version autonome pour Streamlit Cloud.
Fonctionne directement avec les donnees et modeles, sans API.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="IA Finance Dashboard",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent / "models" / "saved"


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Charge les donnees. Si pas de fichier local, collecte et nettoie."""
    all_path = PROCESSED_DIR / "all_tickers_clean.csv"

    if not all_path.exists():
        st.info("Premiere execution : collecte et nettoyage des donnees...")
        from data.collect import collect_all
        from data.clean import run_pipeline
        collect_all()
        run_pipeline()

    return pd.read_csv(all_path, index_col=0, parse_dates=True)


def get_analysis(df: pd.DataFrame, ticker: str) -> dict:
    """Analyse technique d'un ticker."""
    data = df[df["Ticker"] == ticker].sort_index()
    latest = data.iloc[-1]
    current_price = float(latest["Close"])

    def pct_change(periods):
        if len(data) <= periods:
            return None
        past = float(data["Close"].iloc[-1 - periods])
        return (current_price - past) / past if past != 0 else None

    return {
        "current_price": current_price,
        "change_1d": pct_change(1),
        "change_7d": pct_change(5),
        "change_30d": pct_change(22),
        "ma_7": float(latest["MA_7"]) if pd.notna(latest.get("MA_7")) else None,
        "ma_20": float(latest["MA_20"]) if pd.notna(latest.get("MA_20")) else None,
        "ma_50": float(latest["MA_50"]) if pd.notna(latest.get("MA_50")) else None,
        "volatility_20d": float(latest["Volatility_20d"]) if pd.notna(latest.get("Volatility_20d")) else None,
        "rsi_14": float(latest["RSI_14"]) if pd.notna(latest.get("RSI_14")) else None,
        "volume_avg_20d": float(data["Volume"].tail(20).mean()) if "Volume" in data.columns else None,
    }


def predict_xgboost(df: pd.DataFrame, ticker: str):
    """Prediction XGBoost."""
    from models.xgboost_model import load_model, FEATURE_COLS
    try:
        model, scaler = load_model(ticker)
        data = df[df["Ticker"] == ticker].sort_index()
        features = data[FEATURE_COLS].iloc[-1:].values
        features_scaled = scaler.transform(features)
        return float(model.predict(features_scaled)[0])
    except Exception:
        return None


def predict_lstm(df: pd.DataFrame, ticker: str):
    """Prediction LSTM (necessite PyTorch)."""
    try:
        import torch
        from models.lstm_model import load_model, SEQUENCE_LENGTH
        model, scaler = load_model(ticker)
        data = df[df["Ticker"] == ticker].sort_index()
        close_prices = data["Close"].values[-SEQUENCE_LENGTH:].reshape(-1, 1)
        close_scaled = scaler.transform(close_prices)
        input_tensor = torch.FloatTensor(close_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()
        return float(scaler.inverse_transform(pred_scaled)[0][0])
    except ImportError:
        return None
    except Exception:
        return None


def predict_prophet(df: pd.DataFrame, ticker: str):
    """Prediction Prophet."""
    from models.prophet_model import load_model
    try:
        model = load_model(ticker)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        return float(forecast.iloc[-1]["yhat"])
    except Exception:
        return None


def forecast_prophet(df: pd.DataFrame, ticker: str, days: int):
    """Prevision Prophet sur N jours."""
    from models.prophet_model import load_model
    try:
        model = load_model(ticker)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        return forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        return None


# --- Chargement des donnees ---
df = load_data()
tickers = sorted(df["Ticker"].unique().tolist())

# --- Sidebar ---
st.sidebar.title("IA Finance")
st.sidebar.markdown("---")
selected_ticker = st.sidebar.selectbox("Ticker", tickers, index=0)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Prediction",
    "Prevision (Forecast)",
    "Comparaison",
    "Backtesting",
    "Portfolio",
    "Sentiment",
    "Evaluation Modeles",
    "Recommandations",
    "Alertes",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Modeles**")
for t in tickers:
    if t == selected_ticker:
        tf = t.replace("^", "IDX_")
        p = (MODELS_DIR / f"prophet_{tf}.pkl").exists()
        x = (MODELS_DIR / f"xgboost_{tf}.pkl").exists()
        l = (MODELS_DIR / f"lstm_{tf}.pt").exists()
        st.sidebar.caption(f"Prophet {'✅' if p else '❌'} | XGBoost {'✅' if x else '❌'} | LSTM {'✅' if l else '❌'}")


# === PAGE : Vue d'ensemble ===
if page == "Vue d'ensemble":
    st.title(f"📊 Analyse — {selected_ticker}")

    analysis = get_analysis(df, selected_ticker)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prix actuel", f"${analysis['current_price']:.2f}")
    with col2:
        c = analysis["change_1d"]
        st.metric("Variation 1j", f"{c*100:.2f}%" if c else "N/A", delta=f"{c*100:.2f}%" if c else None)
    with col3:
        c = analysis["change_7d"]
        st.metric("Variation 7j", f"{c*100:.2f}%" if c else "N/A", delta=f"{c*100:.2f}%" if c else None)
    with col4:
        c = analysis["change_30d"]
        st.metric("Variation 30j", f"{c*100:.2f}%" if c else "N/A", delta=f"{c*100:.2f}%" if c else None)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Moyennes mobiles**")
        st.write(f"MA 7 : {analysis['ma_7']:.2f}" if analysis["ma_7"] else "MA 7 : N/A")
        st.write(f"MA 20 : {analysis['ma_20']:.2f}" if analysis["ma_20"] else "MA 20 : N/A")
        st.write(f"MA 50 : {analysis['ma_50']:.2f}" if analysis["ma_50"] else "MA 50 : N/A")
    with col2:
        st.markdown("**Indicateurs**")
        st.write(f"RSI 14 : {analysis['rsi_14']:.2f}" if analysis["rsi_14"] else "RSI 14 : N/A")
        st.write(f"Volatilite 20j : {analysis['volatility_20d']:.4f}" if analysis["volatility_20d"] else "Volatilite : N/A")
    with col3:
        st.markdown("**Volume**")
        vol = analysis["volume_avg_20d"]
        st.write(f"Volume moyen 20j : {vol:,.0f}" if vol else "N/A")

    st.markdown("---")

    days = st.slider("Historique (jours)", 30, 365, 180)
    ticker_data = df[df["Ticker"] == selected_ticker].sort_index().tail(days)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ticker_data.index,
        open=ticker_data["Open"],
        high=ticker_data["High"],
        low=ticker_data["Low"],
        close=ticker_data["Close"],
        name="OHLC",
    ))
    fig.update_layout(
        title=f"Historique {selected_ticker} — {days} derniers jours",
        xaxis_title="Date", yaxis_title="Prix ($)",
        xaxis_rangeslider_visible=False, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_vol = px.bar(ticker_data.reset_index(), x="Date", y="Volume", title="Volume")
    fig_vol.update_layout(height=200, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)


# === PAGE : Prediction ===
elif page == "Prediction":
    st.title(f"🤖 Prediction — {selected_ticker}")

    with st.spinner("Calcul des predictions..."):
        preds = {}
        p = predict_prophet(df, selected_ticker)
        if p: preds["Prophet"] = p
        x = predict_xgboost(df, selected_ticker)
        if x: preds["XGBoost"] = x
        l = predict_lstm(df, selected_ticker)
        if l: preds["LSTM"] = l

    if preds:
        ensemble = np.mean(list(preds.values()))
        st.markdown("### Prediction Ensemble")
        st.metric("Prix predit (moyenne)", f"${ensemble:.2f}")
        st.markdown("---")

        st.markdown("### Predictions individuelles")
        cols = st.columns(len(preds))
        for i, (name, val) in enumerate(preds.items()):
            with cols[i]:
                st.metric(name, f"${val:.2f}")

        fig = go.Figure()
        colors = ["#636EFA", "#00CC96", "#EF553B"]
        fig.add_trace(go.Bar(
            x=list(preds.keys()), y=list(preds.values()),
            marker_color=colors[:len(preds)],
            text=[f"${v:.2f}" for v in preds.values()],
            textposition="outside",
        ))
        fig.add_hline(y=ensemble, line_dash="dash", line_color="orange",
                      annotation_text=f"Ensemble: ${ensemble:.2f}")
        fig.update_layout(title="Comparaison des predictions", yaxis_title="Prix ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucun modele disponible. Lancez l'entrainement d'abord.")


# === PAGE : Prevision ===
elif page == "Prevision (Forecast)":
    st.title(f"📈 Prevision Prophet — {selected_ticker}")

    days = st.slider("Horizon de prevision (jours)", 7, 180, 30)

    with st.spinner("Calcul de la prevision..."):
        fc = forecast_prophet(df, selected_ticker, days)

    if fc is not None:
        hist_data = df[df["Ticker"] == selected_ticker].sort_index().tail(90)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data["Close"],
                                 mode="lines", name="Historique", line=dict(color="#636EFA")))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"],
                                 mode="lines", name="Prevision", line=dict(color="#00CC96", dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(0, 204, 150, 0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalle de confiance",
        ))
        fig.update_layout(title=f"Prevision {selected_ticker} — {days} jours",
                          xaxis_title="Date", yaxis_title="Prix ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Donnees de prevision"):
            st.dataframe(fc.rename(columns={"ds": "Date", "yhat": "Predit",
                                            "yhat_lower": "Borne basse", "yhat_upper": "Borne haute"}),
                         use_container_width=True)
    else:
        st.warning("Modele Prophet non disponible pour ce ticker.")


# === PAGE : Comparaison ===
elif page == "Comparaison":
    st.title("⚖️ Comparaison de tickers")

    selected_tickers = st.multiselect("Tickers a comparer", tickers, default=tickers[:3])
    days = st.slider("Periode (jours)", 7, 365, 30)

    if selected_tickers:
        results = []
        for ticker in selected_tickers:
            ticker_data = df[df["Ticker"] == ticker].sort_index().tail(days)
            if len(ticker_data) < 2:
                continue
            start_price = float(ticker_data.iloc[0]["Close"])
            end_price = float(ticker_data.iloc[-1]["Close"])
            perf = (end_price - start_price) / start_price
            vol = float(ticker_data["Close"].pct_change().std())
            results.append({"ticker": ticker, "start": start_price, "end": end_price, "perf": perf, "vol": vol})

        results.sort(key=lambda x: x["perf"], reverse=True)

        cols = st.columns(len(results))
        for i, item in enumerate(results):
            with cols[i]:
                st.metric(item["ticker"], f"${item['end']:.2f}", delta=f"{item['perf']*100:+.2f}%")
                st.caption(f"Volatilite: {item['vol']:.4f}")

        st.markdown("---")

        fig = go.Figure()
        for ticker in selected_tickers:
            ticker_data = df[df["Ticker"] == ticker].sort_index().tail(days)
            base = ticker_data["Close"].iloc[0]
            normalized = (ticker_data["Close"] / base - 1) * 100
            fig.add_trace(go.Scatter(x=ticker_data.index, y=normalized, mode="lines", name=ticker))

        fig.update_layout(title=f"Performance relative — {days} jours",
                          xaxis_title="Date", yaxis_title="Variation (%)",
                          height=500, hovermode="x unified")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        df_compare = pd.DataFrame(results).rename(columns={
            "ticker": "Ticker", "start": "Prix debut ($)", "end": "Prix fin ($)",
            "perf": "Performance (%)", "vol": "Volatilite"
        })
        df_compare["Performance (%)"] = df_compare["Performance (%)"] * 100
        st.dataframe(df_compare, use_container_width=True, hide_index=True)
    else:
        st.info("Selectionnez au moins un ticker.")


# === PAGE : Backtesting ===
elif page == "Backtesting":
    st.title("💰 Backtesting — Simulation de portefeuille")
    st.markdown("Simule ce qui se serait passe si vous aviez suivi les predictions du modele.")

    from models.backtest import backtest_strategy

    col1, col2 = st.columns(2)
    with col1:
        bt_ticker = st.selectbox("Ticker", tickers, key="bt_ticker")
    with col2:
        bt_capital = st.number_input("Capital initial ($)", min_value=100, value=10000, step=1000)

    if st.button("Lancer le backtest"):
        with st.spinner(f"Simulation en cours pour {bt_ticker}..."):
            result = backtest_strategy(df, bt_ticker, initial_capital=float(bt_capital))

        if result:
            # Metriques principales
            st.markdown("### Resultats")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Capital final", f"${result['final_value']:,.2f}",
                          delta=f"{result['strategy_return']:+.1f}%")
            with col2:
                st.metric("Buy & Hold", f"{result['buy_hold_return']:+.1f}%")
            with col3:
                st.metric("Surperformance", f"{result['outperformance']:+.1f}%",
                          delta=f"{result['outperformance']:+.1f}%")
            with col4:
                st.metric("Win Rate", f"{result['win_rate']:.0f}%")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trades", result["total_trades"])
            with col2:
                st.metric("Max Drawdown", f"{result['max_drawdown']:.1f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")

            st.markdown("---")

            # Graphique evolution du portefeuille
            st.markdown("### Evolution du portefeuille")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["dates"], y=result["portfolio_values"],
                name="Strategie IA", line=dict(color="#00CC96"),
            ))
            fig.add_trace(go.Scatter(
                x=result["dates"], y=result["buy_hold_values"],
                name="Buy & Hold", line=dict(color="#636EFA", dash="dash"),
            ))
            fig.add_hline(y=bt_capital, line_dash="dot", line_color="gray",
                          annotation_text=f"Capital initial: ${bt_capital:,}")
            fig.update_layout(
                yaxis_title="Valeur ($)", xaxis_title="Date", height=450,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Trades
            if result["trades"]:
                st.markdown("### Historique des trades")
                df_trades = pd.DataFrame(result["trades"])
                df_trades["date"] = [result["dates"][t["day"]] if t["day"] < len(result["dates"]) else "N/A"
                                     for t in result["trades"]]
                df_trades["price"] = df_trades["price"].round(2)
                st.dataframe(df_trades[["date", "action", "price"]].rename(columns={
                    "date": "Date", "action": "Action", "price": "Prix ($)"
                }), use_container_width=True, hide_index=True)
        else:
            st.warning(f"Pas assez de donnees pour backtester {bt_ticker}.")

    # Backtest multi-tickers
    st.markdown("---")
    st.markdown("### Comparaison multi-tickers")

    if st.button("Backtester tous les tickers"):
        with st.spinner("Backtesting en cours sur tous les tickers..."):
            from models.backtest import backtest_multiple
            all_results = backtest_multiple(df, initial_capital=float(bt_capital))

        if all_results:
            df_bt = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Strategie (%)": r["strategy_return"],
                "Buy&Hold (%)": r["buy_hold_return"],
                "Surperformance (%)": r["outperformance"],
                "Win Rate (%)": r["win_rate"],
                "Trades": r["total_trades"],
                "Max Drawdown (%)": r["max_drawdown"],
            } for r in all_results])
            df_bt = df_bt.sort_values("Surperformance (%)", ascending=False)
            st.dataframe(df_bt, use_container_width=True, hide_index=True)

            # Graphique comparatif
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Strategie IA", x=df_bt["Ticker"], y=df_bt["Strategie (%)"],
                                 marker_color="#00CC96"))
            fig.add_trace(go.Bar(name="Buy & Hold", x=df_bt["Ticker"], y=df_bt["Buy&Hold (%)"],
                                 marker_color="#636EFA"))
            fig.update_layout(title="Strategie IA vs Buy & Hold",
                              yaxis_title="Rendement (%)", barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)


# === PAGE : Portfolio ===
elif page == "Portfolio":
    st.title("💼 Portfolio Manager")

    from api.portfolio import (
        load_portfolio, buy, sell, get_portfolio_value, reset_portfolio, load_transactions
    )

    # Valeur du portefeuille
    pf_value = get_portfolio_value(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Valeur totale", f"${pf_value['total_portfolio_value']:,.2f}")
    with col2:
        st.metric("Cash disponible", f"${pf_value['cash']:,.2f}")
    with col3:
        st.metric("Investissements", f"${pf_value['total_holdings_value']:,.2f}")
    with col4:
        st.metric("P&L total", f"${pf_value['total_pnl']:,.2f}",
                  delta=f"{pf_value['total_pnl_pct']:+.2f}%")

    st.markdown("---")

    # Acheter / Vendre
    st.markdown("### Transaction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tx_action = st.selectbox("Action", ["Acheter", "Vendre"])
    with col2:
        tx_ticker = st.selectbox("Ticker", tickers, key="tx_ticker")
    with col3:
        tx_qty = st.number_input("Quantite", min_value=0.01, value=1.0, step=0.1)
    with col4:
        # Prix actuel
        ticker_data = df[df["Ticker"] == tx_ticker].sort_index()
        current_price = float(ticker_data["Close"].iloc[-1]) if not ticker_data.empty else 0
        tx_price = st.number_input("Prix ($)", min_value=0.01, value=round(current_price, 2))

    if st.button("Executer la transaction"):
        if tx_action == "Acheter":
            result = buy(tx_ticker, tx_qty, tx_price)
        else:
            result = sell(tx_ticker, tx_qty, tx_price)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(result["message"])
            st.rerun()

    st.markdown("---")

    # Holdings
    st.markdown("### Positions actuelles")
    if pf_value["holdings"]:
        df_holdings = pd.DataFrame(pf_value["holdings"])
        df_holdings = df_holdings.rename(columns={
            "ticker": "Ticker", "quantity": "Quantite", "avg_price": "Prix moyen ($)",
            "current_price": "Prix actuel ($)", "invested": "Investi ($)",
            "current_value": "Valeur ($)", "pnl": "P&L ($)", "pnl_pct": "P&L (%)",
        })
        st.dataframe(df_holdings, use_container_width=True, hide_index=True)

        # Graphique repartition
        fig = px.pie(df_holdings, values="Valeur ($)", names="Ticker",
                     title="Repartition du portefeuille")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune position. Achetez des actifs pour commencer.")

    st.markdown("---")

    # Historique transactions
    st.markdown("### Historique des transactions")
    transactions = load_transactions()
    if transactions:
        df_tx = pd.DataFrame(transactions)
        df_tx = df_tx.rename(columns={
            "type": "Type", "ticker": "Ticker", "quantity": "Quantite",
            "price": "Prix ($)", "total": "Total ($)", "date": "Date",
        })
        st.dataframe(df_tx[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Aucune transaction.")

    # Reset
    st.markdown("---")
    if st.button("Reinitialiser le portefeuille ($10,000)"):
        reset_portfolio()
        st.success("Portefeuille reinitialise.")
        st.rerun()


# === PAGE : Sentiment ===
elif page == "Sentiment":
    st.title("📰 Analyse de Sentiment — News")
    st.markdown("Analyse le sentiment des dernieres news financieres pour predire la tendance.")

    from api.sentiment import analyze_ticker_sentiment

    st.markdown(f"### Sentiment pour {selected_ticker}")

    if st.button("Analyser le sentiment"):
        with st.spinner(f"Analyse des news de {selected_ticker}..."):
            result = analyze_ticker_sentiment(selected_ticker)

        if result["news_count"] == 0:
            st.warning(f"Aucune news trouvee pour {selected_ticker}.")
        else:
            # Score global
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_color = {"POSITIF": "green", "NEGATIF": "red", "NEUTRE": "orange"}
                color = sentiment_color.get(result["sentiment_label"], "gray")
                st.markdown(f"**Sentiment global:** :{color}[{result['sentiment_label']}]")
            with col2:
                st.metric("Polarite moyenne", f"{result['avg_polarity']:.3f}")
            with col3:
                signal_emoji = {
                    "ACHAT FORT": "🟢🟢", "ACHAT": "🟢",
                    "VENTE FORT": "🔴🔴", "VENTE": "🔴",
                    "NEUTRE": "🟡",
                }
                st.metric("Signal", f"{signal_emoji.get(result['signal'], '')} {result['signal']}")

            st.markdown("---")

            # Articles
            st.markdown(f"### Dernieres news ({result['news_count']} articles)")
            for article in result["articles"]:
                sentiment_icon = {"POSITIF": "🟢", "NEGATIF": "🔴", "NEUTRE": "🟡"}
                icon = sentiment_icon.get(article["sentiment"], "⚪")
                st.markdown(f"{icon} **{article['title']}**")
                st.caption(f"{article['publisher']} | Polarite: {article['polarity']}")
                st.markdown("---")

            # Graphique des polarites
            if result["articles"]:
                polarities = [a["polarity"] for a in result["articles"]]
                titles = [a["title"][:40] + "..." for a in result["articles"]]

                fig = go.Figure()
                colors = ["green" if p > 0 else "red" if p < 0 else "gray" for p in polarities]
                fig.add_trace(go.Bar(x=titles, y=polarities, marker_color=colors))
                fig.update_layout(title="Polarite par article", yaxis_title="Polarite",
                                  height=350, xaxis_tickangle=-45)
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

    # Multi-ticker
    st.markdown("---")
    st.markdown("### Comparaison multi-tickers")
    selected_for_sentiment = st.multiselect("Tickers a analyser", tickers, default=tickers[:4],
                                            key="sentiment_tickers")
    if st.button("Analyser tous"):
        with st.spinner("Analyse en cours..."):
            from api.sentiment import analyze_multiple_tickers
            all_sentiments = analyze_multiple_tickers(selected_for_sentiment)

        if all_sentiments:
            data_sent = []
            for s in all_sentiments:
                data_sent.append({
                    "Ticker": s["ticker"],
                    "Sentiment": s["sentiment_label"],
                    "Polarite": s["avg_polarity"],
                    "Signal": s["signal"],
                    "Articles": s["news_count"],
                })
            df_sent = pd.DataFrame(data_sent)
            st.dataframe(df_sent, use_container_width=True, hide_index=True)

            # Graphique
            fig = go.Figure()
            colors = ["green" if p > 0 else "red" if p < 0 else "gray"
                      for p in df_sent["Polarite"]]
            fig.add_trace(go.Bar(x=df_sent["Ticker"], y=df_sent["Polarite"],
                                 marker_color=colors, text=df_sent["Signal"],
                                 textposition="outside"))
            fig.update_layout(title="Sentiment par ticker", yaxis_title="Polarite", height=400)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)


# === PAGE : Evaluation Modeles ===
elif page == "Evaluation Modeles":
    st.title("📊 Evaluation des Modeles")
    st.markdown("Compare la precision des modeles Prophet, XGBoost et LSTM.")

    from models.evaluate import evaluate_xgboost, evaluate_prophet, evaluate_lstm

    eval_ticker = st.selectbox("Ticker a evaluer", tickers, key="eval_ticker")

    models_to_eval = st.multiselect(
        "Modeles a evaluer",
        ["XGBoost", "Prophet", "LSTM"],
        default=["XGBoost", "Prophet"],
        key="eval_models",
    )

    if st.button("Lancer l'evaluation"):
        results = []

        for model_name in models_to_eval:
            with st.spinner(f"Evaluation de {model_name}..."):
                if model_name == "XGBoost":
                    r = evaluate_xgboost(df, eval_ticker)
                elif model_name == "Prophet":
                    r = evaluate_prophet(df, eval_ticker)
                elif model_name == "LSTM":
                    r = evaluate_lstm(df, eval_ticker)
                else:
                    r = None

                if r:
                    results.append(r)
                else:
                    st.warning(f"{model_name} non disponible pour {eval_ticker}")

        if results:
            # Tableau comparatif
            st.markdown("### Metriques de precision")
            st.markdown("""
            - **RMSE** : Erreur quadratique moyenne (plus bas = meilleur)
            - **MAE** : Erreur absolue moyenne (plus bas = meilleur)
            - **MAPE** : Erreur en pourcentage (plus bas = meilleur)
            - **R2** : Coefficient de determination (plus proche de 1 = meilleur)
            """)

            df_metrics = pd.DataFrame([{
                "Modele": r["model"],
                "RMSE": r["rmse"],
                "MAE": r["mae"],
                "MAPE (%)": r["mape"],
                "R2": r["r2"],
                "Jours test": r["test_size"],
            } for r in results])

            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Meilleur modele
            best = min(results, key=lambda x: x["mape"])
            st.success(f"Meilleur modele pour {eval_ticker} : **{best['model']}** (MAPE: {best['mape']:.2f}%)")

            # Graphique comparatif des metriques
            st.markdown("### Comparaison visuelle")

            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                for r in results:
                    fig.add_trace(go.Bar(name=r["model"], x=["RMSE", "MAE"], y=[r["rmse"], r["mae"]]))
                fig.update_layout(title="RMSE & MAE (plus bas = meilleur)", barmode="group", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                for r in results:
                    fig.add_trace(go.Bar(name=r["model"], x=["MAPE (%)", "R2"], y=[r["mape"], r["r2"]]))
                fig.update_layout(title="MAPE & R2", barmode="group", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Graphique predictions vs reel
            st.markdown("### Predictions vs Prix reel")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results[0]["dates"], y=results[0]["y_test"],
                mode="lines", name="Prix reel", line=dict(color="black", width=2),
            ))

            colors = {"XGBoost": "blue", "Prophet": "green", "LSTM": "red"}
            for r in results:
                fig.add_trace(go.Scatter(
                    x=r["dates"], y=r["y_pred"],
                    mode="lines", name=f"{r['model']} prediction",
                    line=dict(color=colors.get(r["model"], "gray"), dash="dash"),
                ))

            fig.update_layout(
                title=f"Predictions vs Reel — {eval_ticker}",
                xaxis_title="Date", yaxis_title="Prix ($)",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Erreur par modele dans le temps
            st.markdown("### Erreur de prediction dans le temps")
            fig = go.Figure()
            for r in results:
                errors = [abs(t - p) for t, p in zip(r["y_test"], r["y_pred"])]
                fig.add_trace(go.Scatter(
                    x=r["dates"], y=errors,
                    mode="lines", name=f"{r['model']} erreur absolue",
                    line=dict(color=colors.get(r["model"], "gray")),
                ))
            fig.update_layout(title="Erreur absolue par jour", xaxis_title="Date",
                              yaxis_title="Erreur ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)


# === PAGE : Recommandations ===
elif page == "Recommandations":
    st.title("🎯 Recommandations Automatiques")
    st.markdown("Combine tous les indicateurs techniques pour generer des signaux d'achat/vente.")

    from api.recommendations import get_all_recommendations, analyze_ticker_signals

    if st.button("Generer les recommandations", type="primary"):
        with st.spinner("Analyse de tous les tickers..."):
            recommendations = get_all_recommendations(df, tickers)

        if recommendations:
            # Resume global
            recommendations = [r for r in recommendations if "current_price" in r]
            achats = [r for r in recommendations if "ACHAT" in r["recommendation"]]
            ventes = [r for r in recommendations if "VENTE" in r["recommendation"]]
            neutres = [r for r in recommendations if r["recommendation"] == "NEUTRE"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signaux ACHAT", len(achats))
            with col2:
                st.metric("Signaux VENTE", len(ventes))
            with col3:
                st.metric("NEUTRE", len(neutres))

            st.markdown("---")

            # Tableau des recommandations
            st.markdown("### Classement des actifs")
            valid_recos = [r for r in recommendations if "current_price" in r]
            df_reco = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Prix": f"${r['current_price']:.2f}",
                "Var. 1j (%)": f"{r['change_1d']:+.2f}%",
                "Recommandation": r["recommendation"],
                "Confiance (%)": r["confidence"],
                "Score": r["net_score"],
            } for r in valid_recos])

            st.dataframe(df_reco, use_container_width=True, hide_index=True)

            # Top achats
            if achats:
                st.markdown("### Top Achats")
                for r in achats[:5]:
                    emoji = "🟢🟢" if "FORT" in r["recommendation"] else "🟢"
                    st.markdown(f"{emoji} **{r['ticker']}** — {r['recommendation']} "
                                f"(confiance: {r['confidence']:.0f}%) | Prix: ${r['current_price']:.2f}")

            # Top ventes
            if ventes:
                st.markdown("### Top Ventes")
                for r in ventes[:5]:
                    emoji = "🔴🔴" if "FORT" in r["recommendation"] else "🔴"
                    st.markdown(f"{emoji} **{r['ticker']}** — {r['recommendation']} "
                                f"(confiance: {r['confidence']:.0f}%) | Prix: ${r['current_price']:.2f}")

            # Graphique score par ticker
            st.markdown("### Score net par ticker")
            fig = go.Figure()
            colors = ["green" if r["net_score"] > 0 else "red" if r["net_score"] < 0 else "gray"
                      for r in recommendations]
            fig.add_trace(go.Bar(
                x=[r["ticker"] for r in recommendations],
                y=[r["net_score"] for r in recommendations],
                marker_color=colors,
                text=[r["recommendation"] for r in recommendations],
                textposition="outside",
            ))
            fig.update_layout(title="Score de recommandation (positif = achat, negatif = vente)",
                              yaxis_title="Score", height=400)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    # Detail par ticker
    st.markdown("---")
    st.markdown("### Detail des signaux par ticker")
    detail_ticker = st.selectbox("Voir les signaux de", tickers, key="reco_detail")

    if st.button("Analyser ce ticker"):
        with st.spinner(f"Analyse de {detail_ticker}..."):
            result = analyze_ticker_signals(df, detail_ticker)

        if result["signals"]:
            # Recommandation principale
            reco_color = {"ACHAT FORT": "green", "ACHAT": "green", "VENTE FORT": "red",
                          "VENTE": "red", "NEUTRE": "orange"}
            color = reco_color.get(result["recommendation"], "gray")
            st.markdown(f"## :{color}[{result['recommendation']}] — Confiance: {result['confidence']:.0f}%")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prix actuel", f"${result['current_price']:.2f}")
            with col2:
                st.metric("Variation 1j", f"{result['change_1d']:+.2f}%")
            with col3:
                st.metric("Score net", f"{result['net_score']:+.3f}")

            # Tableau des signaux
            st.markdown("### Indicateurs techniques")
            df_signals = pd.DataFrame([{
                "Indicateur": s["indicator"],
                "Valeur": s["value"],
                "Signal": s["direction"],
                "Force": f"{s['strength']:.2f}",
            } for s in result["signals"]])
            st.dataframe(df_signals, use_container_width=True, hide_index=True)

            # Jauge de signaux
            fig = go.Figure()
            for i, s in enumerate(result["signals"]):
                color = "green" if s["direction"] == "ACHAT" else "red" if s["direction"] == "VENTE" else "gray"
                value = s["strength"] if s["direction"] == "ACHAT" else -s["strength"] if s["direction"] == "VENTE" else 0
                fig.add_trace(go.Bar(
                    x=[value], y=[s["indicator"]], orientation="h",
                    marker_color=color, showlegend=False,
                ))
            fig.update_layout(title="Force des signaux par indicateur",
                              xaxis_title="Force (+ = achat, - = vente)",
                              height=300, xaxis=dict(range=[-1, 1]))
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Pas assez de donnees pour generer des signaux.")


# === PAGE : Alertes ===
elif page == "Alertes":
    st.title("🔔 Alertes de prix")

    from api.alerts import load_alerts, add_alert, remove_alert, check_alerts, get_alert_history, reset_alert

    # Creer une alerte
    st.markdown("### Creer une alerte")
    col1, col2, col3 = st.columns(3)
    with col1:
        alert_ticker = st.selectbox("Ticker", tickers, key="alert_ticker")
    with col2:
        alert_condition = st.selectbox("Condition", [
            "price_above", "price_below", "change_above", "change_below"
        ], format_func=lambda x: {
            "price_above": "Prix depasse (>)",
            "price_below": "Prix descend sous (<)",
            "change_above": "Hausse de X%",
            "change_below": "Baisse de X%",
        }[x])
    with col3:
        alert_threshold = st.number_input("Seuil", min_value=0.0, value=100.0, step=1.0)

    alert_email = st.text_input("Email (optionnel)", placeholder="votre@email.com")

    if st.button("Creer l'alerte"):
        alert = add_alert(alert_ticker, alert_condition, alert_threshold,
                         alert_email if alert_email else None)
        st.success(f"Alerte creee ! (ID: {alert['id']})")

    st.markdown("---")

    # Verifier les alertes
    if st.button("Verifier les alertes maintenant"):
        triggered = check_alerts(df)
        if triggered:
            for a in triggered:
                st.warning(f"🚨 {a.get('message', a['ticker'])}")
        else:
            st.info("Aucune alerte declenchee.")

    st.markdown("---")

    # Alertes actives
    st.markdown("### Alertes actives")
    alerts = load_alerts()
    if alerts:
        for alert in alerts:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                status = "🔴 Declenchee" if alert["triggered"] else "🟢 Active"
                condition_text = {
                    "price_above": f"Prix > ${alert['threshold']}",
                    "price_below": f"Prix < ${alert['threshold']}",
                    "change_above": f"Hausse > {alert['threshold']}%",
                    "change_below": f"Baisse > {alert['threshold']}%",
                }[alert["condition"]]
                st.write(f"{status} | **{alert['ticker']}** — {condition_text}")
            with col2:
                if alert["triggered"]:
                    if st.button("Reactiver", key=f"reset_{alert['id']}"):
                        reset_alert(alert["id"])
                        st.rerun()
            with col3:
                if st.button("Supprimer", key=f"del_{alert['id']}"):
                    remove_alert(alert["id"])
                    st.rerun()
    else:
        st.info("Aucune alerte configuree.")

    st.markdown("---")

    # Historique
    st.markdown("### Historique des alertes")
    history = get_alert_history()
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune alerte declenchee pour le moment.")

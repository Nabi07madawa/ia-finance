import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IA Finance Dashboard",
    page_icon="📈",
    layout="wide",
)


def api_get(endpoint: str, params: dict = None):
    """Appel GET a l'API."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Erreur API ({resp.status_code}): {resp.json().get('detail', 'Erreur inconnue')}")
    except requests.ConnectionError:
        st.error("Impossible de se connecter a l'API. Verifiez que le serveur est lance (uvicorn api.main:app).")
    return None


# --- Sidebar ---
st.sidebar.title("IA Finance")
st.sidebar.markdown("---")

tickers_data = api_get("/tickers")
if not tickers_data:
    st.stop()

tickers = tickers_data["tickers"]
selected_ticker = st.sidebar.selectbox("Ticker", tickers, index=0)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Prediction",
    "Prevision (Forecast)",
    "Comparaison",
])

st.sidebar.markdown("---")
models_data = api_get("/models/status")
if models_data:
    st.sidebar.markdown("**Modeles disponibles**")
    for m in models_data["models"]:
        if m["ticker"] == selected_ticker:
            status = ""
            status += "Prophet " + ("✅" if m["prophet"] else "❌") + " | "
            status += "XGBoost " + ("✅" if m["xgboost"] else "❌") + " | "
            status += "LSTM " + ("✅" if m["lstm"] else "❌")
            st.sidebar.caption(status)


# === PAGE : Vue d'ensemble ===
if page == "Vue d'ensemble":
    st.title(f"📊 Analyse — {selected_ticker}")

    analysis = api_get(f"/analysis/{selected_ticker}")
    if analysis:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prix actuel", f"${analysis['current_price']:.2f}")
        with col2:
            change_1d = analysis.get("change_1d")
            st.metric("Variation 1j", f"{change_1d*100:.2f}%" if change_1d else "N/A",
                      delta=f"{change_1d*100:.2f}%" if change_1d else None)
        with col3:
            change_7d = analysis.get("change_7d")
            st.metric("Variation 7j", f"{change_7d*100:.2f}%" if change_7d else "N/A",
                      delta=f"{change_7d*100:.2f}%" if change_7d else None)
        with col4:
            change_30d = analysis.get("change_30d")
            st.metric("Variation 30j", f"{change_30d*100:.2f}%" if change_30d else "N/A",
                      delta=f"{change_30d*100:.2f}%" if change_30d else None)

        st.markdown("---")

        # Indicateurs techniques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Moyennes mobiles**")
            st.write(f"MA 7 : {analysis.get('ma_7', 'N/A')}")
            st.write(f"MA 20 : {analysis.get('ma_20', 'N/A')}")
            st.write(f"MA 50 : {analysis.get('ma_50', 'N/A')}")
        with col2:
            st.markdown("**Indicateurs**")
            st.write(f"RSI 14 : {analysis.get('rsi_14', 'N/A')}")
            st.write(f"Volatilite 20j : {analysis.get('volatility_20d', 'N/A')}")
        with col3:
            st.markdown("**Volume**")
            vol = analysis.get("volume_avg_20d")
            st.write(f"Volume moyen 20j : {vol:,.0f}" if vol else "N/A")

        st.markdown("---")

    # Graphique historique
    days = st.slider("Historique (jours)", 30, 365, 180)
    history = api_get(f"/analysis/{selected_ticker}/history", {"days": days})

    if history and history["history"]:
        df_hist = pd.DataFrame(history["history"])
        df_hist["date"] = pd.to_datetime(df_hist["date"])

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_hist["date"],
            open=df_hist["open"],
            high=df_hist["high"],
            low=df_hist["low"],
            close=df_hist["close"],
            name="OHLC",
        ))
        fig.update_layout(
            title=f"Historique {selected_ticker} — {days} derniers jours",
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            xaxis_rangeslider_visible=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        fig_vol = px.bar(df_hist, x="date", y="volume", title="Volume")
        fig_vol.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)


# === PAGE : Prediction ===
elif page == "Prediction":
    st.title(f"🤖 Prediction — {selected_ticker}")

    st.markdown("Prediction du prochain prix de cloture avec les 3 modeles.")

    ensemble = api_get(f"/predict/ensemble/{selected_ticker}")
    if ensemble:
        st.markdown("### Prediction Ensemble")
        st.metric("Prix predit (moyenne)", f"${ensemble['ensemble_prediction']:.2f}")

        st.markdown("---")
        st.markdown("### Predictions individuelles")

        col1, col2, col3 = st.columns(3)
        preds = ensemble["individual_predictions"]

        with col1:
            if "prophet" in preds:
                st.metric("Prophet", f"${preds['prophet']:.2f}")
            else:
                st.metric("Prophet", "N/A")
        with col2:
            if "xgboost" in preds:
                st.metric("XGBoost", f"${preds['xgboost']:.2f}")
            else:
                st.metric("XGBoost", "N/A")
        with col3:
            if "lstm" in preds:
                st.metric("LSTM", f"${preds['lstm']:.2f}")
            else:
                st.metric("LSTM", "N/A")

        # Graphique comparatif
        if preds:
            fig = go.Figure()
            models = list(preds.keys())
            values = list(preds.values())
            colors = ["#636EFA", "#00CC96", "#EF553B"]

            fig.add_trace(go.Bar(
                x=models,
                y=values,
                marker_color=colors[:len(models)],
                text=[f"${v:.2f}" for v in values],
                textposition="outside",
            ))
            fig.add_hline(
                y=ensemble["ensemble_prediction"],
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Ensemble: ${ensemble['ensemble_prediction']:.2f}",
            )
            fig.update_layout(
                title="Comparaison des predictions par modele",
                yaxis_title="Prix predit ($)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        if ensemble.get("errors"):
            st.warning(f"Erreurs sur certains modeles : {ensemble['errors']}")


# === PAGE : Prevision (Forecast) ===
elif page == "Prevision (Forecast)":
    st.title(f"📈 Prevision Prophet — {selected_ticker}")

    days = st.slider("Horizon de prevision (jours)", 7, 180, 30)

    forecast_data = api_get(f"/forecast/{selected_ticker}", {"days": days})
    if forecast_data:
        df_fc = pd.DataFrame(forecast_data["forecast"])
        df_fc["date"] = pd.to_datetime(df_fc["date"])

        # Historique recent pour contexte
        history = api_get(f"/analysis/{selected_ticker}/history", {"days": 90})

        fig = go.Figure()

        # Historique
        if history and history["history"]:
            df_hist = pd.DataFrame(history["history"])
            df_hist["date"] = pd.to_datetime(df_hist["date"])
            fig.add_trace(go.Scatter(
                x=df_hist["date"],
                y=df_hist["close"],
                mode="lines",
                name="Historique",
                line=dict(color="#636EFA"),
            ))

        # Prevision
        fig.add_trace(go.Scatter(
            x=df_fc["date"],
            y=df_fc["predicted"],
            mode="lines",
            name="Prevision",
            line=dict(color="#00CC96", dash="dash"),
        ))

        # Intervalle de confiance
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fc["date"], df_fc["date"][::-1]]),
            y=pd.concat([df_fc["upper"], df_fc["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle de confiance",
        ))

        fig.update_layout(
            title=f"Prevision {selected_ticker} — {days} prochains jours",
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tableau des previsions
        with st.expander("Voir les donnees de prevision"):
            st.dataframe(df_fc.rename(columns={
                "date": "Date",
                "predicted": "Prix predit",
                "lower": "Borne basse",
                "upper": "Borne haute",
            }), use_container_width=True)


# === PAGE : Comparaison ===
elif page == "Comparaison":
    st.title("⚖️ Comparaison de tickers")

    selected_tickers = st.multiselect("Tickers a comparer", tickers, default=tickers[:3])
    days = st.slider("Periode (jours)", 7, 365, 30)

    if selected_tickers:
        tickers_str = ",".join(selected_tickers)
        compare_data = api_get("/compare", {"tickers": tickers_str, "days": days})

        if compare_data and compare_data["comparison"]:
            # Metriques
            cols = st.columns(len(compare_data["comparison"]))
            for i, item in enumerate(compare_data["comparison"]):
                with cols[i]:
                    perf = item["performance"]
                    st.metric(
                        item["ticker"],
                        f"${item['end_price']:.2f}",
                        delta=f"{perf:+.2f}%",
                    )
                    st.caption(f"Volatilite: {item['volatility']:.4f}")

            st.markdown("---")

            # Graphique d'evolution normalisee
            fig = go.Figure()
            for ticker in selected_tickers:
                hist = api_get(f"/analysis/{ticker}/history", {"days": days})
                if hist and hist["history"]:
                    df_t = pd.DataFrame(hist["history"])
                    df_t["date"] = pd.to_datetime(df_t["date"])
                    base_price = df_t["close"].iloc[0]
                    df_t["normalized"] = (df_t["close"] / base_price - 1) * 100
                    fig.add_trace(go.Scatter(
                        x=df_t["date"],
                        y=df_t["normalized"],
                        mode="lines",
                        name=ticker,
                    ))

            fig.update_layout(
                title=f"Performance relative — {days} derniers jours",
                xaxis_title="Date",
                yaxis_title="Variation (%)",
                height=500,
                hovermode="x unified",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

            # Tableau recapitulatif
            st.markdown("### Tableau comparatif")
            df_compare = pd.DataFrame(compare_data["comparison"])
            df_compare = df_compare.rename(columns={
                "ticker": "Ticker",
                "start_price": "Prix debut ($)",
                "end_price": "Prix fin ($)",
                "performance": "Performance (%)",
                "volatility": "Volatilite",
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
    else:
        st.info("Selectionnez au moins un ticker pour comparer.")

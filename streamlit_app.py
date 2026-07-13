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

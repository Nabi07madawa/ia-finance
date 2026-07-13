"""
Systeme d'alertes pour les variations de prix.
Surveille les tickers et declenche des alertes quand un seuil est atteint.
"""
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import pandas as pd

ALERTS_FILE = Path(__file__).parent.parent / "data" / "alerts.json"
ALERTS_LOG = Path(__file__).parent.parent / "data" / "alerts_history.json"


def load_alerts() -> list[dict]:
    """Charge les alertes configurees."""
    if ALERTS_FILE.exists():
        return json.loads(ALERTS_FILE.read_text())
    return []


def save_alerts(alerts: list[dict]) -> None:
    """Sauvegarde les alertes."""
    ALERTS_FILE.write_text(json.dumps(alerts, indent=2))


def add_alert(ticker: str, condition: str, threshold: float, email: str = None) -> dict:
    """
    Ajoute une nouvelle alerte.

    condition: 'price_above', 'price_below', 'change_above', 'change_below'
    threshold: valeur du seuil (prix en $ ou variation en %)
    """
    alerts = load_alerts()
    alert = {
        "id": len(alerts) + 1,
        "ticker": ticker.upper(),
        "condition": condition,
        "threshold": threshold,
        "email": email,
        "active": True,
        "created_at": datetime.now().isoformat(),
        "triggered": False,
        "triggered_at": None,
    }
    alerts.append(alert)
    save_alerts(alerts)
    return alert


def remove_alert(alert_id: int) -> bool:
    """Supprime une alerte par ID."""
    alerts = load_alerts()
    alerts = [a for a in alerts if a["id"] != alert_id]
    save_alerts(alerts)
    return True


def check_alerts(df: pd.DataFrame) -> list[dict]:
    """
    Verifie toutes les alertes actives contre les donnees actuelles.
    Retourne la liste des alertes declenchees.
    """
    alerts = load_alerts()
    triggered = []

    for alert in alerts:
        if not alert["active"] or alert["triggered"]:
            continue

        ticker = alert["ticker"]
        ticker_data = df[df["Ticker"] == ticker].sort_index()

        if ticker_data.empty:
            continue

        current_price = float(ticker_data["Close"].iloc[-1])
        prev_price = float(ticker_data["Close"].iloc[-2]) if len(ticker_data) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price) * 100

        condition_met = False
        message = ""

        if alert["condition"] == "price_above" and current_price > alert["threshold"]:
            condition_met = True
            message = f"{ticker} a depasse ${alert['threshold']:.2f} (prix actuel: ${current_price:.2f})"

        elif alert["condition"] == "price_below" and current_price < alert["threshold"]:
            condition_met = True
            message = f"{ticker} est passe sous ${alert['threshold']:.2f} (prix actuel: ${current_price:.2f})"

        elif alert["condition"] == "change_above" and change_pct > alert["threshold"]:
            condition_met = True
            message = f"{ticker} a monte de {change_pct:.2f}% (seuil: {alert['threshold']}%)"

        elif alert["condition"] == "change_below" and change_pct < -alert["threshold"]:
            condition_met = True
            message = f"{ticker} a baisse de {abs(change_pct):.2f}% (seuil: {alert['threshold']}%)"

        if condition_met:
            alert["triggered"] = True
            alert["triggered_at"] = datetime.now().isoformat()
            alert["message"] = message
            triggered.append(alert)

            # Envoyer email si configure
            if alert.get("email"):
                send_email_alert(alert["email"], message, ticker, current_price)

    # Sauvegarder l'etat mis a jour
    save_alerts(alerts)

    # Sauvegarder l'historique
    if triggered:
        save_alert_history(triggered)

    return triggered


def send_email_alert(to_email: str, message: str, ticker: str, price: float) -> bool:
    """Envoie un email d'alerte."""
    try:
        # Configuration SMTP (a personnaliser dans .env)
        import os
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASS", "")

        if not smtp_user or not smtp_pass:
            print(f"  [!] Email non configure (SMTP_USER/SMTP_PASS manquants)")
            return False

        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = f"[IA Finance] Alerte {ticker} - ${price:.2f}"

        body = f"""
        Alerte IA Finance
        ==================

        {message}

        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Ticker: {ticker}
        Prix actuel: ${price:.2f}

        --
        IA Finance - Systeme d'alertes automatique
        """

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())

        print(f"  [OK] Email envoye a {to_email}")
        return True

    except Exception as e:
        print(f"  [ERREUR] Email: {e}")
        return False


def save_alert_history(triggered: list[dict]) -> None:
    """Sauvegarde l'historique des alertes declenchees."""
    history = []
    if ALERTS_LOG.exists():
        history = json.loads(ALERTS_LOG.read_text())

    for alert in triggered:
        history.append({
            "id": alert["id"],
            "ticker": alert["ticker"],
            "condition": alert["condition"],
            "threshold": alert["threshold"],
            "message": alert.get("message", ""),
            "triggered_at": alert["triggered_at"],
        })

    ALERTS_LOG.write_text(json.dumps(history, indent=2))


def get_alert_history() -> list[dict]:
    """Retourne l'historique des alertes declenchees."""
    if ALERTS_LOG.exists():
        return json.loads(ALERTS_LOG.read_text())
    return []


def reset_alert(alert_id: int) -> bool:
    """Reactive une alerte declenchee."""
    alerts = load_alerts()
    for alert in alerts:
        if alert["id"] == alert_id:
            alert["triggered"] = False
            alert["triggered_at"] = None
            break
    save_alerts(alerts)
    return True

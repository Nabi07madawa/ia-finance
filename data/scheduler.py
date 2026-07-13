"""
Mise a jour automatique des donnees et verification des alertes.
Peut etre lance en tache planifiee (cron/Task Scheduler).
"""
import sys
import time
import schedule
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collect import collect_all, TICKERS
from data.clean import run_pipeline
from api.alerts import check_alerts
import pandas as pd


DATA_PATH = Path(__file__).parent / "processed" / "all_tickers_clean.csv"


def update_data():
    """Collecte les nouvelles donnees et met a jour le dataset."""
    print(f"\n{'='*50}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] MISE A JOUR")
    print(f"{'='*50}")

    print("\n[1/3] Collecte des donnees...")
    collect_all(period="5d")  # Seulement les 5 derniers jours

    print("\n[2/3] Nettoyage...")
    run_pipeline()

    print("\n[3/3] Verification des alertes...")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        triggered = check_alerts(df)
        if triggered:
            print(f"\n  [!] {len(triggered)} alerte(s) declenchee(s):")
            for a in triggered:
                print(f"      - {a.get('message', a['ticker'])}")
        else:
            print("  Aucune alerte declenchee.")
    else:
        print("  [!] Pas de donnees disponibles.")

    print(f"\n[OK] Mise a jour terminee a {datetime.now().strftime('%H:%M')}")


def run_scheduler(interval_minutes: int = 60):
    """Lance le scheduler qui met a jour les donnees periodiquement."""
    print(f"Scheduler demarre - mise a jour toutes les {interval_minutes} min")
    print(f"Prochain run: {datetime.now().strftime('%H:%M')}")
    print("Appuyez sur Ctrl+C pour arreter.\n")

    # Premier run immediat
    update_data()

    # Planifier les prochains runs
    schedule.every(interval_minutes).minutes.do(update_data)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nScheduler arrete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mise a jour automatique IA Finance")
    parser.add_argument("--once", action="store_true", help="Executer une seule fois")
    parser.add_argument("--interval", type=int, default=60, help="Intervalle en minutes (defaut: 60)")
    args = parser.parse_args()

    if args.once:
        update_data()
    else:
        run_scheduler(args.interval)

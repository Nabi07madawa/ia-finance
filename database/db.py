"""
Connexion a la base de donnees Supabase (PostgreSQL).
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Charger les variables d'environnement
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_client() -> Client:
    """Retourne un client Supabase connecte."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL et SUPABASE_KEY doivent etre definis dans .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def test_connection() -> bool:
    """Teste la connexion a Supabase."""
    try:
        client = get_client()
        # Tente une requete simple
        result = client.table("prices").select("*").limit(1).execute()
        print("[OK] Connexion Supabase reussie")
        return True
    except Exception as e:
        print(f"[!] Erreur connexion: {e}")
        return False

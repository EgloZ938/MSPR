import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    try:
        import torch
        import fastapi
        import pymongo
        import pandas
        import numpy
        import sklearn
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("💡 Installez les dépendances avec: pip install -r requirements.txt")
        return False

def check_mongodb_connection():
    """Vérifie la connexion MongoDB"""
    try:
        from pymongo import MongoClient
        
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        client = MongoClient(mongo_uri)
        client.admin.command('ping')
        client.close()
        print("✅ Connexion MongoDB OK")
        return True
    except Exception as e:
        print(f"❌ Erreur MongoDB: {e}")
        print("💡 Vérifiez que MongoDB est démarré et accessible")
        return False

def check_model_files():
    """Vérifie la présence des fichiers du modèle"""
    model_dir = Path("models")
    required_files = [
        "covid_lstm_model.pth",
        "time_scaler.pkl", 
        "demo_scaler.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Fichiers modèle manquants: {missing_files}")
        print("💡 Lancez d'abord l'entraînement avec: python train_model.py")
        return False
    else:
        print("✅ Fichiers du modèle trouvés")
        return True

def train_model():
    """Lance l'entraînement du modèle"""
    print("🏋️ Lancement de l'entraînement du modèle...")
    try:
        result = subprocess.run([sys.executable, "train_model.py"], check=True)
        print("✅ Entraînement terminé avec succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        return False

def start_api(port=8000, reload=True):
    """Démarre l'API FastAPI"""
    print(f"🚀 Démarrage de l'API IA sur le port {port}...")
    
    cmd = [
        "uvicorn", 
        "app:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 API arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du démarrage: {e}")

def main():
    parser = argparse.ArgumentParser(description="Démarrage de l'API IA COVID-19")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port pour l'API (défaut: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Désactiver le rechargement automatique")
    parser.add_argument("--train", "-t", action="store_true", help="Forcer le re-entraînement du modèle")
    parser.add_argument("--skip-checks", action="store_true", help="Ignorer les vérifications préliminaires")
    
    args = parser.parse_args()
    
    print("🤖 API IA COVID-19 - Démarrage")
    print("=" * 40)
    
    if not args.skip_checks:
        # Vérifications préliminaires
        print("🔍 Vérifications préliminaires...")
        
        if not check_requirements():
            sys.exit(1)
        
        if not check_mongodb_connection():
            print("⚠️  Continuez avec MongoDB hors ligne (certaines fonctionnalités limitées)")
        
        # Vérifier le modèle ou l'entraîner
        if args.train or not check_model_files():
            if not train_model():
                print("❌ Impossible de continuer sans modèle entraîné")
                sys.exit(1)
    
    # Démarrer l'API
    start_api(port=args.port, reload=not args.no_reload)

if __name__ == "__main__":
    main()
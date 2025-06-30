import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse
from dotenv import load_dotenv
load_dotenv()

def print_revolutionary_banner():
    """Affiche la bannière révolutionnaire multi-horizon"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║  🚀 COVID-19 IA RÉVOLUTIONNAIRE v3.0 - MULTI-HORIZON EDITION         ║
║                                                                       ║
║  🧠 Transformer Hybride + LSTM Multi-Têtes                            ║
║  💉 Logique Vaccination Progressive Intelligente                     ║
║  👥 Intégration Démographique Révolutionnaire                        ║
║  📅 Prédictions Multi-Temporelles RÉVOLUTIONNAIRES:                  ║
║     📈 Court terme: 1j, 7j, 14j, 30j (précision maximale)           ║
║     📊 Moyen terme: 90j, 180j (impact vaccination)                   ║
║     🌟 Long terme: 1an, 2ans, 5ans (projections démographiques)     ║
║  📂 100% CSV - AUCUNE DÉPENDANCE EXTERNE!                           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_python_version():
    """Vérifie la version Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis. Version actuelle:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Vérifie et installe les dépendances"""
    print("\n📦 Vérification des dépendances...")
    
    required_packages = {
        'torch': 'torch>=2.0.1',
        'fastapi': 'fastapi>=0.95.2',
        'pandas': 'pandas>=2.0.3',
        'numpy': 'numpy>=1.24.3',
        'sklearn': 'scikit-learn>=1.3.0',
        'matplotlib': 'matplotlib>=3.7.2',
        'seaborn': 'seaborn>=0.12.2',
        'uvicorn': 'uvicorn[standard]>=0.22.0'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} manquant")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n🔧 Installation de {len(missing_packages)} packages manquants...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Toutes les dépendances installées !")
        except subprocess.CalledProcessError:
            print("❌ Erreur installation. Essayez manuellement:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_csv_files():
    """Vérifie la présence des fichiers CSV multi-horizon"""
    print("\n📊 Vérification des fichiers CSV Multi-Horizon...")
    
    data_path = Path("../data/dataset_clean")
    required_files = [
        # Fichiers COVID (au moins un requis)
        ("covid_19_clean_complete_clean.csv", "OBLIGATOIRE - Données COVID principales"),
        ("full_grouped_clean.csv", "ALTERNATIF - Données COVID alternatives"),
        # Fichiers pour multi-horizon
        ("cumulative-covid-vaccinations_clean.csv", "ESSENTIEL - Données vaccination pour moyen/long terme"),
        ("consolidated_demographics_data.csv", "ESSENTIEL - Profils démographiques pour long terme")
    ]
    
    critical_missing = []
    important_missing = []
    
    # Vérifier fichiers COVID
    covid_files = [
        data_path / "covid_19_clean_complete_clean.csv",
        data_path / "full_grouped_clean.csv"
    ]
    covid_found = any(f.exists() for f in covid_files)
    
    if not covid_found:
        critical_missing.extend([
            "covid_19_clean_complete_clean.csv",
            "full_grouped_clean.csv"
        ])
        print("❌ AUCUN fichier COVID principal trouvé!")
    else:
        for f in covid_files:
            if f.exists():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"✅ {f.name} ({size_mb:.1f} MB)")
                break
    
    # Vérifier fichiers multi-horizon
    multihorizon_files = [
        ("cumulative-covid-vaccinations_clean.csv", "vaccination"),
        ("consolidated_demographics_data.csv", "démographie")
    ]
    
    for file_name, description in multihorizon_files:
        file_path = data_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✅ {file_name} ({size_mb:.1f} MB) - {description}")
        else:
            print(f"⚠️ {file_name} MANQUANT - Données {description}")
            important_missing.append(file_name)
    
    if critical_missing:
        print(f"\n❌ ERREUR CRITIQUE: Fichiers manquants obligatoires!")
        print("💡 SOLUTION: Assurez-vous d'avoir au moins un fichier COVID")
        return False
    
    if important_missing:
        print(f"\n⚠️ Fichiers multi-horizon manquants: {len(important_missing)}")
        print("   Impact: Capacités long terme limitées")
        if "cumulative-covid-vaccinations_clean.csv" in important_missing:
            print("   - Pas de logique vaccination progressive")
        if "consolidated_demographics_data.csv" in important_missing:
            print("   - Pas d'impact démographique")
        print("   Le modèle utilisera des valeurs par défaut")
    
    return True

def create_multihorizon_env_file():
    """Crée le fichier .env multi-horizon"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\n⚙️ Création du fichier .env multi-horizon...")
        
        env_content = """# Configuration COVID IA Révolutionnaire v3.0 Multi-Horizon
CSV_DATA_PATH=../data/dataset_clean
MODEL_DIR=models
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=100
API_PORT=8000
LOG_LEVEL=INFO

# Horizons Multi-Temporels
PREDICTION_HORIZONS=1,7,14,30,90,180,365,730,1825
SHORT_TERM_HORIZONS=1,7,14,30
MEDIUM_TERM_HORIZONS=90,180
LONG_TERM_HORIZONS=365,730,1825

# Features Révolutionnaires
ENABLE_VACCINATION_LOGIC=true
ENABLE_DEMOGRAPHIC_IMPACT=true
ENABLE_MULTIHORIZON_HEADS=true

# Plus besoin de MongoDB !
# MONGO_URI=mongodb://localhost:27017
"""
        
        env_file.write_text(env_content)
        print("✅ Fichier .env multi-horizon créé")
    else:
        print("✅ Fichier .env existant")

def create_directories():
    """Crée les dossiers nécessaires"""
    print("\n📁 Création des dossiers...")
    
    directories = ['models', 'outputs', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def run_multihorizon_training(quick_mode=False):
    """Lance l'entraînement multi-horizon"""
    print("\n🧠 LANCEMENT DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE MULTI-HORIZON...")
    print("=" * 70)
    
    # Arguments d'entraînement
    args = [sys.executable, "trainer.py"]
    
    if quick_mode:
        args.extend(["--quick"])
        print("⚡ Mode rapide activé (modèle réduit, 20 epochs)")
        print("   Idéal pour: Tests, validation setup, développement")
    else:
        print("🔥 Mode complet (modèle full-size, 100 epochs)")
        print("   Idéal pour: Production, performance maximale")
    
    print("📅 Horizons entraînés: 1j, 7j, 14j, 30j, 90j, 180j, 1an, 2ans, 5ans")
    print("🧠 Architecture: Transformer + LSTM + 3 têtes spécialisées")
    
    try:
        # Lancer l'entraînement
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Afficher la sortie en temps réel
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 ENTRAÎNEMENT MULTI-HORIZON TERMINÉ AVEC SUCCÈS !")
            return True
        else:
            print(f"\n❌ Erreur entraînement multi-horizon (code: {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement multi-horizon interrompu par l'utilisateur")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Erreur multi-horizon: {e}")
        return False

def launch_multihorizon_api():
    """Lance l'API multi-horizon"""
    print("\n🚀 LANCEMENT DE L'API RÉVOLUTIONNAIRE MULTI-HORIZON...")
    
    try:
        # Vérifier que le modèle existe
        model_file = Path("models/covid_revolutionary_longterm_model.pth")
        if not model_file.exists():
            print("❌ Modèle multi-horizon non trouvé. Lancez d'abord l'entraînement.")
            return False
        
        print("🌐 API Multi-Horizon démarrant sur http://localhost:8000")
        print("📖 Documentation interactive: http://localhost:8000/docs")
        print("📅 Endpoints spécialisés:")
        print("   POST /predict - Prédictions multi-horizon")
        print("   GET /horizons - Horizons supportés")
        print("   GET /model/performance - Performance par horizon")
        print("💡 Ctrl+C pour arrêter")
        
        # Lancer l'API
        subprocess.run([
            sys.executable, "covid_api.py"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ API multi-horizon arrêtée par l'utilisateur")
        return True
    except Exception as e:
        print(f"\n❌ Erreur API multi-horizon: {e}")
        return False

def test_multihorizon_api():
    """Teste l'API multi-horizon"""
    print("\n🧪 Test de l'API Multi-Horizon...")
    
    try:
        import requests
        import time
        
        # Attendre que l'API soit prête
        print("⏳ Attente du démarrage de l'API multi-horizon...")
        time.sleep(5)
        
        # Test health check
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Multi-Horizon accessible")
            print(f"   Type: {data.get('data_source', 'Unknown')}")
            print(f"   Pays disponibles: {data.get('countries_available', 0)}")
            print(f"   Horizons supportés: {data.get('supported_horizons', {})}")
        else:
            print("❌ API Multi-Horizon non accessible")
            return False
        
        # Test horizons supportés
        response = requests.get("http://localhost:8000/horizons", timeout=10)
        if response.status_code == 200:
            horizons_data = response.json()
            print("✅ Horizons multi-temporels configurés")
            for category, horizons in horizons_data['supported_horizons'].items():
                print(f"   {category}: {horizons}")
        
        # Test prédiction multi-horizon
        test_payload = {
            "country": "France",
            "prediction_horizons": [7, 90, 365],  # Court, moyen, long terme
            "include_uncertainty": True,
            "horizon_category": "auto"
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prédiction multi-horizon test réussie !")
            print(f"   Pays: {result['country']}")
            print(f"   Prédictions: {len(result['predictions'])} horizons")
            print(f"   Horizons résumé: {result['horizons_summary']}")
            print(f"   Confiance globale: {result['model_confidence']['overall_confidence']:.1%}")
            
            # Afficher échantillon prédictions
            for pred in result['predictions'][:3]:  # Afficher 3 premières
                cat = pred['horizon_category']
                horizon = pred['horizon_days']
                confirmed = pred['confirmed']
                print(f"   {horizon}j ({cat}): {confirmed:,.0f} cas confirmés")
            
            return True
        else:
            print(f"❌ Erreur prédiction multi-horizon: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test API multi-horizon: {e}")
        return False

def show_multihorizon_integration_guide():
    """Affiche le guide d'intégration multi-horizon"""
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║  🎯 INTÉGRATION MULTI-HORIZON AVEC TON DASHBOARD VUE.JS             ║
╚══════════════════════════════════════════════════════════════════════╝

1️⃣ AVANTAGES RÉVOLUTIONNAIRES:
   ✅ Prédictions court/moyen/long terme dans un seul modèle
   ✅ Logique vaccination progressive intelligente
   ✅ Impact démographique intégré pour long terme
   ✅ API 100% CSV - aucune dépendance externe

2️⃣ NOUVEAUX ENDPOINTS MULTI-HORIZON:
   POST /predict - Prédictions adaptatives selon horizon
   GET /horizons - Liste complète des horizons supportés
   GET /model/performance - Performance par catégorie d'horizon

3️⃣ UTILISATION FRONTEND:
   ```javascript
   // Prédictions court terme (précision maximale)
   const shortTerm = await axios.post('/predict', {
     country: 'France',
     prediction_horizons: [1, 7, 14, 30],
     horizon_category: 'short_term'
   });
   
   // Prédictions moyen terme (impact vaccination)
   const mediumTerm = await axios.post('/predict', {
     country: 'France', 
     prediction_horizons: [90, 180],
     horizon_category: 'medium_term'
   });
   
   // Prédictions long terme (projections démographiques)
   const longTerm = await axios.post('/predict', {
     country: 'France',
     prediction_horizons: [365, 730, 1825],
     horizon_category: 'long_term'
   });
   ```

4️⃣ LOGIQUE RÉVOLUTIONNAIRE INTÉGRÉE:
   🧠 Court terme: Précision maximale sur données récentes
   💉 Moyen terme: Impact vaccination progressive
   👥 Long terme: Influence démographique + vulnérabilité
   📊 Auto-adaptation selon horizon demandé

5️⃣ NOUVEAUX CHAMPS DE RÉPONSE:
   - horizon_category: "short_term" | "medium_term" | "long_term"
   - vaccination_impact: Impact vaccination pour horizon
   - demographic_context: Contexte démographique long terme
   - horizons_summary: Résumé par catégorie

🚀 TON DASHBOARD PEUT MAINTENANT FAIRE DES PRÉDICTIONS SUR 5 ANS !
"""
    print(guide)

def main():
    """Point d'entrée principal multi-horizon"""
    parser = argparse.ArgumentParser(description="Démarrage rapide COVID IA v3.0 Multi-Horizon")
    parser.add_argument("--quick", action="store_true", help="Mode entraînement rapide (20 epochs)")
    parser.add_argument("--skip-training", action="store_true", help="Ignorer l'entraînement")
    parser.add_argument("--api-only", action="store_true", help="Lancer seulement l'API")
    parser.add_argument("--test-only", action="store_true", help="Tests seulement")
    
    args = parser.parse_args()
    
    print_revolutionary_banner()
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    create_multihorizon_env_file()
    create_directories()
    
    if args.test_only:
        print("\n🧪 MODE TEST MULTI-HORIZON SEULEMENT")
        check_csv_files()
        return
    
    if not check_csv_files():
        sys.exit(1)
    
    # API seulement
    if args.api_only:
        print("\n🚀 MODE API MULTI-HORIZON SEULEMENT")
        if launch_multihorizon_api():
            show_multihorizon_integration_guide()
        return
    
    # Entraînement multi-horizon (si pas ignoré)
    if not args.skip_training:
        training_success = run_multihorizon_training(quick_mode=args.quick)
        if not training_success:
            print("\n❌ Échec de l'entraînement multi-horizon")
            sys.exit(1)
    
    # Proposer de lancer l'API multi-horizon
    print("\n" + "="*70)
    choice = input("🚀 Voulez-vous lancer l'API Multi-Horizon maintenant ? (y/N): ").lower()
    
    if choice in ['y', 'yes', 'oui']:
        if launch_multihorizon_api():
            show_multihorizon_integration_guide()
    else:
        print("\n💡 Pour lancer l'API Multi-Horizon plus tard:")
        print("   python3 covid_api.py")
        show_multihorizon_integration_guide()
    
    print("\n🎉 RÉVOLUTION MULTI-HORIZON TERMINÉE !")
    print("🌟 Ton IA COVID peut maintenant prédire de 1 jour à 5 ans ! 🚀")

if __name__ == "__main__":
    main()
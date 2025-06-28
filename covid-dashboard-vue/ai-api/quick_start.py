import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse
from dotenv import load_dotenv
load_dotenv()

def print_banner():
    """Affiche la bannière révolutionnaire"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║  🚀 COVID-19 IA RÉVOLUTIONNAIRE v2.0                         ║
║                                                              ║
║  🧠 Transformer Hybride + LSTM                               ║
║  💉 Intégration Vaccination + Démographie                    ║
║  🎯 Prédictions Multi-Horizons (1,7,14,30 jours)             ║
║  📊 Incertitude Quantifiée + Intervalles de Confiance        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
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
        'pymongo': 'pymongo>=4.3.3',
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

def check_data_files():
    """Vérifie la présence des fichiers de données"""
    print("\n📊 Vérification des fichiers de données...")
    
    data_path = Path("../data/dataset_clean")
    required_files = [
        "cumulative-covid-vaccinations_clean.csv",
        "consolidated_demographics_data.csv"
    ]
    
    all_present = True
    for file in required_files:
        file_path = data_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} MANQUANT !")
            print(f"   Attendu dans: {file_path}")
            all_present = False
    
    if not all_present:
        print("\n💡 SOLUTION: Assurez-vous d'avoir les fichiers CSV dans le bon dossier !")
        return False
    
    return True

def check_mongodb():
    """Vérifie la connexion MongoDB"""
    print("\n🗄️ Vérification MongoDB...")
    
    try:
        from pymongo import MongoClient
        
        mongo_uri = os.getenv('MONGO_URI')
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        
        db_name = os.getenv('DB_NAME')
        db = client[db_name]
        
        countries_count = db.countries.count_documents({})
        stats_count = db.daily_stats.count_documents({})
        
        if countries_count == 0:
            print("⚠️ Base MongoDB vide. Importez d'abord les données COVID.")
            return False
        
        print(f"✅ MongoDB connecté ({countries_count} pays, {stats_count:,} statistiques)")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB non accessible: {e}")
        print("💡 Démarrez MongoDB ou vérifiez MONGO_URI")
        return False

def create_env_file():
    """Crée le fichier .env si manquant"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\n⚙️ Création du fichier .env...")
        
        env_content = """# Configuration COVID IA Révolutionnaire v2.0
MONGO_URI=mongodb://localhost:27017
DB_NAME=MSPR
CSV_DATA_PATH=../data/dataset_clean
MODEL_DIR=models
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=100
API_PORT=8000
LOG_LEVEL=INFO
"""
        
        env_file.write_text(env_content)
        print("✅ Fichier .env créé")
    else:
        print("✅ Fichier .env existant")

def create_directories():
    """Crée les dossiers nécessaires"""
    print("\n📁 Création des dossiers...")
    
    directories = ['models', 'outputs', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def run_training(quick_mode=False):
    """Lance l'entraînement"""
    print("\n🧠 LANCEMENT DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE...")
    print("=" * 60)
    
    # Arguments d'entraînement
    args = [sys.executable, "trainer.py"]
    
    if quick_mode:
        args.extend(["--epochs", "20", "--batch-size", "16"])
        print("⚡ Mode rapide activé (20 epochs)")
    else:
        print("🔥 Mode complet (100 epochs)")
    
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
            print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
            return True
        else:
            print(f"\n❌ Erreur entraînement (code: {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False

def launch_api():
    """Lance l'API en arrière-plan"""
    print("\n🚀 LANCEMENT DE L'API RÉVOLUTIONNAIRE...")
    
    try:
        # Vérifier que le modèle existe
        model_file = Path("models/covid_revolutionary_model.pth")
        if not model_file.exists():
            print("❌ Modèle non trouvé. Lancez d'abord l'entraînement.")
            return False
        
        print("🌐 API démarrant sur http://localhost:8000")
        print("📖 Documentation: http://localhost:8000/docs")
        print("💡 Ctrl+C pour arrêter")
        
        # Lancer l'API
        subprocess.run([
            sys.executable, "covid_api.py"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ API arrêtée par l'utilisateur")
        return True
    except Exception as e:
        print(f"\n❌ Erreur API: {e}")
        return False

def test_api():
    """Teste l'API avec une prédiction échantillon"""
    print("\n🧪 Test de l'API...")
    
    try:
        import requests
        import time
        
        # Attendre que l'API soit prête
        print("⏳ Attente du démarrage de l'API...")
        time.sleep(5)
        
        # Test health check
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            print("✅ API accessible")
        else:
            print("❌ API non accessible")
            return False
        
        # Test prédiction
        test_payload = {
            "country": "France",
            "prediction_horizons": [7, 14],
            "include_uncertainty": True
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prédiction test réussie !")
            print(f"   Pays: {result['country']}")
            print(f"   Prédictions: {len(result['predictions'])} horizons")
            print(f"   Confiance: {result['model_confidence']['overall_confidence']:.1%}")
            return True
        else:
            print(f"❌ Erreur prédiction: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test API: {e}")
        return False

def show_integration_guide():
    """Affiche le guide d'intégration"""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║  🎯 INTÉGRATION AVEC TON DASHBOARD VUE.JS                   ║
╚══════════════════════════════════════════════════════════════╝

1️⃣ REMPLACER L'ANCIENNE API:
   Dans ton frontend Vue.js, remplace les appels vers l'ancienne API:
   
   ANCIEN: axios.post('/api/predict', {...})
   NOUVEAU: axios.post('http://localhost:8000/predict', {...})

2️⃣ NOUVEAU COMPOSANT:
   Copie le composant RevolutionaryPredictions.vue dans:
   frontend/src/components/AI/RevolutionaryPredictions.vue

3️⃣ MISE À JOUR ModeleView.vue:
   Remplace le contenu de components/Dashboard/Modele.vue

4️⃣ ENDPOINTS DISPONIBLES:
   • POST /predict - Prédictions révolutionnaires
   • GET /model/performance - Performance du modèle
   • GET /vaccination/{country} - Analyse vaccination
   • GET /countries - Liste des pays

5️⃣ EXEMPLE D'UTILISATION:
   ```javascript
   const response = await axios.post('http://localhost:8000/predict', {
     country: 'France',
     prediction_horizons: [1, 7, 14, 30],
     include_uncertainty: true
   });
   ```

🚀 TON DASHBOARD EST MAINTENANT RÉVOLUTIONNAIRE !
"""
    print(guide)

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Démarrage rapide COVID IA v2.0")
    parser.add_argument("--quick", action="store_true", help="Mode entraînement rapide (20 epochs)")
    parser.add_argument("--skip-training", action="store_true", help="Ignorer l'entraînement")
    parser.add_argument("--api-only", action="store_true", help="Lancer seulement l'API")
    parser.add_argument("--test-only", action="store_true", help="Tests seulement")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    create_env_file()
    create_directories()
    
    if args.test_only:
        print("\n🧪 MODE TEST SEULEMENT")
        check_data_files()
        check_mongodb()
        return
    
    if not check_data_files():
        sys.exit(1)
    
    if not check_mongodb():
        sys.exit(1)
    
    # API seulement
    if args.api_only:
        print("\n🚀 MODE API SEULEMENT")
        if launch_api():
            show_integration_guide()
        return
    
    # Entraînement (si pas ignoré)
    if not args.skip_training:
        training_success = run_training(quick_mode=args.quick)
        if not training_success:
            print("\n❌ Échec de l'entraînement")
            sys.exit(1)
    
    # Proposer de lancer l'API
    print("\n" + "="*60)
    choice = input("🚀 Voulez-vous lancer l'API maintenant ? (y/N): ").lower()
    
    if choice in ['y', 'yes', 'oui']:
        if launch_api():
            show_integration_guide()
    else:
        print("\n💡 Pour lancer l'API plus tard:")
        print("   python covid_api.py")
        show_integration_guide()
    
    print("\n🎉 RÉVOLUTION TERMINÉE ! Ton IA COVID est maintenant de niveau WORLD-CLASS ! 🌟")

if __name__ == "__main__":
    main()
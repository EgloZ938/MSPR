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
    """Affiche la bannière révolutionnaire CSV"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║  🚀 COVID-19 IA RÉVOLUTIONNAIRE v2.1 - CSV EDITION           ║
║                                                              ║
║  🧠 Transformer Hybride + LSTM                               ║
║  💉 Intégration Vaccination + Démographie                    ║
║  🎯 Prédictions Multi-Horizons (1,7,14,30 jours)             ║
║  📊 Incertitude Quantifiée + Intervalles de Confiance        ║
║  📂 100% CSV - PAS DE MONGODB REQUIS!                        ║
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
    """Vérifie la présence des fichiers CSV"""
    print("\n📊 Vérification des fichiers CSV...")
    
    data_path = Path("../data/dataset_clean")
    required_files = [
        # Fichiers COVID (au moins un requis)
        ("covid_19_clean_complete_clean.csv", "OBLIGATOIRE - Données COVID principales"),
        ("full_grouped_clean.csv", "ALTERNATIF - Données COVID alternatives"),
        # Fichiers complémentaires
        ("cumulative-covid-vaccinations_clean.csv", "RECOMMANDÉ - Données vaccination"),
        ("consolidated_demographics_data.csv", "RECOMMANDÉ - Données démographiques")
    ]
    
    critical_missing = []
    optional_missing = []
    
    # Vérifier les fichiers COVID (au moins un requis)
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
        print("   Il vous faut au moins un de ces fichiers:")
        for f in covid_files:
            print(f"   - {f}")
    else:
        for f in covid_files:
            if f.exists():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"✅ {f.name} ({size_mb:.1f} MB)")
                break
    
    # Vérifier les autres fichiers
    for file_name, description in required_files[2:]:
        file_path = data_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✅ {file_name} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️ {file_name} MANQUANT - {description}")
            optional_missing.append(file_name)
    
    if critical_missing:
        print(f"\n❌ ERREUR CRITIQUE: Fichiers manquants obligatoires!")
        print("💡 SOLUTION: Assurez-vous d'avoir au moins un fichier COVID dans:")
        print(f"   {data_path}")
        return False
    
    if optional_missing:
        print(f"\n⚠️ Fichiers optionnels manquants: {len(optional_missing)}")
        print("   Le modèle utilisera des valeurs par défaut")
    
    return True

def create_env_file():
    """Crée le fichier .env si manquant"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\n⚙️ Création du fichier .env...")
        
        env_content = """# Configuration COVID IA Révolutionnaire v2.1 CSV
CSV_DATA_PATH=../data/dataset_clean
MODEL_DIR=models
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=100
API_PORT=8000
LOG_LEVEL=INFO

# Plus besoin de MongoDB !
# MONGO_URI=mongodb://localhost:27017
# DB_NAME=MSPR
"""
        
        env_file.write_text(env_content)
        print("✅ Fichier .env créé (version CSV)")
    else:
        print("✅ Fichier .env existant")

def create_directories():
    """Crée les dossiers nécessaires"""
    print("\n📁 Création des dossiers...")
    
    directories = ['models', 'outputs', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def run_csv_training(quick_mode=False):
    """Lance l'entraînement CSV"""
    print("\n🧠 LANCEMENT DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE CSV...")
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
            print("\n🎉 ENTRAÎNEMENT CSV TERMINÉ AVEC SUCCÈS !")
            return True
        else:
            print(f"\n❌ Erreur entraînement CSV (code: {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement CSV interrompu par l'utilisateur")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Erreur CSV: {e}")
        return False

def launch_csv_api():
    """Lance l'API CSV en arrière-plan"""
    print("\n🚀 LANCEMENT DE L'API RÉVOLUTIONNAIRE CSV...")
    
    try:
        # Vérifier que le modèle existe
        model_file = Path("models/covid_revolutionary_model.pth")
        if not model_file.exists():
            print("❌ Modèle non trouvé. Lancez d'abord l'entraînement CSV.")
            return False
        
        print("🌐 API CSV démarrant sur http://localhost:8000")
        print("📖 Documentation: http://localhost:8000/docs")
        print("💡 Ctrl+C pour arrêter")
        
        # Lancer l'API CSV
        subprocess.run([
            sys.executable, "covid_api.py"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ API CSV arrêtée par l'utilisateur")
        return True
    except Exception as e:
        print(f"\n❌ Erreur API CSV: {e}")
        return False

def test_csv_api():
    """Teste l'API CSV avec une prédiction échantillon"""
    print("\n🧪 Test de l'API CSV...")
    
    try:
        import requests
        import time
        
        # Attendre que l'API soit prête
        print("⏳ Attente du démarrage de l'API CSV...")
        time.sleep(5)
        
        # Test health check
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API CSV accessible")
            print(f"   Source de données: {data.get('data_source', 'Unknown')}")
            print(f"   Pays disponibles: {data.get('countries_available', 0)}")
        else:
            print("❌ API CSV non accessible")
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
            print("✅ Prédiction CSV test réussie !")
            print(f"   Pays: {result['country']}")
            print(f"   Prédictions: {len(result['predictions'])} horizons")
            print(f"   Source: {result['model_info'].get('data_source', 'CSV')}")
            print(f"   Confiance: {result['model_confidence']['overall_confidence']:.1%}")
            return True
        else:
            print(f"❌ Erreur prédiction CSV: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test API CSV: {e}")
        return False

def show_csv_integration_guide():
    """Affiche le guide d'intégration CSV"""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║  🎯 INTÉGRATION CSV AVEC TON DASHBOARD VUE.JS               ║
╚══════════════════════════════════════════════════════════════╝

1️⃣ AVANTAGE CSV vs MongoDB:
   ✅ Plus de problèmes d'import MongoDB
   ✅ Données directement depuis tes CSV
   ✅ Plus rapide et plus fiable
   ✅ Même qualité de prédictions

2️⃣ REMPLACER L'ANCIENNE API:
   Dans ton frontend Vue.js, remplace les appels vers l'ancienne API:
   
   ANCIEN: axios.post('/api/predict', {...})
   NOUVEAU: axios.post('http://localhost:8000/predict', {...})

3️⃣ ENDPOINTS CSV DISPONIBLES:
   • POST /predict - Prédictions révolutionnaires (CSV)
   • GET /model/performance - Performance du modèle
   • GET /vaccination/{country} - Analyse vaccination (CSV)
   • GET /countries - Liste des pays (CSV)
   • GET /csv/data-info - Infos sur les données CSV

4️⃣ EXEMPLE D'UTILISATION CSV:
   ```javascript
   const response = await axios.post('http://localhost:8000/predict', {
     country: 'France',
     prediction_horizons: [1, 7, 14, 30],
     include_uncertainty: true
   });
   ```

5️⃣ AVANTAGES RÉVOLUTIONNAIRES CSV:
   🔥 Aucune dépendance MongoDB
   🔥 Données en temps réel depuis tes CSV
   🔥 Même architecture Transformer + LSTM
   🔥 Même précision de prédictions
   🔥 Plus facile à déployer

🚀 TON DASHBOARD EST MAINTENANT 100% CSV RÉVOLUTIONNAIRE !
"""
    print(guide)

def main():
    """Point d'entrée principal CSV"""
    parser = argparse.ArgumentParser(description="Démarrage rapide COVID IA v2.1 CSV")
    parser.add_argument("--quick", action="store_true", help="Mode entraînement rapide (20 epochs)")
    parser.add_argument("--skip-training", action="store_true", help="Ignorer l'entraînement")
    parser.add_argument("--api-only", action="store_true", help="Lancer seulement l'API CSV")
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
        print("\n🧪 MODE TEST CSV SEULEMENT")
        check_csv_files()
        return
    
    if not check_csv_files():
        sys.exit(1)
    
    # API seulement
    if args.api_only:
        print("\n🚀 MODE API CSV SEULEMENT")
        if launch_csv_api():
            show_csv_integration_guide()
        return
    
    # Entraînement CSV (si pas ignoré)
    if not args.skip_training:
        training_success = run_csv_training(quick_mode=args.quick)
        if not training_success:
            print("\n❌ Échec de l'entraînement CSV")
            sys.exit(1)
    
    # Proposer de lancer l'API CSV
    print("\n" + "="*60)
    choice = input("🚀 Voulez-vous lancer l'API CSV maintenant ? (y/N): ").lower()
    
    if choice in ['y', 'yes', 'oui']:
        if launch_csv_api():
            show_csv_integration_guide()
    else:
        print("\n💡 Pour lancer l'API CSV plus tard:")
        print("   python3 covid_api.py")
        show_csv_integration_guide()
    
    print("\n🎉 RÉVOLUTION CSV TERMINÉE ! Ton IA COVID est maintenant 100% CSV ! 🌟")

if __name__ == "__main__":
    main()
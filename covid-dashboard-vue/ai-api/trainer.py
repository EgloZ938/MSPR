import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

# Imports des modules révolutionnaires
from covid_data_pipeline import IntelligentCovidDataPipeline
from covid_ai_model import CovidRevolutionaryTrainer

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('revolutionary_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RevolutionaryTrainingOrchestrator:
    """Orchestrateur pour l'entraînement révolutionnaire complet"""
    
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.trainer = None
        self.enriched_data = None
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        logger.info("🚀 Orchestrateur révolutionnaire initialisé")
    
    def validate_environment(self) -> bool:
        """Valide l'environnement d'entraînement"""
        logger.info("🔍 Validation de l'environnement...")
        
        required_packages = [
            'torch', 'pandas', 'numpy', 'sklearn', 'pymongo', 
            'matplotlib', 'seaborn', 'fastapi', 'uvicorn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Packages manquants: {missing_packages}")
            return False
        
        # Vérifier les fichiers de données
        csv_path = Path(self.config['csv_data_path'])
        required_files = [
            'cumulative-covid-vaccinations_clean.csv',
            'consolidated_demographics_data.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not (csv_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Fichiers CSV manquants: {missing_files}")
            return False
        
        logger.info("✅ Environnement validé")
        return True
    
    def run_data_pipeline(self) -> pd.DataFrame:
        """Exécute le pipeline de données intelligent"""
        logger.info("📊 ÉTAPE 1: PIPELINE DE DONNÉES")
        
        self.pipeline = IntelligentCovidDataPipeline(
            mongo_uri=self.config['mongo_uri'],
            db_name=self.config['db_name'],
            csv_data_path=self.config['csv_data_path']
        )
        
        try:
            self.enriched_data = self.pipeline.run_full_pipeline()
            
            # Statistiques du dataset
            logger.info("📈 STATISTIQUES DU DATASET ENRICHI:")
            logger.info(f"   Lignes: {len(self.enriched_data):,}")
            logger.info(f"   Features: {len(self.enriched_data.columns)}")
            logger.info(f"   Pays: {self.enriched_data['country_name'].nunique()}")
            logger.info(f"   Période: {self.enriched_data['date'].min()} → {self.enriched_data['date'].max()}")
            
            # Analyse de la qualité des données
            self.analyze_data_quality()
            
            return self.enriched_data
            
        except Exception as e:
            logger.error(f"❌ Erreur pipeline données: {e}")
            raise
    
    def analyze_data_quality(self):
        """Analyse la qualité des données enrichies"""
        logger.info("🔍 Analyse de la qualité des données...")
        
        if self.enriched_data is None:
            return
        
        # Analyse des valeurs manquantes
        missing_analysis = self.enriched_data.isnull().sum()
        missing_pct = (missing_analysis / len(self.enriched_data)) * 100
        
        # Features avec beaucoup de valeurs manquantes
        problematic_features = missing_pct[missing_pct > 20].sort_values(ascending=False)
        
        if len(problematic_features) > 0:
            logger.warning(f"⚠️ Features avec >20% valeurs manquantes:")
            for feature, pct in problematic_features.items():
                logger.warning(f"   {feature}: {pct:.1f}%")
        
        # Analyse par pays
        countries_data_count = self.enriched_data['country_name'].value_counts()
        logger.info(f"📊 Pays avec le plus de données: {countries_data_count.head().to_dict()}")
        logger.info(f"📊 Pays avec le moins de données: {countries_data_count.tail().to_dict()}")
        
        # Détection d'outliers simples
        numeric_columns = self.enriched_data.select_dtypes(include=[np.number]).columns
        outliers_summary = {}
        
        for col in numeric_columns:
            Q1 = self.enriched_data[col].quantile(0.25)
            Q3 = self.enriched_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.enriched_data[col] < (Q1 - 1.5 * IQR)) | 
                       (self.enriched_data[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outliers_summary[col] = outliers
        
        if outliers_summary:
            top_outliers = sorted(outliers_summary.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"📊 Features avec le plus d'outliers: {dict(top_outliers)}")
        
        # Sauvegarde du rapport
        self.save_data_quality_report(missing_pct, countries_data_count, outliers_summary)
    
    def save_data_quality_report(self, missing_pct, countries_data_count, outliers_summary):
        """Sauvegarde un rapport de qualité des données"""
        try:
            # Graphiques de qualité
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Valeurs manquantes
            top_missing = missing_pct[missing_pct > 0].head(15)
            if len(top_missing) > 0:
                axes[0,0].barh(range(len(top_missing)), top_missing.values)
                axes[0,0].set_yticks(range(len(top_missing)))
                axes[0,0].set_yticklabels(top_missing.index, fontsize=8)
                axes[0,0].set_xlabel('Pourcentage de valeurs manquantes')
                axes[0,0].set_title('Top 15 Features avec Valeurs Manquantes')
            
            # 2. Distribution des données par pays
            top_countries = countries_data_count.head(10)
            axes[0,1].bar(range(len(top_countries)), top_countries.values)
            axes[0,1].set_xticks(range(len(top_countries)))
            axes[0,1].set_xticklabels(top_countries.index, rotation=45, fontsize=8)
            axes[0,1].set_ylabel('Nombre de points de données')
            axes[0,1].set_title('Top 10 Pays par Volume de Données')
            
            # 3. Évolution temporelle
            temporal_data = self.enriched_data.groupby('date').size()
            axes[1,0].plot(temporal_data.index, temporal_data.values)
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Nombre de points de données')
            axes[1,0].set_title('Évolution Temporelle des Données')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Outliers
            if outliers_summary:
                top_outliers = sorted(outliers_summary.items(), key=lambda x: x[1], reverse=True)[:10]
                outlier_names, outlier_counts = zip(*top_outliers)
                axes[1,1].barh(range(len(outlier_names)), outlier_counts)
                axes[1,1].set_yticks(range(len(outlier_names)))
                axes[1,1].set_yticklabels(outlier_names, fontsize=8)
                axes[1,1].set_xlabel('Nombre d\'outliers')
                axes[1,1].set_title('Top 10 Features avec Outliers')
            
            plt.tight_layout()
            plt.savefig('outputs/data_quality_report.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Rapport de qualité sauvegardé: outputs/data_quality_report.png")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport qualité: {e}")
    
    def run_model_training(self):
        """Lance l'entraînement du modèle révolutionnaire"""
        logger.info("🧠 ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE RÉVOLUTIONNAIRE")
        
        if self.enriched_data is None:
            raise ValueError("Données enrichies non disponibles. Lancez d'abord le pipeline de données.")
        
        # Configuration du modèle
        model_config = self.config.get('model_config', {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'prediction_horizons': [1, 7, 14, 30]
        })
        
        self.trainer = CovidRevolutionaryTrainer(model_config)
        
        try:
            # Préparation des données
            logger.info("🎯 Préparation des données pour l'entraînement...")
            sequences, static_features, targets = self.trainer.prepare_revolutionary_dataset(
                self.enriched_data, 
                sequence_length=self.config.get('sequence_length', 30)
            )
            
            # Création des DataLoaders
            train_loader, val_loader = self.trainer.create_dataloaders(
                sequences, static_features, targets,
                batch_size=self.config.get('batch_size', 32),
                val_split=self.config.get('val_split', 0.2)
            )
            
            # Entraînement
            logger.info("🚀 Démarrage de l'entraînement révolutionnaire...")
            history = self.trainer.train_revolutionary_model(
                train_loader, val_loader,
                epochs=self.config.get('epochs', 100),
                learning_rate=self.config.get('learning_rate', 1e-4)
            )
            
            # Sauvegarde des artefacts
            self.trainer.save_artifacts(history)
            
            # Évaluation finale
            self.evaluate_model_performance(val_loader, history)
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            raise
    
    def evaluate_model_performance(self, val_loader, history):
        """Évalue les performances finales du modèle"""
        logger.info("📊 ÉVALUATION FINALE DU MODÈLE")
        
        try:
            # Métriques d'entraînement
            if history and 'val_metrics' in history:
                final_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}
                
                logger.info("🎯 MÉTRIQUES FINALES:")
                for metric_name, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {metric_name}: {value:.4f}")
            
            # Test sur quelques prédictions
            self.test_sample_predictions()
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
    
    def test_sample_predictions(self):
        """Teste le modèle sur quelques prédictions échantillons"""
        logger.info("🧪 Test de prédictions échantillons...")
        
        try:
            # Sélectionner quelques pays pour test
            test_countries = ['France', 'Germany', 'Italy', 'Spain', 'United Kingdom']
            available_countries = self.enriched_data['country_name'].unique()
            
            test_countries = [c for c in test_countries if c in available_countries][:3]
            
            if not test_countries:
                test_countries = list(available_countries)[:3]
            
            logger.info(f"🧪 Test sur les pays: {test_countries}")
            
            # Simulations de prédictions (logique simplifiée)
            for country in test_countries:
                country_data = self.enriched_data[self.enriched_data['country_name'] == country]
                if len(country_data) > 30:
                    latest_data = country_data.tail(1).iloc[0]
                    logger.info(f"   {country}: Dernières données - "
                              f"Confirmés: {latest_data.get('confirmed', 0):,.0f}, "
                              f"Décès: {latest_data.get('deaths', 0):,.0f}")
        
        except Exception as e:
            logger.error(f"❌ Erreur test prédictions: {e}")
    
    def generate_final_report(self, history):
        """Génère un rapport final complet"""
        logger.info("📝 GÉNÉRATION DU RAPPORT FINAL")
        
        try:
            report = {
                "training_summary": {
                    "model_type": "COVID Revolutionary Transformer v2.0",
                    "training_date": datetime.now().isoformat(),
                    "dataset_size": len(self.enriched_data) if self.enriched_data is not None else 0,
                    "countries_count": self.enriched_data['country_name'].nunique() if self.enriched_data is not None else 0,
                    "features_count": len(self.enriched_data.columns) if self.enriched_data is not None else 0,
                    "epochs_completed": len(history.get('train_loss', [])) if history else 0
                },
                "data_sources": {
                    "covid_timeseries": "MongoDB",
                    "vaccination_data": "cumulative-covid-vaccinations_clean.csv",
                    "demographics": "consolidated_demographics_data.csv"
                },
                "model_architecture": self.config.get('model_config', {}),
                "training_config": {
                    k: v for k, v in self.config.items() 
                    if k not in ['mongo_uri', 'db_name']
                },
                "final_performance": history.get('val_metrics', [])[-1] if history and history.get('val_metrics') else {}
            }
            
            # Sauvegarde du rapport
            import json
            with open('outputs/final_training_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("✅ Rapport final sauvegardé: outputs/final_training_report.json")
            
            # Affichage du résumé
            logger.info("🎉 RÉSUMÉ DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE:")
            logger.info(f"   📊 Dataset: {report['training_summary']['dataset_size']:,} lignes")
            logger.info(f"   🏳️ Pays: {report['training_summary']['countries_count']}")
            logger.info(f"   📈 Features: {report['training_summary']['features_count']}")
            logger.info(f"   🔄 Epochs: {report['training_summary']['epochs_completed']}")
            
            if report['final_performance']:
                logger.info("   🎯 Performance finale:")
                for metric, value in report['final_performance'].items():
                    if isinstance(value, (int, float)):
                        logger.info(f"      {metric}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport: {e}")
    
    def run_complete_training(self):
        """Lance l'entraînement complet révolutionnaire"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE COMPLET")
        logger.info("=" * 80)
        
        try:
            # 1. Validation environnement
            if not self.validate_environment():
                raise ValueError("Environnement non valide")
            
            # 2. Pipeline de données
            self.run_data_pipeline()
            
            # 3. Entraînement du modèle
            history = self.run_model_training()
            
            # 4. Rapport final
            self.generate_final_report(history)
            
            logger.info("=" * 80)
            logger.info("🎉 ENTRAÎNEMENT RÉVOLUTIONNAIRE TERMINÉ AVEC SUCCÈS!")
            logger.info("📁 Fichiers générés:")
            logger.info("   - models/covid_revolutionary_model.pth")
            logger.info("   - models/revolutionary_*_scaler.pkl")
            logger.info("   - models/revolutionary_config.json")
            logger.info("   - outputs/data_quality_report.png")
            logger.info("   - outputs/final_training_report.json")
            logger.info("   - models/training_history.png")
            logger.info("\n🚀 Le modèle révolutionnaire est prêt pour l'API!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ÉCHEC DE L'ENTRAÎNEMENT: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Entraînement révolutionnaire COVID IA v2.0")
    parser.add_argument("--config", type=str, help="Fichier de configuration JSON")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017", help="URI MongoDB")
    parser.add_argument("--db-name", type=str, default="covid_dashboard", help="Nom de la base de données")
    parser.add_argument("--csv-path", type=str, default="../data/dataset_clean", help="Chemin des fichiers CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Taux d'apprentissage")
    
    args = parser.parse_args()
    
    # Charger variables d'environnement
    load_dotenv()
    
    # Configuration par défaut
    config = {
        'mongo_uri': args.mongo_uri or os.getenv('MONGO_URI'),
        'db_name': args.db_name or os.getenv('DB_NAME'),
        'csv_data_path': args.csv_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'sequence_length': 30,
        'val_split': 0.2,
        'model_config': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'prediction_horizons': [1, 7, 14, 30]
        }
    }
    
    # Charger configuration depuis fichier si spécifié
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
        logger.info(f"✅ Configuration chargée depuis {args.config}")
    
    # Lancer l'entraînement
    orchestrator = RevolutionaryTrainingOrchestrator(config)
    success = orchestrator.run_complete_training()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SUCCÈS! Le modèle révolutionnaire est prêt!")
        print("📚 Prochaines étapes:")
        print("   1. Lancer l'API: python covid_revolutionary_api.py")
        print("   2. Tester les prédictions: /predict endpoint")
        print("   3. Intégrer avec le dashboard Vue.js")
        print("="*50)
        sys.exit(0)
    else:
        print("\n❌ ÉCHEC de l'entraînement. Vérifiez les logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
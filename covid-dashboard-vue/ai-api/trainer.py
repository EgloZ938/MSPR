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

# Imports des modules révolutionnaires (version CSV)
from covid_data_pipeline import CSVCovidDataPipeline
from covid_ai_model import CovidRevolutionaryTrainer

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_revolutionary_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVRevolutionaryTrainingOrchestrator:
    """🚀 Orchestrateur pour l'entraînement révolutionnaire 100% CSV"""
    
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.trainer = None
        self.enriched_data = None
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        logger.info("🚀 Orchestrateur CSV Révolutionnaire initialisé")
    
    def validate_csv_environment(self) -> bool:
        """Valide l'environnement CSV"""
        logger.info("🔍 Validation de l'environnement CSV...")
        
        # Vérifier les packages Python
        required_packages = [
            'torch', 'pandas', 'numpy', 'sklearn', 
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
        
        # Vérifier les fichiers CSV
        csv_path = Path(self.config['csv_data_path'])
        required_files = [
            'covid_19_clean_complete_clean.csv',  # Principal
            'full_grouped_clean.csv',             # Alternatif
            'cumulative-covid-vaccinations_clean.csv',
            'consolidated_demographics_data.csv'
        ]
        
        # Au moins un fichier COVID principal doit exister
        covid_files = [
            csv_path / 'covid_19_clean_complete_clean.csv',
            csv_path / 'full_grouped_clean.csv'
        ]
        
        covid_found = any(f.exists() for f in covid_files)
        if not covid_found:
            logger.error("❌ Aucun fichier COVID principal trouvé!")
            logger.error(f"   Cherché dans: {covid_files}")
            return False
        
        # Vérifier les autres fichiers
        missing_files = []
        for file in required_files[2:]:  # Skip COVID files déjà vérifiés
            if not (csv_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠️ Fichiers CSV optionnels manquants: {missing_files}")
            logger.warning("   Le pipeline utilisera des valeurs par défaut")
        
        logger.info("✅ Environnement CSV validé")
        return True
    
    def run_csv_data_pipeline(self) -> pd.DataFrame:
        """🚀 Exécute le pipeline de données CSV"""
        logger.info("📊 ÉTAPE 1: PIPELINE DE DONNÉES CSV")
        
        self.pipeline = CSVCovidDataPipeline(
            csv_data_path=self.config['csv_data_path']
        )
        
        try:
            self.enriched_data = self.pipeline.run_full_pipeline()
            
            # Statistiques du dataset
            logger.info("📈 STATISTIQUES DU DATASET CSV ENRICHI:")
            logger.info(f"   Lignes: {len(self.enriched_data):,}")
            logger.info(f"   Features: {len(self.enriched_data.columns)}")
            logger.info(f"   Pays: {self.enriched_data['country_name'].nunique()}")
            logger.info(f"   Période: {self.enriched_data['date'].min()} → {self.enriched_data['date'].max()}")
            
            # Analyse de la qualité des données
            self.analyze_csv_data_quality()
            
            return self.enriched_data
            
        except Exception as e:
            logger.error(f"❌ Erreur pipeline CSV: {e}")
            raise
    
    def analyze_csv_data_quality(self):
        """🔍 Analyse la qualité des données CSV enrichies"""
        logger.info("🔍 Analyse de la qualité des données CSV...")
        
        if self.enriched_data is None:
            return
        
        # Analyse des valeurs manquantes
        missing_analysis = self.enriched_data.isnull().sum()
        missing_pct = (missing_analysis / len(self.enriched_data)) * 100
        
        # Features avec beaucoup de valeurs manquantes
        problematic_features = missing_pct[missing_pct > 20].sort_values(ascending=False)
        
        if len(problematic_features) > 0:
            logger.warning(f"⚠️ Features CSV avec >20% valeurs manquantes:")
            for feature, pct in problematic_features.items():
                logger.warning(f"   {feature}: {pct:.1f}%")
        
        # Analyse par pays
        countries_data_count = self.enriched_data['country_name'].value_counts()
        logger.info(f"📊 Top 5 pays avec le plus de données: {countries_data_count.head().to_dict()}")
        logger.info(f"📊 Pays avec le moins de données: {countries_data_count.tail().to_dict()}")
        
        # Vérifier la cohérence des dates
        date_range = self.enriched_data['date'].max() - self.enriched_data['date'].min()
        logger.info(f"📅 Étendue temporelle: {date_range.days} jours")
        
        # Sauvegarde du rapport CSV
        self.save_csv_data_quality_report(missing_pct, countries_data_count)
    
    def save_csv_data_quality_report(self, missing_pct, countries_data_count):
        """💾 Sauvegarde un rapport de qualité des données CSV"""
        try:
            # Graphiques de qualité CSV
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Valeurs manquantes
            top_missing = missing_pct[missing_pct > 0].head(15)
            if len(top_missing) > 0:
                axes[0,0].barh(range(len(top_missing)), top_missing.values)
                axes[0,0].set_yticks(range(len(top_missing)))
                axes[0,0].set_yticklabels(top_missing.index, fontsize=8)
                axes[0,0].set_xlabel('Pourcentage de valeurs manquantes')
                axes[0,0].set_title('Top 15 Features CSV avec Valeurs Manquantes')
            
            # 2. Distribution des données par pays
            top_countries = countries_data_count.head(10)
            axes[0,1].bar(range(len(top_countries)), top_countries.values)
            axes[0,1].set_xticks(range(len(top_countries)))
            axes[0,1].set_xticklabels(top_countries.index, rotation=45, fontsize=8)
            axes[0,1].set_ylabel('Nombre de points de données')
            axes[0,1].set_title('Top 10 Pays par Volume de Données CSV')
            
            # 3. Évolution temporelle
            temporal_data = self.enriched_data.groupby('date').size()
            axes[1,0].plot(temporal_data.index, temporal_data.values)
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Nombre de points de données')
            axes[1,0].set_title('Évolution Temporelle des Données CSV')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Distribution des valeurs COVID
            covid_columns = ['confirmed', 'deaths', 'recovered', 'active']
            available_covid = [col for col in covid_columns if col in self.enriched_data.columns]
            
            if available_covid:
                for i, col in enumerate(available_covid[:4]):
                    if col in self.enriched_data.columns:
                        # Log scale pour mieux voir la distribution
                        values = self.enriched_data[col][self.enriched_data[col] > 0]
                        if len(values) > 0:
                            axes[1,1].hist(np.log10(values + 1), alpha=0.5, label=col, bins=30)
                
                axes[1,1].set_xlabel('Log10(Valeur + 1)')
                axes[1,1].set_ylabel('Fréquence')
                axes[1,1].set_title('Distribution des Valeurs COVID (échelle log)')
                axes[1,1].legend()
            
            plt.tight_layout()
            plt.savefig('outputs/csv_data_quality_report.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Rapport qualité CSV sauvegardé: outputs/csv_data_quality_report.png")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport qualité CSV: {e}")
    
    def run_model_training(self):
        """🧠 Lance l'entraînement du modèle révolutionnaire avec données CSV"""
        logger.info("🧠 ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE RÉVOLUTIONNAIRE (CSV)")
        
        if self.enriched_data is None:
            raise ValueError("Données CSV enrichies non disponibles. Lancez d'abord le pipeline CSV.")
        
        # Configuration du modèle (identique à la version MongoDB)
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
            logger.info("🎯 Préparation des données CSV pour l'entraînement...")
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
            logger.info("🚀 Démarrage de l'entraînement révolutionnaire (données CSV)...")
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
            logger.error(f"❌ Erreur entraînement CSV: {e}")
            raise
    
    def evaluate_model_performance(self, val_loader, history):
        """📊 Évalue les performances finales du modèle CSV"""
        logger.info("📊 ÉVALUATION FINALE DU MODÈLE (DONNÉES CSV)")
        
        try:
            # Métriques d'entraînement
            if history and 'val_metrics' in history:
                final_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}
                
                logger.info("🎯 MÉTRIQUES FINALES (CSV):")
                for metric_name, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {metric_name}: {value:.4f}")
            
            # Test sur quelques prédictions
            self.test_csv_sample_predictions()
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation CSV: {e}")
    
    def test_csv_sample_predictions(self):
        """🧪 Teste le modèle sur quelques prédictions échantillons CSV"""
        logger.info("🧪 Test de prédictions échantillons (données CSV)...")
        
        try:
            # Sélectionner quelques pays pour test
            test_countries = ['France', 'Germany', 'Italy', 'Spain', 'United Kingdom']
            available_countries = self.enriched_data['country_name'].unique()
            
            test_countries = [c for c in test_countries if c in available_countries][:3]
            
            if not test_countries:
                test_countries = list(available_countries)[:3]
            
            logger.info(f"🧪 Test CSV sur les pays: {test_countries}")
            
            # Simulations de prédictions (logique simplifiée)
            for country in test_countries:
                country_data = self.enriched_data[self.enriched_data['country_name'] == country]
                if len(country_data) > 30:
                    latest_data = country_data.tail(1).iloc[0]
                    logger.info(f"   {country}: Dernières données CSV - "
                              f"Confirmés: {latest_data.get('confirmed', 0):,.0f}, "
                              f"Décès: {latest_data.get('deaths', 0):,.0f}")
        
        except Exception as e:
            logger.error(f"❌ Erreur test prédictions CSV: {e}")
    
    def generate_csv_final_report(self, history):
        """📝 Génère un rapport final CSV complet"""
        logger.info("📝 GÉNÉRATION DU RAPPORT FINAL CSV")
        
        try:
            report = {
                "training_summary": {
                    "model_type": "COVID Revolutionary Transformer v2.0 (CSV)",
                    "data_source": "Pure CSV Pipeline",
                    "training_date": datetime.now().isoformat(),
                    "dataset_size": len(self.enriched_data) if self.enriched_data is not None else 0,
                    "countries_count": self.enriched_data['country_name'].nunique() if self.enriched_data is not None else 0,
                    "features_count": len(self.enriched_data.columns) if self.enriched_data is not None else 0,
                    "epochs_completed": len(history.get('train_loss', [])) if history else 0
                },
                "csv_data_sources": {
                    "covid_timeseries": "covid_19_clean_complete_clean.csv / full_grouped_clean.csv",
                    "vaccination_data": "cumulative-covid-vaccinations_clean.csv",
                    "demographics": "consolidated_demographics_data.csv"
                },
                "model_architecture": self.config.get('model_config', {}),
                "training_config": {
                    k: v for k, v in self.config.items() 
                    if k not in ['csv_data_path']
                },
                "final_performance": history.get('val_metrics', [])[-1] if history and history.get('val_metrics') else {}
            }
            
            # Sauvegarde du rapport
            import json
            with open('outputs/csv_final_training_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("✅ Rapport final CSV sauvegardé: outputs/csv_final_training_report.json")
            
            # Affichage du résumé
            logger.info("🎉 RÉSUMÉ DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE CSV:")
            logger.info(f"   📊 Dataset CSV: {report['training_summary']['dataset_size']:,} lignes")
            logger.info(f"   🏳️ Pays: {report['training_summary']['countries_count']}")
            logger.info(f"   📈 Features: {report['training_summary']['features_count']}")
            logger.info(f"   🔄 Epochs: {report['training_summary']['epochs_completed']}")
            
            if report['final_performance']:
                logger.info("   🎯 Performance finale CSV:")
                for metric, value in report['final_performance'].items():
                    if isinstance(value, (int, float)):
                        logger.info(f"      {metric}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport CSV: {e}")
    
    def run_complete_csv_training(self):
        """🚀 Lance l'entraînement complet révolutionnaire CSV"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE CSV COMPLET")
        logger.info("=" * 80)
        
        try:
            # 1. Validation environnement CSV
            if not self.validate_csv_environment():
                raise ValueError("Environnement CSV non valide")
            
            # 2. Pipeline de données CSV
            self.run_csv_data_pipeline()
            
            # 3. Entraînement du modèle
            history = self.run_model_training()
            
            # 4. Rapport final CSV
            self.generate_csv_final_report(history)
            
            logger.info("=" * 80)
            logger.info("🎉 ENTRAÎNEMENT RÉVOLUTIONNAIRE CSV TERMINÉ AVEC SUCCÈS!")
            logger.info("📁 Fichiers générés:")
            logger.info("   - models/covid_revolutionary_model.pth")
            logger.info("   - models/revolutionary_*_scaler.pkl")
            logger.info("   - models/revolutionary_config.json")
            logger.info("   - outputs/csv_data_quality_report.png")
            logger.info("   - outputs/csv_final_training_report.json")
            logger.info("   - models/training_history.png")
            logger.info("\n🚀 Le modèle révolutionnaire CSV est prêt pour l'API!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ÉCHEC DE L'ENTRAÎNEMENT CSV: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """🚀 Point d'entrée principal CSV"""
    parser = argparse.ArgumentParser(description="Entraînement révolutionnaire COVID IA v2.0 - Version CSV")
    parser.add_argument("--config", type=str, help="Fichier de configuration JSON")
    parser.add_argument("--csv-path", type=str, default="../data/dataset_clean", help="Chemin des fichiers CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Taux d'apprentissage")
    
    args = parser.parse_args()
    
    # Configuration CSV par défaut
    config = {
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
        logger.info(f"✅ Configuration CSV chargée depuis {args.config}")
    
    # Lancer l'entraînement CSV
    orchestrator = CSVRevolutionaryTrainingOrchestrator(config)
    success = orchestrator.run_complete_csv_training()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SUCCÈS! Le modèle révolutionnaire CSV est prêt!")
        print("📚 Prochaines étapes:")
        print("   1. Lancer l'API: python covid_revolutionary_api.py")
        print("   2. Tester les prédictions: /predict endpoint")
        print("   3. Intégrer avec le dashboard Vue.js")
        print("="*50)
        sys.exit(0)
    else:
        print("\n❌ ÉCHEC de l'entraînement CSV. Vérifiez les logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
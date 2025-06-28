import pandas as pd
import numpy as np
from pymongo import MongoClient
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentCovidDataPipeline:
    """
    Pipeline révolutionnaire qui fusionne intelligemment :
    - Données temporelles COVID (MongoDB)
    - Données vaccination (CSV)
    - Données démographiques (CSV)
    """
    
    def __init__(self, mongo_uri: str, db_name: str, csv_data_path: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.csv_data_path = csv_data_path
        self.client = None
        self.db = None
        
        # Caches pour optimiser les performances
        self.vaccination_cache = {}
        self.demographics_cache = {}
        self.country_mapping = {}
        
        logger.info("🚀 Initialisation du Pipeline Intelligent COVID IA v2.0")
    
    def connect_mongodb(self) -> bool:
        """Connexion optimisée à MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.db.command('ping')
            logger.info("✅ MongoDB connecté avec succès")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur MongoDB: {e}")
            return False
    
    def load_vaccination_data(self) -> pd.DataFrame:
        """Charge et préprocess les données de vaccination"""
        logger.info("💉 Chargement des données de vaccination...")
        
        vax_file = os.path.join(self.csv_data_path, 'cumulative-covid-vaccinations_clean.csv')
        
        if not os.path.exists(vax_file):
            logger.error(f"❌ Fichier vaccination introuvable: {vax_file}")
            return pd.DataFrame()
        
        # Chargement et nettoyage
        vax_df = pd.read_csv(vax_file)
        vax_df.columns = vax_df.columns.str.strip()
        
        # Conversion et nettoyage des types
        vax_df['date'] = pd.to_datetime(vax_df['date'], errors='coerce')
        vax_df['cumulative_vaccinations'] = pd.to_numeric(vax_df['cumulative_vaccinations'], errors='coerce').fillna(0)
        vax_df['daily_vaccinations'] = pd.to_numeric(vax_df['daily_vaccinations'], errors='coerce').fillna(0)
        
        # Suppression des valeurs aberrantes
        vax_df = vax_df.dropna(subset=['date'])
        vax_df['cumulative_vaccinations'] = vax_df['cumulative_vaccinations'].clip(lower=0)
        vax_df['daily_vaccinations'] = vax_df['daily_vaccinations'].clip(lower=0)
        
        # Calcul de features avancées de vaccination
        vax_df = vax_df.sort_values(['country', 'date'])
        
        # Taux de vaccination (moyenne mobile 7 jours)
        vax_df['vaccination_rate_7d'] = vax_df.groupby('country')['daily_vaccinations'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Accélération de la vaccination
        vax_df['vaccination_acceleration'] = vax_df.groupby('country')['daily_vaccinations'].transform(
            lambda x: x.diff()
        )
        
        # Pourcentage de population vaccinée (estimé)
        vax_df['vaccination_coverage_est'] = vax_df['cumulative_vaccinations'] / 100000  # À ajuster avec démographie
        
        logger.info(f"💉 {len(vax_df)} enregistrements vaccination traités")
        logger.info(f"🏳️ {vax_df['country'].nunique()} pays avec données vaccination")
        logger.info(f"📅 Période: {vax_df['date'].min()} → {vax_df['date'].max()}")
        
        # Cache par pays pour optimisation
        for country in vax_df['country'].unique():
            if pd.notna(country):
                country_data = vax_df[vax_df['country'] == country].sort_values('date')
                self.vaccination_cache[country.strip().lower()] = country_data
        
        return vax_df
    
    def load_demographics_data(self) -> pd.DataFrame:
        """Charge et préprocess les données démographiques"""
        logger.info("👥 Chargement des données démographiques...")
        
        demo_file = os.path.join(self.csv_data_path, 'consolidated_demographics_data.csv')
        
        if not os.path.exists(demo_file):
            logger.error(f"❌ Fichier démographique introuvable: {demo_file}")
            return pd.DataFrame()
        
        # Chargement des données
        demo_df = pd.read_csv(demo_file)
        demo_df.columns = demo_df.columns.str.strip()
        
        # Nettoyage et conversion des types
        numeric_columns = [
            'Total population (thousands)', 'Birth rate', 'Mortality rate', 
            'Life expectancy', 'Infant mortality rate', 'Number of children per woman',
            'Growth rate', 'Share of people aged 65 and over (%)'
        ]
        
        for col in numeric_columns:
            if col in demo_df.columns:
                demo_df[col] = pd.to_numeric(demo_df[col], errors='coerce')
        
        # Calcul de features démographiques avancées
        demo_df['population_millions'] = demo_df['Total population (thousands)'] / 1000
        demo_df['elderly_ratio'] = demo_df['Share of people aged 65 and over (%)'] / 100
        demo_df['demographic_vulnerability'] = (
            demo_df['elderly_ratio'] * demo_df['Mortality rate'] / 100
        )
        demo_df['population_density_category'] = pd.cut(
            demo_df['population_millions'], 
            bins=[0, 1, 10, 50, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        
        # Extraction des noms de pays depuis les régions quand possible
        # (certaines lignes contiennent des pays individuels)
        demo_df['country_extracted'] = demo_df['Countries'].str.strip()
        
        logger.info(f"👥 {len(demo_df)} enregistrements démographiques traités")
        logger.info(f"🏳️ {demo_df['Countries'].nunique()} entités démographiques")
        
        # Cache démographique
        for idx, row in demo_df.iterrows():
            country_key = row['Countries'].strip().lower() if pd.notna(row['Countries']) else None
            if country_key:
                self.demographics_cache[country_key] = row.to_dict()
        
        return demo_df
    
    def load_covid_timeseries(self) -> pd.DataFrame:
        """Charge les données COVID depuis MongoDB - VERSION BASÉE SUR server_mongo.js"""
        logger.info("🦠 Chargement des séries temporelles COVID...")
        
        # D'abord, récupérer la liste des pays (comme dans server_mongo.js)
        countries_cursor = self.db.countries.find({}, {"country_name": 1, "_id": 1})
        countries_list = list(countries_cursor)
        
        logger.info(f"🏳️ {len(countries_list)} pays trouvés")
        
        all_covid_data = []
        
        # Pour chaque pays, utiliser la même logique que server_mongo.js
        for country_doc in countries_list:
            country_name = country_doc['country_name']
            
            # Pipeline IDENTIQUE à server_mongo.js
            pipeline = [
                {
                    "$lookup": {
                        "from": "countries",
                        "localField": "country_id",
                        "foreignField": "_id",
                        "as": "country"
                    }
                },
                {"$unwind": "$country"},
                {"$match": {"country.country_name": country_name}},  # ⭐ Comme dans ton JS !
                {"$sort": {"date": 1}},
                {
                    "$project": {
                        "country_name": "$country.country_name",
                        "date": 1,
                        "confirmed": 1,
                        "deaths": 1,
                        "recovered": 1,
                        "active": 1
                    }
                }
            ]
            
            country_data = list(self.db.daily_stats.aggregate(pipeline))
            
            if country_data:
                all_covid_data.extend(country_data)
                if len(all_covid_data) % 1000 == 0:  # Log de progression
                    logger.info(f"   📈 {len(all_covid_data)} points chargés...")
        
        if len(all_covid_data) == 0:
            raise ValueError("❌ Aucune donnée COVID trouvée dans MongoDB!")
        
        covid_df = pd.DataFrame(all_covid_data)
        covid_df['date'] = pd.to_datetime(covid_df['date'])
        covid_df = covid_df.sort_values(['country_name', 'date'])
        
        # Calcul des features COVID avancées (reste pareil...)
        covid_df['new_cases'] = covid_df.groupby('country_name')['confirmed'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        # ... etc (le reste du code reste identique)
        
        logger.info(f"🦠 {len(covid_df)} points temporels COVID chargés")
        logger.info(f"🏳️ {covid_df['country_name'].nunique()} pays")
        logger.info(f"📅 Période: {covid_df['date'].min()} → {covid_df['date'].max()}")
        
        return covid_df
    
    def smart_country_matching(self, country_name: str) -> str:
        """Matching intelligent des noms de pays entre sources"""
        # Normalisation
        normalized = country_name.strip().lower()
        
        # Mapping manuel pour les cas courants
        mapping = {
            'united states': 'us',
            'usa': 'us',
            'united kingdom': 'uk',
            'south korea': 'korea, south',
            'north korea': 'korea, north',
            'czech republic': 'czechia',
            'russia': 'russian federation',
            'iran': 'iran, islamic republic of',
            'venezuela': 'venezuela, bolivarian republic of',
            'syria': 'syrian arab republic',
            'tanzania': 'tanzania, united republic of',
            'bolivia': 'bolivia, plurinational state of',
            'vietnam': 'viet nam',
        }
        
        return mapping.get(normalized, normalized)
    
    def get_vaccination_features(self, country: str, target_date: datetime) -> Dict:
        """Récupère les features de vaccination avancées pour un pays/date"""
        country_key = self.smart_country_matching(country)
        
        # Valeurs par défaut
        features = {
            'cumulative_vaccinations': 0,
            'daily_vaccinations': 0,
            'vaccination_rate_7d': 0,
            'vaccination_acceleration': 0,
            'vaccination_coverage_est': 0,
            'vaccination_momentum': 0,  # Nouvelle feature
            'days_since_vax_start': 0,  # Nouvelle feature
        }
        
        # Recherche dans le cache
        vaccination_data = None
        for cached_country, data in self.vaccination_cache.items():
            if country_key in cached_country or cached_country in country_key:
                vaccination_data = data
                break
        
        if vaccination_data is not None and len(vaccination_data) > 0:
            # Trouver la date la plus proche
            vaccination_data = vaccination_data.copy()
            vaccination_data['date_diff'] = abs((vaccination_data['date'] - target_date).dt.days)
            
            # Prendre la date la plus proche dans une fenêtre de 30 jours
            valid_dates = vaccination_data[vaccination_data['date_diff'] <= 30]
            
            if len(valid_dates) > 0:
                closest_idx = valid_dates['date_diff'].idxmin()
                closest_vax = valid_dates.loc[closest_idx]
                
                features['cumulative_vaccinations'] = float(closest_vax['cumulative_vaccinations'])
                features['daily_vaccinations'] = float(closest_vax['daily_vaccinations'])
                features['vaccination_rate_7d'] = float(closest_vax.get('vaccination_rate_7d', 0))
                features['vaccination_acceleration'] = float(closest_vax.get('vaccination_acceleration', 0))
                features['vaccination_coverage_est'] = float(closest_vax.get('vaccination_coverage_est', 0))
                
                # Nouvelles features calculées
                # Momentum = tendance récente de vaccination
                recent_data = vaccination_data[vaccination_data['date'] <= target_date].tail(14)
                if len(recent_data) > 7:
                    recent_avg = recent_data['daily_vaccinations'].mean()
                    older_avg = recent_data['daily_vaccinations'].head(7).mean()
                    features['vaccination_momentum'] = (recent_avg - older_avg) / (older_avg + 1)
                
                # Jours depuis le début de la vaccination
                first_vax_date = vaccination_data[vaccination_data['cumulative_vaccinations'] > 0]['date'].min()
                if pd.notna(first_vax_date):
                    features['days_since_vax_start'] = (target_date - first_vax_date).days
        
        return features
    
    def get_demographic_features(self, country: str) -> Dict:
        """Récupère les features démographiques pour un pays"""
        country_key = self.smart_country_matching(country)
        
        # Valeurs par défaut
        features = {
            'population_millions': 50,  # Valeur médiane mondiale
            'birth_rate': 15,
            'mortality_rate': 8,
            'life_expectancy': 70,
            'infant_mortality_rate': 25,
            'fertility_rate': 2.5,
            'growth_rate': 1.0,
            'elderly_ratio': 0.08,
            'demographic_vulnerability': 0.5,
            'population_density_cat': 'Medium'
        }
        
        # Recherche dans le cache démographique
        demo_data = None
        for cached_country, data in self.demographics_cache.items():
            if country_key in cached_country or cached_country in country_key:
                demo_data = data
                break
        
        if demo_data:
            features.update({
                'population_millions': demo_data.get('population_millions', features['population_millions']),
                'birth_rate': demo_data.get('Birth rate', features['birth_rate']),
                'mortality_rate': demo_data.get('Mortality rate', features['mortality_rate']),
                'life_expectancy': demo_data.get('Life expectancy', features['life_expectancy']),
                'infant_mortality_rate': demo_data.get('Infant mortality rate', features['infant_mortality_rate']),
                'fertility_rate': demo_data.get('Number of children per woman', features['fertility_rate']),
                'growth_rate': demo_data.get('Growth rate', features['growth_rate']),
                'elderly_ratio': demo_data.get('elderly_ratio', features['elderly_ratio']),
                'demographic_vulnerability': demo_data.get('demographic_vulnerability', features['demographic_vulnerability']),
            })
        
        return features
    
    def create_advanced_features(self, covid_df: pd.DataFrame) -> pd.DataFrame:
        """Créé un dataset unifié avec features avancées"""
        logger.info("🧠 Création des features avancées...")
        
        enriched_rows = []
        
        for idx, row in covid_df.iterrows():
            country = row['country_name']
            date = row['date']
            
            # Features COVID de base
            base_features = row.to_dict()
            
            # Features temporelles avancées
            temporal_features = {
                'day_of_year': date.timetuple().tm_yday,
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'week_of_year': date.isocalendar().week,
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'weekday': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
            }
            
            # Features vaccination
            vaccination_features = self.get_vaccination_features(country, date)
            
            # Features démographiques
            demographic_features = self.get_demographic_features(country)
            
            # Features d'interaction (RÉVOLUTIONNAIRE)
            interaction_features = {
                'vax_per_capita': vaccination_features['cumulative_vaccinations'] / 
                                (demographic_features['population_millions'] * 1000 + 1),
                'elderly_vax_urgency': demographic_features['elderly_ratio'] * 
                                     (1 - vaccination_features['vaccination_coverage_est']),
                'demographic_covid_risk': demographic_features['demographic_vulnerability'] * 
                                        base_features['mortality_rate'],
                'vax_effectiveness_lag': vaccination_features['cumulative_vaccinations'] * 
                                       np.exp(-vaccination_features['days_since_vax_start'] / 21),  # Effet avec délai
            }
            
            # Fusion de toutes les features
            enriched_row = {
                **base_features,
                **temporal_features,
                **vaccination_features,
                **demographic_features,
                **interaction_features
            }
            
            enriched_rows.append(enriched_row)
        
        enriched_df = pd.DataFrame(enriched_rows)
        
        logger.info(f"🧠 Dataset enrichi créé: {len(enriched_df)} lignes × {len(enriched_df.columns)} features")
        logger.info(f"📊 Nouvelles features: {len(enriched_df.columns) - len(covid_df.columns)}")
        
        return enriched_df
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Execute le pipeline complet"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
        
        # 1. Connexion MongoDB
        if not self.connect_mongodb():
            raise Exception("❌ Impossible de se connecter à MongoDB")
        
        # 2. Chargement des données
        vaccination_df = self.load_vaccination_data()
        demographics_df = self.load_demographics_data()
        covid_df = self.load_covid_timeseries()
        
        # 3. Création du dataset unifié enrichi
        enriched_dataset = self.create_advanced_features(covid_df)
        
        # 4. Sauvegarde
        output_file = os.path.join(self.csv_data_path, 'enriched_covid_dataset.csv')
        enriched_dataset.to_csv(output_file, index=False)
        logger.info(f"💾 Dataset enrichi sauvegardé: {output_file}")
        
        logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        return enriched_dataset

if __name__ == "__main__":
    # Configuration
    MONGO_URI = os.getenv('MONGO_URI')
    DB_NAME = os.getenv('DB_NAME')
    CSV_DATA_PATH = '../data/dataset_clean'
    
    # Exécution du pipeline
    pipeline = IntelligentCovidDataPipeline(MONGO_URI, DB_NAME, CSV_DATA_PATH)
    enriched_data = pipeline.run_full_pipeline()
    
    print("\n🎯 STATISTIQUES FINALES:")
    print(f"📊 Lignes: {len(enriched_data):,}")
    print(f"📈 Features: {len(enriched_data.columns)}")
    print(f"🏳️ Pays: {enriched_data['country_name'].nunique()}")
    print(f"📅 Période: {enriched_data['date'].min()} → {enriched_data['date'].max()}")
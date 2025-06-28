import pandas as pd
import numpy as np
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

class CSVCovidDataPipeline:
    """
    🚀 PIPELINE RÉVOLUTIONNAIRE 100% CSV
    
    Fusionne intelligemment :
    - Données temporelles COVID (CSV)
    - Données vaccination (CSV) 
    - Données démographiques (CSV)
    - Features temporelles avancées
    """
    
    def __init__(self, csv_data_path: str):
        self.csv_data_path = csv_data_path
        
        # Caches pour optimiser les performances
        self.vaccination_cache = {}
        self.demographics_cache = {}
        self.who_regions_cache = {}
        
        # Mapping des fichiers CSV
        self.csv_files = {
            'covid_clean_complete': 'covid_19_clean_complete_clean.csv',
            'country_wise_latest': 'country_wise_latest_clean.csv', 
            'day_wise': 'day_wise_clean.csv',
            'full_grouped': 'full_grouped_clean.csv',
            'usa_county': 'usa_county_wise_clean.csv',
            'worldometer': 'worldometer_data_clean.csv',
            'vaccination': 'cumulative-covid-vaccinations_clean.csv',
            'demographics': 'consolidated_demographics_data.csv'
        }
        
        logger.info("🚀 Pipeline CSV Révolutionnaire initialisé")
    
    def load_vaccination_data(self) -> pd.DataFrame:
        """Charge et préprocess les données de vaccination"""
        logger.info("💉 Chargement des données de vaccination...")
        
        vax_file = os.path.join(self.csv_data_path, self.csv_files['vaccination'])
        
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
        vax_df['vaccination_coverage_est'] = vax_df['cumulative_vaccinations'] / 100000
        
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
        
        demo_file = os.path.join(self.csv_data_path, self.csv_files['demographics'])
        
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
        
        logger.info(f"👥 {len(demo_df)} enregistrements démographiques traités")
        
        # Cache démographique
        for idx, row in demo_df.iterrows():
            country_key = row['Countries'].strip().lower() if pd.notna(row['Countries']) else None
            if country_key:
                self.demographics_cache[country_key] = row.to_dict()
        
        return demo_df
    
    def load_covid_timeseries_from_csv(self) -> pd.DataFrame:
        """🦠 Charge les séries temporelles COVID depuis les CSV"""
        logger.info("🦠 Chargement des séries temporelles COVID depuis CSV...")
        
        # Option 1: Utiliser covid_19_clean_complete_clean.csv (le plus complet)
        covid_file = os.path.join(self.csv_data_path, self.csv_files['covid_clean_complete'])
        
        if not os.path.exists(covid_file):
            logger.error(f"❌ Fichier COVID principal introuvable: {covid_file}")
            # Essayer le fichier alternatif
            covid_file = os.path.join(self.csv_data_path, self.csv_files['full_grouped'])
            
            if not os.path.exists(covid_file):
                raise ValueError("❌ Aucun fichier COVID trouvé!")
        
        logger.info(f"📄 Lecture du fichier: {covid_file}")
        
        # Chargement du CSV principal
        covid_df = pd.read_csv(covid_file)
        covid_df.columns = covid_df.columns.str.strip()
        
        # Normalisation des noms de colonnes (adaptation selon le format)
        column_mapping = {
            'Country/Region': 'country_name',
            'Country_Region': 'country_name',
            'Country': 'country_name',
            'Date': 'date',
            'Confirmed': 'confirmed',
            'Deaths': 'deaths', 
            'Recovered': 'recovered',
            'Active': 'active'
        }
        
        # Appliquer le mapping
        for old_col, new_col in column_mapping.items():
            if old_col in covid_df.columns:
                covid_df = covid_df.rename(columns={old_col: new_col})
        
        # Vérifier les colonnes essentielles
        required_cols = ['country_name', 'date', 'confirmed', 'deaths', 'recovered']
        missing_cols = [col for col in required_cols if col not in covid_df.columns]
        
        if missing_cols:
            logger.error(f"❌ Colonnes manquantes: {missing_cols}")
            logger.info(f"📊 Colonnes disponibles: {list(covid_df.columns)}")
            raise ValueError(f"Colonnes essentielles manquantes: {missing_cols}")
        
        # Nettoyage et conversion des types
        covid_df['date'] = pd.to_datetime(covid_df['date'], errors='coerce')
        covid_df['confirmed'] = pd.to_numeric(covid_df['confirmed'], errors='coerce').fillna(0)
        covid_df['deaths'] = pd.to_numeric(covid_df['deaths'], errors='coerce').fillna(0)
        covid_df['recovered'] = pd.to_numeric(covid_df['recovered'], errors='coerce').fillna(0)
        
        # Calculer 'active' si pas présent
        if 'active' not in covid_df.columns:
            covid_df['active'] = covid_df['confirmed'] - covid_df['deaths'] - covid_df['recovered']
        else:
            covid_df['active'] = pd.to_numeric(covid_df['active'], errors='coerce').fillna(0)
        
        # Supprimer les lignes avec dates invalides
        covid_df = covid_df.dropna(subset=['date'])
        
        # Filtrer les pays valides
        covid_df = covid_df[covid_df['country_name'].notna()]
        covid_df = covid_df[covid_df['country_name'] != '']
        
        # Trier par pays et date
        covid_df = covid_df.sort_values(['country_name', 'date'])
        
        # Calcul des features COVID avancées
        logger.info("🧠 Calcul des features COVID avancées...")
        
        # Features par pays
        covid_df['new_cases'] = covid_df.groupby('country_name')['confirmed'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        covid_df['new_deaths'] = covid_df.groupby('country_name')['deaths'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        covid_df['new_recovered'] = covid_df.groupby('country_name')['recovered'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        
        # Moyennes mobiles (7 jours)
        covid_df['new_cases_ma7'] = covid_df.groupby('country_name')['new_cases'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        covid_df['new_deaths_ma7'] = covid_df.groupby('country_name')['new_deaths'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Taux calculés
        covid_df['growth_rate'] = covid_df.groupby('country_name')['confirmed'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        covid_df['mortality_rate'] = (covid_df['deaths'] / covid_df['confirmed'].clip(lower=1) * 100).fillna(0)
        covid_df['recovery_rate'] = (covid_df['recovered'] / covid_df['confirmed'].clip(lower=1) * 100).fillna(0)
        
        # Tendance (7 jours)
        covid_df['trend_7d'] = covid_df.groupby('country_name')['new_cases_ma7'].transform(
            lambda x: x.diff().apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
        )
        
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
            'vaccination_momentum': 0,
            'days_since_vax_start': 0,
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
            'population_millions': 50,
            'birth_rate': 15,
            'mortality_rate': 8,
            'life_expectancy': 70,
            'infant_mortality_rate': 25,
            'fertility_rate': 2.5,
            'growth_rate': 1.0,
            'elderly_ratio': 0.08,
            'demographic_vulnerability': 0.5,
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
                'week_of_year': date.isocalendar()[1],  # Fix: [1] pour récupérer la semaine
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
                                       np.exp(-vaccination_features['days_since_vax_start'] / 21),
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
        """Execute le pipeline complet 100% CSV"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE CSV COMPLET")
        
        # 1. Chargement des données auxiliaires
        vaccination_df = self.load_vaccination_data()
        demographics_df = self.load_demographics_data()
        
        # 2. Chargement des séries temporelles COVID depuis CSV
        covid_df = self.load_covid_timeseries_from_csv()
        
        # 3. Création du dataset unifié enrichi
        enriched_dataset = self.create_advanced_features(covid_df)
        
        # 4. Sauvegarde
        output_file = os.path.join(self.csv_data_path, 'enriched_covid_dataset.csv')
        enriched_dataset.to_csv(output_file, index=False)
        logger.info(f"💾 Dataset enrichi sauvegardé: {output_file}")
        
        logger.info("🎉 PIPELINE CSV TERMINÉ AVEC SUCCÈS!")
        return enriched_dataset

if __name__ == "__main__":
    # Configuration
    CSV_DATA_PATH = '../data/dataset_clean'
    
    # Exécution du pipeline
    pipeline = CSVCovidDataPipeline(CSV_DATA_PATH)
    enriched_data = pipeline.run_full_pipeline()
    
    print("\n🎯 STATISTIQUES FINALES:")
    print(f"📊 Lignes: {len(enriched_data):,}")
    print(f"📈 Features: {len(enriched_data.columns)}")
    print(f"🏳️ Pays: {enriched_data['country_name'].nunique()}")
    print(f"📅 Période: {enriched_data['date'].min()} → {enriched_data['date'].max()}")
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
    🚀 PIPELINE LONG TERME - VERSION 2.1
    
    NOUVEAUTÉS :
    - Prédictions multi-temporelles : 6 mois, 1 an, 2 ans, 5 ans
    - Démographie = profil fixe exploité intelligemment
    - Vaccination = trigger progressif selon couverture réelle
    - Logique conditionnelle avant/après vaccination
    """
    
    def __init__(self, csv_data_path: str):
        self.csv_data_path = csv_data_path
        
        # Caches optimisés
        self.vaccination_cache = {}
        self.demographics_cache = {}
        self.country_population_cache = {}
        
        # 🚀 NOUVEAUX HORIZONS RÉVOLUTIONNAIRES
        self.prediction_horizons = [
            # Court terme (compatibilité)
            1, 7, 14, 30,
            # Moyen terme 
            90, 180,  # 3 mois, 6 mois
            # Long terme RÉVOLUTIONNAIRE
            365, 730, 1825  # 1 an, 2 ans, 5 ans
        ]
        
        # Mapping des fichiers CSV
        self.csv_files = {
            'covid_clean_complete': 'covid_19_clean_complete_clean.csv',
            'vaccination': 'cumulative-covid-vaccinations_clean.csv',
            'demographics': 'consolidated_demographics_data.csv'
        }
        
        logger.info("🚀 Pipeline CSV RÉVOLUTIONNAIRE LONG TERME initialisé")
        logger.info(f"📅 Horizons de prédiction : {self.prediction_horizons}")
    
    def load_vaccination_data(self) -> pd.DataFrame:
        """💉 Charge données vaccination avec logique progressive"""
        logger.info("💉 Chargement vaccination - Mode Progressif...")
        
        vax_file = os.path.join(self.csv_data_path, self.csv_files['vaccination'])
        
        if not os.path.exists(vax_file):
            logger.error(f"❌ Fichier vaccination introuvable: {vax_file}")
            return pd.DataFrame()
        
        vax_df = pd.read_csv(vax_file)
        vax_df.columns = vax_df.columns.str.strip()
        
        # Nettoyage
        vax_df['date'] = pd.to_datetime(vax_df['date'], errors='coerce')
        vax_df['cumulative_vaccinations'] = pd.to_numeric(vax_df['cumulative_vaccinations'], errors='coerce').fillna(0)
        vax_df['daily_vaccinations'] = pd.to_numeric(vax_df['daily_vaccinations'], errors='coerce').fillna(0)
        
        vax_df = vax_df.dropna(subset=['date'])
        vax_df = vax_df.sort_values(['country', 'date'])
        
        # 🧠 CALCULS INTELLIGENTS par pays
        for country in vax_df['country'].unique():
            if pd.notna(country):
                country_data = vax_df[vax_df['country'] == country].copy()
                
                # Population du pays (pour calculer la couverture réelle)
                population = self.get_country_population(country)
                
                # 📊 Features vaccination INTELLIGENTES
                country_data['vaccination_coverage_real'] = (
                    country_data['cumulative_vaccinations'] / population * 100
                ).clip(0, 100)  # Max 100%
                
                # Impact progressif (pas tout d'un coup!)
                country_data['protection_factor'] = np.tanh(
                    country_data['vaccination_coverage_real'] / 30  # Saturation à 30% de couverture
                )
                
                # Efficacité temporelle (baisse avec le temps)
                country_data['days_since_start'] = (
                    country_data['date'] - country_data['date'].min()
                ).dt.days
                
                country_data['vaccine_effectiveness'] = (
                    country_data['protection_factor'] * 
                    np.exp(-country_data['days_since_start'] / 365)  # Baisse chaque année
                )
                
                # Cache optimisé
                self.vaccination_cache[country.strip().lower()] = country_data
        
        logger.info(f"💉 {len(vax_df)} vaccinations traitées avec logique progressive")
        return vax_df
    
    def get_country_population(self, country: str) -> float:
        """👥 Récupère population pays depuis démographie"""
        country_key = self.smart_country_matching(country)
        
        # Cache pour éviter les recalculs
        if country_key in self.country_population_cache:
            return self.country_population_cache[country_key]
        
        # Valeur par défaut
        default_population = 50_000_000  # 50M par défaut
        
        # Chercher dans les données démographiques
        demo_file = os.path.join(self.csv_data_path, self.csv_files['demographics'])
        if os.path.exists(demo_file):
            try:
                demo_df = pd.read_csv(demo_file)
                
                # Chercher le pays (plusieurs variantes possibles)
                country_matches = demo_df[
                    demo_df['Countries'].str.lower().str.contains(country_key, na=False)
                ]
                
                if len(country_matches) > 0:
                    # Prendre la donnée la plus récente
                    latest = country_matches.sort_values('Year', ascending=False).iloc[0]
                    population = latest.get('Total population (thousands)', default_population / 1000) * 1000
                    self.country_population_cache[country_key] = population
                    return population
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture population {country}: {e}")
        
        self.country_population_cache[country_key] = default_population
        return default_population
    
    def load_demographics_data(self) -> pd.DataFrame:
        """👥 Charge démographie comme PROFIL FIXE"""
        logger.info("👥 Chargement démographie - Mode Profil Fixe...")
        
        demo_file = os.path.join(self.csv_data_path, self.csv_files['demographics'])
        
        if not os.path.exists(demo_file):
            logger.error(f"❌ Fichier démographique introuvable")
            return pd.DataFrame()
        
        demo_df = pd.read_csv(demo_file)
        demo_df.columns = demo_df.columns.str.strip()
        
        # 🧠 EXPLOITATION INTELLIGENTE (pas de corrélation temporelle)
        for idx, row in demo_df.iterrows():
            country_name = row.get('Countries', '')
            if pd.notna(country_name) and country_name.strip():
                country_key = country_name.strip().lower()
                
                # 📊 PROFIL DÉMOGRAPHIQUE FIXE
                profile = {
                    'population_millions': row.get('Total population (thousands)', 50000) / 1000,
                    'birth_rate': row.get('Birth rate', 15),
                    'mortality_rate': row.get('Mortality rate', 8),
                    'life_expectancy': row.get('Life expectancy', 70),
                    'infant_mortality_rate': row.get('Infant mortality rate', 25),
                    'fertility_rate': row.get('Number of children per woman', 2.5),
                    'growth_rate': row.get('Growth rate', 1.0),
                    'elderly_ratio': row.get('Share of people aged 65 and over (%)', 8) / 100,
                    
                    # 🧠 FACTEURS CALCULÉS INTELLIGENTS
                    'covid_vulnerability': self.calculate_covid_vulnerability(row),
                    'demographic_resilience': self.calculate_demographic_resilience(row),
                    'age_mortality_factor': self.calculate_age_mortality_factor(row),
                }
                
                self.demographics_cache[country_key] = profile
        
        logger.info(f"👥 {len(self.demographics_cache)} profils démographiques créés")
        return demo_df
    
    def calculate_covid_vulnerability(self, demo_row) -> float:
        """🧠 Calcule vulnérabilité COVID selon démographie"""
        elderly_pct = demo_row.get('Share of people aged 65 and over (%)', 8)
        life_expectancy = demo_row.get('Life expectancy', 70)
        mortality_rate = demo_row.get('Mortality rate', 8)
        
        # Plus de personnes âgées = plus vulnérable
        # Moins d'espérance de vie = plus vulnérable
        vulnerability = (
            (elderly_pct / 20) * 0.4 +  # 20% elderly = max factor
            (1 - life_expectancy / 85) * 0.3 +  # 85 ans = life expectancy max
            (mortality_rate / 15) * 0.3  # 15‰ = mortality rate max
        )
        
        return np.clip(vulnerability, 0.1, 1.0)
    
    def calculate_demographic_resilience(self, demo_row) -> float:
        """💪 Calcule résilience démographique"""
        birth_rate = demo_row.get('Birth rate', 15)
        growth_rate = demo_row.get('Growth rate', 1.0)
        
        # Plus de naissances et croissance = plus résilient
        resilience = (
            (birth_rate / 40) * 0.6 +  # 40‰ = birth rate max
            (growth_rate / 3) * 0.4  # 3% = growth rate max
        )
        
        return np.clip(resilience, 0.1, 1.0)
    
    def calculate_age_mortality_factor(self, demo_row) -> float:
        """⚰️ Facteur mortalité selon âge population"""
        elderly_pct = demo_row.get('Share of people aged 65 and over (%)', 8)
        infant_mortality = demo_row.get('Infant mortality rate', 25)
        
        # Plus de personnes âgées = mortalité COVID plus élevée
        # Mortalité infantile élevée = système santé faible
        age_factor = (
            (elderly_pct / 25) * 0.7 +  # 25% elderly = max
            (infant_mortality / 100) * 0.3  # 100‰ = infant mortality max
        )
        
        return np.clip(age_factor, 0.1, 2.0)
    
    def get_vaccination_impact(self, country: str, target_date: datetime) -> Dict:
        """💉 Impact vaccination INTELLIGENT selon date"""
        country_key = self.smart_country_matching(country)
        
        # Valeurs par défaut (pas de vaccination)
        impact = {
            'has_vaccination': False,
            'coverage_percent': 0.0,
            'protection_factor': 0.0,
            'case_reduction_factor': 1.0,  # 1.0 = pas de réduction
            'mortality_reduction_factor': 1.0,
            'days_since_vax_start': 0,
            'vaccination_momentum': 0.0
        }
        
        # Chercher données vaccination pour ce pays
        vaccination_data = None
        for cached_country, data in self.vaccination_cache.items():
            if country_key in cached_country or cached_country in country_key:
                vaccination_data = data
                break
        
        if vaccination_data is not None and len(vaccination_data) > 0:
            # Trouver la date la plus proche <= target_date
            valid_dates = vaccination_data[vaccination_data['date'] <= target_date]
            
            if len(valid_dates) > 0:
                latest_vax = valid_dates.sort_values('date').iloc[-1]
                
                coverage = latest_vax.get('vaccination_coverage_real', 0.0)
                protection = latest_vax.get('protection_factor', 0.0)
                effectiveness = latest_vax.get('vaccine_effectiveness', 0.0)
                
                if coverage > 0.1:  # Au moins 0.1% de couverture
                    impact.update({
                        'has_vaccination': True,
                        'coverage_percent': coverage,
                        'protection_factor': protection,
                        # 🧠 RÉDUCTIONS INTELLIGENTES
                        'case_reduction_factor': 1.0 - (effectiveness * 0.8),  # Max 80% réduction cas
                        'mortality_reduction_factor': 1.0 - (effectiveness * 0.95),  # Max 95% réduction décès
                        'days_since_vax_start': latest_vax.get('days_since_start', 0),
                        'vaccination_momentum': latest_vax.get('daily_vaccinations', 0)
                    })
        
        return impact
    
    def smart_country_matching(self, country_name: str) -> str:
        """🎯 Matching intelligent noms pays"""
        normalized = country_name.strip().lower()
        
        # Mapping manuel étendu
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
            'france': 'france',
            'germany': 'germany',
            'italy': 'italy',
            'spain': 'spain'
        }
        
        return mapping.get(normalized, normalized)
    
    def create_revolutionary_features(self, covid_df: pd.DataFrame) -> pd.DataFrame:
        """🧠 Features révolutionnaires pour prédictions long terme"""
        logger.info("🧠 Création features révolutionnaires LONG TERME...")
        
        enriched_rows = []
        
        for idx, row in covid_df.iterrows():
            country = row['country_name']
            date = row['date']
            
            # Features COVID de base
            base_features = row.to_dict()
            
            # 📅 Features temporelles étendues
            temporal_features = {
                'day_of_year': date.timetuple().tm_yday,
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'week_of_year': date.isocalendar()[1],
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'weekday': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                
                # 🚀 FEATURES LONG TERME
                'pandemic_year': (date - pd.to_datetime('2020-01-01')).days / 365.25,
                'pandemic_phase': min((date - pd.to_datetime('2020-01-01')).days / 730, 1.0),  # 2 ans max
                'seasonal_factor': np.cos(2 * np.pi * date.timetuple().tm_yday / 365.25),
            }
            
            # 👥 PROFIL DÉMOGRAPHIQUE FIXE
            demographic_profile = self.get_demographic_profile(country)
            
            # 💉 IMPACT VACCINATION (selon la date!)
            vaccination_impact = self.get_vaccination_impact(country, date)
            
            # 🧠 FEATURES D'INTERACTION RÉVOLUTIONNAIRES
            interaction_features = {
                # Impact démographique sur COVID
                'demographic_covid_severity': (
                    demographic_profile['covid_vulnerability'] * 
                    base_features['mortality_rate']
                ),
                
                # Capacité de résilience
                'country_resilience_score': (
                    demographic_profile['demographic_resilience'] * 
                    (1 - demographic_profile['covid_vulnerability'])
                ),
                
                # Impact vaccination modulé par démographie
                'vaccination_effectiveness_adjusted': (
                    vaccination_impact['protection_factor'] *
                    (2 - demographic_profile['covid_vulnerability'])  # Plus efficace si pop moins vulnérable
                ),
                
                # Mortalité prédite selon âge + vaccination
                'predicted_mortality_factor': (
                    demographic_profile['age_mortality_factor'] *
                    vaccination_impact['mortality_reduction_factor']
                ),
                
                # Phase épidémique
                'epidemic_phase': self.calculate_epidemic_phase(base_features, vaccination_impact),
            }
            
            # 🚀 FUSION TOTALE
            enriched_row = {
                **base_features,
                **temporal_features,
                **demographic_profile,
                **vaccination_impact,
                **interaction_features
            }
            
            enriched_rows.append(enriched_row)
        
        enriched_df = pd.DataFrame(enriched_rows)
        
        logger.info(f"🧠 Dataset révolutionnaire créé: {len(enriched_df)} lignes × {len(enriched_df.columns)} features")
        return enriched_df
    
    def get_demographic_profile(self, country: str) -> Dict:
        """👥 Récupère profil démographique du cache"""
        country_key = self.smart_country_matching(country)
        
        # Chercher dans le cache
        for cached_country, profile in self.demographics_cache.items():
            if country_key in cached_country or cached_country in country_key:
                return profile
        
        # Profil par défaut si pas trouvé
        return {
            'population_millions': 50,
            'birth_rate': 15,
            'mortality_rate': 8,
            'life_expectancy': 70,
            'infant_mortality_rate': 25,
            'fertility_rate': 2.5,
            'growth_rate': 1.0,
            'elderly_ratio': 0.08,
            'covid_vulnerability': 0.5,
            'demographic_resilience': 0.5,
            'age_mortality_factor': 1.0,
        }
    
    def calculate_epidemic_phase(self, covid_data: Dict, vaccination_impact: Dict) -> float:
        """📊 Calcule phase épidémique (croissance/décroissance/plateau)"""
        growth_rate = covid_data.get('growth_rate', 0)
        has_vaccination = vaccination_impact.get('has_vaccination', False)
        protection = vaccination_impact.get('protection_factor', 0)
        
        if not has_vaccination:
            # Phase pré-vaccination : croissance naturelle
            if growth_rate > 0.05:  # Croissance forte
                return 1.0  # Phase ascendante
            elif growth_rate > -0.05:  # Plateau
                return 0.5  # Phase plateau
            else:
                return 0.0  # Phase descendante naturelle
        else:
            # Phase vaccination : décroissance espérée
            expected_decline = -protection * 0.1  # Plus de protection = plus de déclin
            if growth_rate > expected_decline + 0.05:
                return 0.8  # Résistance à la vaccination
            elif growth_rate > expected_decline - 0.05:
                return 0.3  # Déclin selon attentes
            else:
                return 0.1  # Déclin rapide
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """🚀 Pipeline complet RÉVOLUTIONNAIRE LONG TERME"""
        logger.info("🚀 DÉMARRAGE PIPELINE RÉVOLUTIONNAIRE LONG TERME")
        
        # 1. Charger données démographiques (profils fixes)
        demographics_df = self.load_demographics_data()
        
        # 2. Charger données vaccination (impact progressif)
        vaccination_df = self.load_vaccination_data()
        
        # 3. Charger séries temporelles COVID
        covid_df = self.load_covid_timeseries_from_csv()
        
        # 4. Créer dataset révolutionnaire
        enriched_dataset = self.create_revolutionary_features(covid_df)
        
        # 5. Sauvegarde
        output_file = os.path.join(self.csv_data_path, 'enriched_covid_dataset_longterm.csv')
        enriched_dataset.to_csv(output_file, index=False)
        logger.info(f"💾 Dataset révolutionnaire sauvegardé: {output_file}")
        
        logger.info("🎉 PIPELINE RÉVOLUTIONNAIRE LONG TERME TERMINÉ!")
        return enriched_dataset
    
    def load_covid_timeseries_from_csv(self) -> pd.DataFrame:
        """🦠 Charge séries temporelles COVID (méthode existante)"""
        logger.info("🦠 Chargement séries temporelles COVID...")
        
        covid_file = os.path.join(self.csv_data_path, self.csv_files['covid_clean_complete'])
        
        if not os.path.exists(covid_file):
            covid_file = os.path.join(self.csv_data_path, 'full_grouped_clean.csv')
            if not os.path.exists(covid_file):
                raise ValueError("❌ Aucun fichier COVID trouvé!")
        
        covid_df = pd.read_csv(covid_file)
        covid_df.columns = covid_df.columns.str.strip()
        
        # Normalisation colonnes
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
        
        for old_col, new_col in column_mapping.items():
            if old_col in covid_df.columns:
                covid_df = covid_df.rename(columns={old_col: new_col})
        
        # Nettoyage
        covid_df['date'] = pd.to_datetime(covid_df['date'], errors='coerce')
        covid_df['confirmed'] = pd.to_numeric(covid_df['confirmed'], errors='coerce').fillna(0)
        covid_df['deaths'] = pd.to_numeric(covid_df['deaths'], errors='coerce').fillna(0)
        covid_df['recovered'] = pd.to_numeric(covid_df['recovered'], errors='coerce').fillna(0)
        
        if 'active' not in covid_df.columns:
            covid_df['active'] = covid_df['confirmed'] - covid_df['deaths'] - covid_df['recovered']
        else:
            covid_df['active'] = pd.to_numeric(covid_df['active'], errors='coerce').fillna(0)
        
        covid_df = covid_df.dropna(subset=['date'])
        covid_df = covid_df[covid_df['country_name'].notna()]
        covid_df = covid_df.sort_values(['country_name', 'date'])
        
        # Features COVID étendues
        covid_df['new_cases'] = covid_df.groupby('country_name')['confirmed'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        covid_df['new_deaths'] = covid_df.groupby('country_name')['deaths'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        covid_df['new_recovered'] = covid_df.groupby('country_name')['recovered'].transform(
            lambda x: x.diff().fillna(0).clip(lower=0)
        )
        
        covid_df['new_cases_ma7'] = covid_df.groupby('country_name')['new_cases'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        covid_df['new_deaths_ma7'] = covid_df.groupby('country_name')['new_deaths'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        covid_df['growth_rate'] = covid_df.groupby('country_name')['confirmed'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        covid_df['mortality_rate'] = (covid_df['deaths'] / covid_df['confirmed'].clip(lower=1) * 100).fillna(0)
        covid_df['recovery_rate'] = (covid_df['recovered'] / covid_df['confirmed'].clip(lower=1) * 100).fillna(0)
        
        covid_df['trend_7d'] = covid_df.groupby('country_name')['new_cases_ma7'].transform(
            lambda x: x.diff().apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
        )
        
        logger.info(f"🦠 {len(covid_df)} points COVID chargés")
        return covid_df

if __name__ == "__main__":
    CSV_DATA_PATH = '../data/dataset_clean'
    
    pipeline = CSVCovidDataPipeline(CSV_DATA_PATH)
    enriched_data = pipeline.run_full_pipeline()
    
    print("\n🎯 STATISTIQUES RÉVOLUTIONNAIRES LONG TERME:")
    print(f"📊 Lignes: {len(enriched_data):,}")
    print(f"📈 Features: {len(enriched_data.columns)}")
    print(f"🏳️ Pays: {enriched_data['country_name'].nunique()}")
    print(f"📅 Période: {enriched_data['date'].min()} → {enriched_data['date'].max()}")
    print(f"🚀 Prêt pour prédictions LONG TERME!")
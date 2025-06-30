from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union, Tuple
import joblib
import os
import json
import logging
from pathlib import Path

# Import du modèle révolutionnaire multi-horizon
import sys
sys.path.append('.')
from covid_ai_model import CovidRevolutionaryLongTermTransformer
from covid_data_pipeline import CSVCovidDataPipeline
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="COVID-19 Revolutionary AI API - Multi-Horizon Edition", 
    version="3.0.0",
    description="API révolutionnaire avec prédictions court/moyen/long terme (1j → 5 ans)"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CSV_DATA_PATH = os.getenv('CSV_DATA_PATH', '../data/dataset_clean')
MODEL_DIR = 'models'

# 🚀 HORIZONS RÉVOLUTIONNAIRES MULTI-TEMPORELS
PREDICTION_HORIZONS = {
    'short_term': [1, 7, 14, 30],
    'medium_term': [90, 180],  # 3 mois, 6 mois
    'long_term': [365, 730, 1825]  # 1 an, 2 ans, 5 ans
}

ALL_HORIZONS = [h for category in PREDICTION_HORIZONS.values() for h in category]

# Modèles Pydantic ÉTENDUS
class MultiHorizonPredictionRequest(BaseModel):
    country: str = Field(..., description="Nom du pays")
    region: Optional[str] = Field(None, description="Région (optionnel)")
    prediction_horizons: List[int] = Field([1, 7, 14, 30, 365], description="Horizons de prédiction en jours")
    start_date: Optional[str] = Field(None, description="Date de début des prédictions (YYYY-MM-DD)")
    include_uncertainty: bool = Field(True, description="Inclure les intervalles de confiance")
    include_attention: bool = Field(False, description="Inclure les poids d'attention")
    horizon_category: Optional[str] = Field("auto", description="auto, short_term, medium_term, long_term")

class PredictionResult(BaseModel):
    date: str
    horizon_days: int
    horizon_category: str  # "short_term", "medium_term", "long_term"
    confirmed: float
    deaths: float
    recovered: float
    active: float
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    attention_score: Optional[float] = None
    vaccination_impact: Optional[Dict[str, float]] = None
    demographic_context: Optional[Dict[str, float]] = None

class MultiHorizonPredictionResponse(BaseModel):
    country: str
    region: Optional[str]
    prediction_start_date: str
    model_info: Dict[str, Union[str, int, float]]
    predictions: List[PredictionResult]
    horizons_summary: Dict[str, List[int]]
    vaccination_timeline: Dict[str, Union[str, float]]
    demographic_profile: Dict[str, Union[str, float]]
    model_confidence: Dict[str, float]
    prediction_timestamp: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    csv_data_available: bool
    supported_horizons: Dict[str, List[int]]
    countries_available: int
    model_performance: Optional[Dict[str, float]] = None
    data_source: str = "CSV Files Multi-Horizon"

# Variables globales
model = None
sequence_scaler = None
static_scaler = None
target_scaler = None
model_config = None
csv_pipeline = None
enriched_data = None

# Features synchronisées avec l'entraînement
TEMPORAL_FEATURES = [
    'confirmed', 'deaths', 'recovered', 'active',
    'new_cases', 'new_deaths', 'new_recovered',
    'new_cases_ma7', 'new_deaths_ma7',
    'growth_rate', 'mortality_rate', 'recovery_rate', 'trend_7d',
    'month_sin', 'month_cos'
]

STATIC_FEATURES = [
    # Démographiques
    'population_millions', 'birth_rate', 'mortality_rate', 'life_expectancy',
    'infant_mortality_rate', 'fertility_rate', 'growth_rate', 'elderly_ratio',
    'covid_vulnerability', 'demographic_resilience', 'age_mortality_factor',
    
    # Vaccination
    'has_vaccination', 'coverage_percent', 'protection_factor',
    'case_reduction_factor', 'mortality_reduction_factor', 'vaccination_momentum',
    
    # Temporelles étendues
    'pandemic_year', 'pandemic_phase', 'seasonal_factor',
    'day_of_year', 'quarter', 'week_of_year', 'weekday', 'is_weekend',
    
    # Interactions
    'demographic_covid_severity', 'country_resilience_score',
    'vaccination_effectiveness_adjusted', 'predicted_mortality_factor', 'epidemic_phase'
]

def get_horizon_category(horizon: int) -> str:
    """Détermine la catégorie d'un horizon"""
    if horizon in PREDICTION_HORIZONS['short_term']:
        return 'short_term'
    elif horizon in PREDICTION_HORIZONS['medium_term']:
        return 'medium_term'
    elif horizon in PREDICTION_HORIZONS['long_term']:
        return 'long_term'
    else:
        return 'custom'

def validate_horizons(requested_horizons: List[int]) -> Tuple[List[int], List[str]]:
    """Valide et catégorise les horizons demandés"""
    valid_horizons = []
    warnings = []
    
    for horizon in requested_horizons:
        if horizon in ALL_HORIZONS:
            valid_horizons.append(horizon)
        else:
            # Trouver l'horizon le plus proche
            closest = min(ALL_HORIZONS, key=lambda x: abs(x - horizon))
            valid_horizons.append(closest)
            warnings.append(f"Horizon {horizon}j non supporté, remplacé par {closest}j")
    
    return valid_horizons, warnings

class MultiHorizonCovidPredictor:
    """🚀 Prédicteur RÉVOLUTIONNAIRE Multi-Horizon"""
    
    def __init__(self):
        self.sequence_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vaccination_cache = {}
        self.demographics_cache = {}
        
        logger.info(f"🚀 Prédicteur Multi-Horizon initialisé sur {self.device}")
    
    def load_csv_data(self):
        """📂 Charge les données CSV multi-horizon"""
        global enriched_data
        
        try:
            enriched_file = os.path.join(CSV_DATA_PATH, 'enriched_covid_dataset_longterm.csv')
            
            if os.path.exists(enriched_file):
                logger.info(f"📂 Chargement dataset multi-horizon: {enriched_file}")
                enriched_data = pd.read_csv(enriched_file)
                enriched_data['date'] = pd.to_datetime(enriched_data['date'])
                logger.info(f"✅ Dataset multi-horizon chargé: {len(enriched_data)} lignes")
            else:
                logger.info("🔄 Création dataset multi-horizon...")
                global csv_pipeline
                csv_pipeline = CSVCovidDataPipeline(CSV_DATA_PATH)
                enriched_data = csv_pipeline.run_full_pipeline()
            
            self.update_caches_from_enriched_data()
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données: {e}")
            return False
    
    def update_caches_from_enriched_data(self):
        """📋 Met à jour les caches depuis les données enrichies"""
        if enriched_data is None:
            return
        
        # Cache vaccination et démographie
        for country in enriched_data['country_name'].unique():
            if pd.notna(country):
                country_data = enriched_data[enriched_data['country_name'] == country]
                
                # Cache vaccination
                vaccination_features = [col for col in enriched_data.columns 
                                      if any(vax_term in col for vax_term in ['vaccination', 'vax', 'coverage', 'protection'])]
                if vaccination_features:
                    vax_data = country_data[['date'] + vaccination_features].sort_values('date')
                    self.vaccination_cache[country.strip().lower()] = vax_data
                
                # Cache démographie
                demo_features = [col for col in enriched_data.columns 
                               if any(demo_term in col for demo_term in ['population', 'birth_rate', 'life_expectancy', 
                                                                         'elderly_ratio', 'vulnerability', 'resilience'])]
                if demo_features and len(country_data) > 0:
                    demo_profile = country_data[demo_features].mean().to_dict()
                    self.demographics_cache[country.strip().lower()] = demo_profile
        
        logger.info(f"📋 Caches mis à jour: {len(self.vaccination_cache)} pays")
    
    async def load_country_data_from_csv(self, country: str):
        """📊 Charge données pays depuis CSV multi-horizon"""
        try:
            if enriched_data is None:
                raise HTTPException(status_code=503, detail="Données CSV non chargées")
            
            country_data = enriched_data[enriched_data['country_name'] == country].copy()
            
            if len(country_data) == 0:
                raise HTTPException(status_code=404, detail=f"Aucune donnée pour {country}")
            
            country_data = country_data.sort_values('date')
            
            # Calculer active si manquant
            if 'active' not in country_data.columns:
                country_data['active'] = (
                    country_data['confirmed'] - 
                    country_data['deaths'] - 
                    country_data['recovered']
                )
            
            logger.info(f"📊 Données multi-horizon chargées pour {country}: {len(country_data)} points")
            return country_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur chargement {country}: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    def create_features_for_prediction(self, covid_df: pd.DataFrame, country: str, prediction_date: datetime):
        """🧠 Crée features pour prédiction multi-horizon"""
        
        # Features temporelles pour la date de prédiction
        temporal_features = {
            'month_sin': np.sin(2 * np.pi * prediction_date.month / 12),
            'month_cos': np.cos(2 * np.pi * prediction_date.month / 12),
            'day_of_year': prediction_date.timetuple().tm_yday,
            'quarter': (prediction_date.month - 1) // 3 + 1,
            'week_of_year': prediction_date.isocalendar()[1],
            'weekday': prediction_date.weekday(),
            'is_weekend': 1 if prediction_date.weekday() >= 5 else 0,
            'pandemic_year': (prediction_date - pd.to_datetime('2020-01-01')).days / 365.25,
            'pandemic_phase': min((prediction_date - pd.to_datetime('2020-01-01')).days / 730, 1.0),
            'seasonal_factor': np.cos(2 * np.pi * prediction_date.timetuple().tm_yday / 365.25),
        }
        
        # Features vaccination (selon date de prédiction!)
        vaccination_features = self.get_vaccination_features_csv(country, prediction_date)
        
        # Features démographiques (profil fixe)
        demographic_features = self.get_demographic_features_csv(country)
        
        # Features d'interaction
        interaction_features = {
            'demographic_covid_severity': (
                demographic_features.get('covid_vulnerability', 0.5) * 
                covid_df['mortality_rate'].iloc[-1] if len(covid_df) > 0 else 0
            ),
            'country_resilience_score': (
                demographic_features.get('demographic_resilience', 0.5) * 
                (1 - demographic_features.get('covid_vulnerability', 0.5))
            ),
            'vaccination_effectiveness_adjusted': (
                vaccination_features['protection_factor'] *
                (2 - demographic_features.get('covid_vulnerability', 0.5))
            ),
            'predicted_mortality_factor': (
                demographic_features.get('age_mortality_factor', 1.0) *
                vaccination_features['mortality_reduction_factor']
            ),
            'epidemic_phase': self.calculate_epidemic_phase(covid_df, vaccination_features),
        }
        
        # Fusion complète
        static_features = {
            **demographic_features,
            **vaccination_features,
            **interaction_features,
            **temporal_features
        }
        
        return static_features
    
    def get_vaccination_features_csv(self, country: str, target_date: datetime) -> Dict:
        """💉 Features vaccination selon date (logique intelligente)"""
        country_norm = country.strip().lower()
        
        features = {
            'has_vaccination': False,
            'coverage_percent': 0.0,
            'protection_factor': 0.0,
            'case_reduction_factor': 1.0,
            'mortality_reduction_factor': 1.0,
            'vaccination_momentum': 0.0,
        }
        
        # Recherche dans le cache
        if country_norm in self.vaccination_cache:
            vax_data = self.vaccination_cache[country_norm]
            
            # Trouver données <= target_date
            valid_data = vax_data[vax_data['date'] <= target_date]
            
            if len(valid_data) > 0:
                latest = valid_data.sort_values('date').iloc[-1]
                
                # Extraire features disponibles
                coverage = latest.get('coverage_percent', 0.0)
                protection = latest.get('protection_factor', 0.0)
                
                if coverage > 0.1:  # Au moins 0.1% de couverture
                    features.update({
                        'has_vaccination': True,
                        'coverage_percent': coverage,
                        'protection_factor': protection,
                        'case_reduction_factor': latest.get('case_reduction_factor', 1.0),
                        'mortality_reduction_factor': latest.get('mortality_reduction_factor', 1.0),
                        'vaccination_momentum': latest.get('vaccination_momentum', 0.0),
                    })
        
        return features
    
    def get_demographic_features_csv(self, country: str) -> Dict:
        """👥 Features démographiques depuis cache"""
        country_norm = country.strip().lower()
        
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
            'covid_vulnerability': 0.5,
            'demographic_resilience': 0.5,
            'age_mortality_factor': 1.0,
        }
        
        # Recherche dans le cache
        if country_norm in self.demographics_cache:
            cached_features = self.demographics_cache[country_norm]
            features.update(cached_features)
        
        return features
    
    def calculate_epidemic_phase(self, covid_df: pd.DataFrame, vaccination_features: Dict) -> float:
        """📊 Calcule phase épidémique"""
        if len(covid_df) == 0:
            return 0.5
        
        growth_rate = covid_df['growth_rate'].iloc[-1] if 'growth_rate' in covid_df.columns else 0
        has_vaccination = vaccination_features.get('has_vaccination', False)
        protection = vaccination_features.get('protection_factor', 0)
        
        if not has_vaccination:
            # Phase pré-vaccination
            if growth_rate > 0.05:
                return 1.0  # Croissance
            elif growth_rate > -0.05:
                return 0.5  # Plateau
            else:
                return 0.0  # Décroissance
        else:
            # Phase vaccination
            expected_decline = -protection * 0.1
            if growth_rate > expected_decline + 0.05:
                return 0.8  # Résistance
            elif growth_rate > expected_decline - 0.05:
                return 0.3  # Déclin attendu
            else:
                return 0.1  # Déclin rapide
    
    async def predict_multihorizon(self, country: str, region: str = None, 
                                  prediction_horizons: List[int] = [1, 7, 14, 30, 365],
                                  start_date: str = None, include_uncertainty: bool = True,
                                  include_attention: bool = False,
                                  horizon_category: str = "auto"):
        """🚀 Prédiction RÉVOLUTIONNAIRE Multi-Horizon"""
        try:
            if model is None:
                raise HTTPException(status_code=503, detail="Modèle non chargé")
            
            # Valider et ajuster les horizons
            valid_horizons, horizon_warnings = validate_horizons(prediction_horizons)
            
            # Filtrer selon catégorie si spécifiée
            if horizon_category != "auto":
                if horizon_category in PREDICTION_HORIZONS:
                    valid_horizons = [h for h in valid_horizons if h in PREDICTION_HORIZONS[horizon_category]]
                else:
                    raise HTTPException(status_code=400, detail=f"Catégorie {horizon_category} inconnue")
            
            # Charger données COVID
            covid_df = await self.load_country_data_from_csv(country)
            
            # Date de début
            if start_date:
                prediction_start = pd.to_datetime(start_date)
            else:
                prediction_start = covid_df['date'].max() + timedelta(days=1)
            
            # Préparer séquence d'entrée
            if len(covid_df) < self.sequence_length:
                raise HTTPException(status_code=400, detail=f"Pas assez de données pour {country}")
            
            recent_data = covid_df.tail(self.sequence_length).copy()
            
            # Créer séquence temporelle
            temporal_sequence = []
            for _, row in recent_data.iterrows():
                day_features = [row.get(f, 0) for f in TEMPORAL_FEATURES if f in row.index]
                temporal_sequence.append(day_features)
            
            temporal_sequence = np.array(temporal_sequence, dtype=np.float32)
            temporal_sequence_scaled = sequence_scaler.transform(
                temporal_sequence.reshape(-1, temporal_sequence.shape[-1])
            ).reshape(temporal_sequence.shape)
            
            # Prédictions pour tous les horizons demandés
            predictions_results = []
            
            for horizon in valid_horizons:
                # Date de prédiction pour cet horizon
                pred_date = prediction_start + timedelta(days=horizon)
                
                # Créer features statiques pour cette date
                static_features_dict = self.create_features_for_prediction(covid_df, country, pred_date)
                static_features = np.array([static_features_dict.get(f, 0) for f in STATIC_FEATURES], dtype=np.float32)
                static_features_scaled = static_scaler.transform(static_features.reshape(1, -1))[0]
                
                # Tenseurs
                temporal_tensor = torch.FloatTensor(temporal_sequence_scaled).unsqueeze(0).to(predictor.device)
                static_tensor = torch.FloatTensor(static_features_scaled).unsqueeze(0).to(predictor.device)
                
                # Prédiction pour cet horizon spécifique
                model.eval()
                with torch.no_grad():
                    pred, uncertainty, attention_weights = model(temporal_tensor, static_tensor, target_horizon=horizon)
                    
                    # Dénormaliser
                    print(f"Valeur normalisée: {pred.cpu().numpy()}")
                    pred_denorm = target_scaler.inverse_transform(pred.cpu().numpy())[0]
                    print(f"Valeur dénormalisée: {pred_denorm}")
                    uncertainty_denorm = uncertainty.cpu().numpy()[0] if include_uncertainty else None
                    
                    # Extraire prédictions
                    confirmed_pred = max(0, float(pred_denorm[0]))
                    deaths_pred = max(0, float(pred_denorm[1]))
                    recovered_pred = max(0, float(pred_denorm[2]))
                    
                    # 🧠 LOGIQUE INTELLIGENTE selon HORIZON
                    if horizon >= 365:  # Long terme (1 an+)
                        # Appliquer contraintes démographiques fortes
                        demo_features = static_features_dict
                        vulnerability = demo_features.get('covid_vulnerability', 0.5)
                        
                        # Croissance plus conservative
                        growth_factor = 1.1 ** (horizon / 365)  # 10% par an max
                        confirmed_pred = min(confirmed_pred, covid_df['confirmed'].iloc[-1] * growth_factor)
                        
                        # Mortalité selon vulnérabilité démographique
                        max_mortality_rate = 5 + (vulnerability * 10)  # 5-15% selon vulnérabilité
                        deaths_pred = min(deaths_pred, confirmed_pred * max_mortality_rate / 100)
                        
                        # Guérisons selon disponibilité vaccination
                        has_vax = static_features_dict.get('has_vaccination', False)
                        if has_vax:
                            min_recovery_rate = 70 + (static_features_dict.get('protection_factor', 0) * 20)  # 70-90%
                        else:
                            min_recovery_rate = 40 + (20 * (1 - vulnerability))  # 40-60% selon vulnérabilité
                        
                        recovered_pred = max(recovered_pred, confirmed_pred * min_recovery_rate / 100)
                    
                    elif horizon >= 90:  # Moyen terme (3-6 mois)
                        # Logique vaccination progressive
                        vax_coverage = static_features_dict.get('coverage_percent', 0)
                        if vax_coverage > 10:  # Si vaccination significative
                            # Réduction progressive des cas
                            reduction_factor = min(0.8, vax_coverage / 100)
                            confirmed_pred *= (1 - reduction_factor * 0.5)
                            deaths_pred *= (1 - reduction_factor * 0.8)  # Réduction plus forte des décès
                    
                    # CONTRAINTES UNIVERSELLES
                    deaths_pred = min(deaths_pred, confirmed_pred * 0.15)  # Max 15% mortalité
                    recovered_pred = min(recovered_pred, confirmed_pred * 0.95)  # Max 95% guérison
                    
                    # Cohérence mathématique
                    if deaths_pred + recovered_pred > confirmed_pred:
                        ratio = confirmed_pred * 0.95 / (deaths_pred + recovered_pred)
                        deaths_pred *= ratio
                        recovered_pred *= ratio
                    
                    active_pred = max(0, confirmed_pred - deaths_pred - recovered_pred)
                    
                    # Créer résultat
                    result = PredictionResult(
                        date=pred_date.strftime("%Y-%m-%d"),
                        horizon_days=horizon,
                        horizon_category=get_horizon_category(horizon),
                        confirmed=confirmed_pred,
                        deaths=deaths_pred,
                        recovered=recovered_pred,
                        active=active_pred
                    )
                    
                    # Intervalles de confiance
                    if include_uncertainty and uncertainty_denorm is not None:
                        result.confidence_intervals = {
                            'confirmed': {
                                'lower': max(0, confirmed_pred - 1.96 * abs(uncertainty_denorm[0])),
                                'upper': confirmed_pred + 1.96 * abs(uncertainty_denorm[0])
                            },
                            'deaths': {
                                'lower': max(0, deaths_pred - 1.96 * abs(uncertainty_denorm[1])),
                                'upper': min(confirmed_pred, deaths_pred + 1.96 * abs(uncertainty_denorm[1]))
                            },
                            'recovered': {
                                'lower': max(0, recovered_pred - 1.96 * abs(uncertainty_denorm[2])),
                                'upper': min(confirmed_pred, recovered_pred + 1.96 * abs(uncertainty_denorm[2]))
                            },
                            'active': {
                                'lower': max(0, active_pred * 0.8),
                                'upper': active_pred * 1.2
                            }
                        }
                    
                    # Impact vaccination pour ce horizon
                    if horizon >= 90:  # Moyen/long terme
                        result.vaccination_impact = {
                            'has_vaccination_data': static_features_dict.get('has_vaccination', False),
                            'coverage_percent': static_features_dict.get('coverage_percent', 0),
                            'protection_factor': static_features_dict.get('protection_factor', 0),
                            'expected_case_reduction': (1 - static_features_dict.get('case_reduction_factor', 1)) * 100
                        }
                    
                    # Contexte démographique pour long terme
                    if horizon >= 365:
                        result.demographic_context = {
                            'covid_vulnerability': static_features_dict.get('covid_vulnerability', 0.5),
                            'population_age_factor': static_features_dict.get('age_mortality_factor', 1.0),
                            'country_resilience': static_features_dict.get('country_resilience_score', 0.5)
                        }
                    
                    # Score attention
                    if include_attention and attention_weights:
                        avg_attention = torch.stack(attention_weights).mean().item()
                        result.attention_score = float(avg_attention)
                    
                    predictions_results.append(result)
            
            # Résumé des horizons
            horizons_summary = {
                'short_term': [h for h in valid_horizons if h in PREDICTION_HORIZONS['short_term']],
                'medium_term': [h for h in valid_horizons if h in PREDICTION_HORIZONS['medium_term']],
                'long_term': [h for h in valid_horizons if h in PREDICTION_HORIZONS['long_term']]
            }
            
            # Timeline vaccination
            vaccination_timeline = {
                'current_status': "Available" if static_features_dict.get('has_vaccination', False) else "Not Started",
                'coverage_percent': static_features_dict.get('coverage_percent', 0),
                'expected_impact_timeframe': "3-6 months" if static_features_dict.get('has_vaccination', False) else "Unknown"
            }
            
            # Profil démographique
            demographic_profile = {
                'population_millions': static_features_dict.get('population_millions', 50),
                'vulnerability_score': static_features_dict.get('covid_vulnerability', 0.5),
                'resilience_score': static_features_dict.get('demographic_resilience', 0.5),
                'elderly_ratio_percent': static_features_dict.get('elderly_ratio', 0.08) * 100
            }
            
            # Confiance modèle
            model_confidence = {
                'data_quality': min(1.0, len(covid_df) / 365),
                'horizon_coverage': len(valid_horizons) / len(prediction_horizons),
                'vaccination_data_reliability': 0.9 if static_features_dict.get('has_vaccination', False) else 0.6,
                'overall_confidence': 0.85
            }
            
            return {
                "predictions": predictions_results,
                "horizons_summary": horizons_summary,
                "vaccination_timeline": vaccination_timeline,
                "demographic_profile": demographic_profile,
                "model_confidence": model_confidence,
                "model_info": {
                    "model_type": "COVID Revolutionary Multi-Horizon Transformer v3.0",
                    "sequence_length": self.sequence_length,
                    "prediction_start_date": prediction_start.strftime("%Y-%m-%d"),
                    "last_data_date": covid_df['date'].max().strftime("%Y-%m-%d"),
                    "data_points_used": len(covid_df),
                    "features_count": len(TEMPORAL_FEATURES) + len(STATIC_FEATURES),
                    "device": str(self.device),
                    "horizons_supported": str(ALL_HORIZONS),
                    "data_source": "Multi-Horizon CSV Pipeline",
                    "warnings": str(horizon_warnings) if horizon_warnings else ""
                }
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur prédiction multi-horizon: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Instance du prédicteur
predictor = MultiHorizonCovidPredictor()

@app.on_event("startup")
async def startup_event():
    """🚀 Initialisation API Multi-Horizon"""
    global model, sequence_scaler, static_scaler, target_scaler, model_config
    
    logger.info("🚀 Démarrage API Révolutionnaire Multi-Horizon v3.0...")
    
    # Charger données CSV
    csv_loaded = predictor.load_csv_data()
    if not csv_loaded:
        logger.warning("⚠️ Données CSV non chargées - API en mode dégradé")
    
    # Chargement modèle
    try:
        model_path = os.path.join(MODEL_DIR, 'covid_revolutionary_longterm_model.pth')
        config_path = os.path.join(MODEL_DIR, 'revolutionary_longterm_config.json')
        sequence_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_longterm_sequence_scaler.pkl')
        static_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_longterm_static_scaler.pkl')
        target_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_longterm_target_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            # Config
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_config = config['model_config']
                features = config['features']
            
            # Modèle
            model = CovidRevolutionaryLongTermTransformer(
                sequence_features=features['sequence_features'],
                static_features=features['static_features'],
                prediction_horizons=ALL_HORIZONS,
                **model_config
            ).to(predictor.device)
            
            model.load_state_dict(torch.load(model_path, map_location=predictor.device))
            model.eval()
            logger.info("✅ Modèle multi-horizon chargé")
            
            # Scalers
            if os.path.exists(sequence_scaler_path):
                sequence_scaler = joblib.load(sequence_scaler_path)
                logger.info("✅ Sequence scaler chargé")
            
            if os.path.exists(static_scaler_path):
                static_scaler = joblib.load(static_scaler_path)
                logger.info("✅ Static scaler chargé")
            
            if os.path.exists(target_scaler_path):
                target_scaler = joblib.load(target_scaler_path)
                logger.info("✅ Target scaler chargé")
        else:
            logger.warning("⚠️ Modèle multi-horizon non trouvé")
    
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
    
    logger.info("✅ API Multi-Horizon prête!")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """🏥 Health check multi-horizon"""
    csv_data_available = enriched_data is not None
    model_loaded = model is not None
    
    countries_count = 0
    if csv_data_available:
        try:
            countries_count = enriched_data['country_name'].nunique()
        except:
            pass
    
    # Performance par catégorie d'horizon
    model_performance = None
    if model_config:
        try:
            with open(os.path.join(MODEL_DIR, 'revolutionary_longterm_config.json'), 'r') as f:
                config = json.load(f)
                if 'training_history' in config and config['training_history'].get('horizon_metrics'):
                    metrics = config['training_history']['horizon_metrics']
                    model_performance = {
                        'short_term_avg_r2': np.mean([metrics.get(str(h), [0])[-1] for h in [1, 7, 14, 30] if str(h) in metrics]),
                        'medium_term_avg_r2': np.mean([metrics.get(str(h), [0])[-1] for h in [90, 180] if str(h) in metrics]),
                        'long_term_avg_r2': np.mean([metrics.get(str(h), [0])[-1] for h in [365, 730, 1825] if str(h) in metrics])
                    }
        except:
            pass
    
    return HealthCheck(
        status="revolutionary_multihorizon" if all([csv_data_available, model_loaded]) else "partial",
        model_loaded=model_loaded,
        csv_data_available=csv_data_available,
        supported_horizons=PREDICTION_HORIZONS,
        countries_available=countries_count,
        model_performance=model_performance,
        data_source="CSV Files Multi-Horizon"
    )

@app.post("/predict", response_model=MultiHorizonPredictionResponse)
async def predict_covid_multihorizon(request: MultiHorizonPredictionRequest):
    """🚀 Prédiction COVID multi-horizon révolutionnaire"""
    logger.info(f"🧠 Prédiction multi-horizon pour {request.country}, horizons: {request.prediction_horizons}")
    
    try:
        result = await predictor.predict_multihorizon(
            country=request.country,
            region=request.region,
            prediction_horizons=request.prediction_horizons,
            start_date=request.start_date,
            include_uncertainty=request.include_uncertainty,
            include_attention=request.include_attention,
            horizon_category=request.horizon_category
        )
        
        return MultiHorizonPredictionResponse(
            country=request.country,
            region=request.region,
            prediction_start_date=result["model_info"]["prediction_start_date"],
            model_info=result["model_info"],
            predictions=result["predictions"],
            horizons_summary=result["horizons_summary"],
            vaccination_timeline=result["vaccination_timeline"],
            demographic_profile=result["demographic_profile"],
            model_confidence=result["model_confidence"],
            prediction_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur multi-horizon: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur détaillée: {str(e)}")

@app.get("/countries")
async def get_available_countries():
    """📋 Liste des pays disponibles"""
    try:
        if enriched_data is None:
            raise HTTPException(status_code=503, detail="Données CSV non chargées")
        
        countries_list = sorted(enriched_data['country_name'].dropna().unique().tolist())
        return {
            "countries": countries_list, 
            "count": len(countries_list),
            "multihorizon_features_available": True,
            "supported_horizons": PREDICTION_HORIZONS,
            "data_source": "CSV Files Multi-Horizon"
        }
    except Exception as e:
        logger.error(f"Erreur pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération pays")

@app.get("/horizons")
async def get_supported_horizons():
    """📅 Horizons de prédiction supportés"""
    return {
        "supported_horizons": PREDICTION_HORIZONS,
        "all_horizons": ALL_HORIZONS,
        "horizon_descriptions": {
            "short_term": "Court terme - Prédictions précises pour planification immédiate",
            "medium_term": "Moyen terme - Tendances pour planification stratégique",
            "long_term": "Long terme - Projections pour politique de santé publique"
        },
        "model_type": "Multi-Horizon Revolutionary Transformer"
    }

@app.get("/model/performance")
async def get_model_performance():
    """📊 Performance détaillée multi-horizon"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        with open(os.path.join(MODEL_DIR, 'revolutionary_longterm_config.json'), 'r') as f:
            config = json.load(f)
        
        return {
            "model_architecture": {
                "type": "Revolutionary Multi-Horizon Transformer v3.0",
                "parameters": sum(p.numel() for p in model.parameters()),
                "specialized_heads": 3,  # Court, moyen, long terme
                "horizon_categories": config.get('horizon_categories', PREDICTION_HORIZONS),
                "adaptive_logic": "Vaccination + Demographic context-aware"
            },
            "performance_by_horizon": config.get('training_history', {}).get('horizon_metrics', {}),
            "data_sources": {
                "covid_timeseries": "CSV Multi-Horizon Pipeline", 
                "vaccination_data": "Progressive vaccination logic", 
                "demographics": "Fixed demographic profiles",
                "features_engineered": len(TEMPORAL_FEATURES) + len(STATIC_FEATURES)
            },
            "capabilities": {
                "multi_horizon_prediction": ALL_HORIZONS,
                "adaptive_vaccination_logic": True,
                "demographic_integration": True,
                "uncertainty_estimation": True,
                "long_term_projections": True,
                "vaccination_timeline_awareness": True
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur performance: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération performance")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("covid_api:app", host="0.0.0.0", port=8000, reload=True)
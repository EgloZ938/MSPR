from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import joblib
import os
import json
import logging
from pathlib import Path

# Import du modèle révolutionnaire
import sys
sys.path.append('.')
from covid_ai_model import CovidRevolutionaryTransformer
from covid_data_pipeline import CSVCovidDataPipeline
from dotenv import load_dotenv
load_dotenv()

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="COVID-19 Revolutionary AI API - CSV Edition", 
    version="2.1.0",
    description="API révolutionnaire avec Transformer hybride pour prédictions COVID multi-horizons (Version CSV Pure)"
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

# Modèles Pydantic (identiques à la version MongoDB)
class RevolutionaryPredictionRequest(BaseModel):
    country: str = Field(..., description="Nom du pays")
    region: Optional[str] = Field(None, description="Région (optionnel)")
    prediction_horizons: List[int] = Field([1, 7, 14, 30], description="Horizons de prédiction en jours")
    start_date: Optional[str] = Field(None, description="Date de début des prédictions (YYYY-MM-DD)")
    include_uncertainty: bool = Field(True, description="Inclure les intervalles de confiance")
    include_attention: bool = Field(False, description="Inclure les poids d'attention")

class PredictionResult(BaseModel):
    date: str
    horizon_days: int
    confirmed: float
    deaths: float
    recovered: float
    active: float
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    attention_score: Optional[float] = None

class RevolutionaryPredictionResponse(BaseModel):
    country: str
    region: Optional[str]
    prediction_start_date: str
    model_info: Dict[str, Union[str, int, float]]
    predictions: List[PredictionResult]
    vaccination_impact: Dict[str, Union[str, float]]
    demographic_factors: Dict[str, Union[str, float]]
    model_confidence: Dict[str, float]
    prediction_timestamp: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    csv_data_available: bool
    revolutionary_features_count: int
    countries_available: int
    model_performance: Optional[Dict[str, float]] = None
    data_source: str = "CSV Files"

# Variables globales
model = None
sequence_scaler = None
static_scaler = None
target_scaler = None
model_config = None
csv_pipeline = None
enriched_data = None

# Features lists (synchronisées avec l'entraînement)
TEMPORAL_FEATURES = [
    'confirmed', 'deaths', 'recovered', 'active',
    'new_cases', 'new_deaths', 'new_recovered',
    'new_cases_ma7', 'new_deaths_ma7',
    'growth_rate', 'mortality_rate', 'recovery_rate', 'trend_7d',
    'month_sin', 'month_cos'
]

STATIC_FEATURES = [
    'population_millions', 'birth_rate', 'mortality_rate', 'life_expectancy',
    'infant_mortality_rate', 'fertility_rate', 'growth_rate', 'elderly_ratio',
    'demographic_vulnerability',
    'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate_7d',
    'vaccination_acceleration', 'vaccination_coverage_est', 'vaccination_momentum',
    'days_since_vax_start',
    'vax_per_capita', 'elderly_vax_urgency', 'demographic_covid_risk',
    'vax_effectiveness_lag',
    'day_of_year', 'quarter', 'week_of_year', 'weekday', 'is_weekend'
]

class CSVRevolutionaryCovidPredictor:
    """🚀 Prédicteur révolutionnaire 100% CSV"""
    
    def __init__(self):
        self.sequence_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vaccination_cache = {}
        self.demographics_cache = {}
        
        logger.info(f"🚀 Prédicteur CSV révolutionnaire initialisé sur {self.device}")
    
    def load_csv_data(self):
        """🗂️ Charge les données CSV enrichies"""
        global enriched_data
        
        try:
            # Essayer de charger le dataset enrichi s'il existe
            enriched_file = os.path.join(CSV_DATA_PATH, 'enriched_covid_dataset.csv')
            
            if os.path.exists(enriched_file):
                logger.info(f"📂 Chargement du dataset enrichi: {enriched_file}")
                enriched_data = pd.read_csv(enriched_file)
                enriched_data['date'] = pd.to_datetime(enriched_data['date'])
                logger.info(f"✅ Dataset enrichi chargé: {len(enriched_data)} lignes")
            else:
                # Créer le dataset enrichi à la volée
                logger.info("🔄 Création du dataset enrichi à partir des CSV...")
                global csv_pipeline
                csv_pipeline = CSVCovidDataPipeline(CSV_DATA_PATH)
                enriched_data = csv_pipeline.run_full_pipeline()
            
            # Mettre à jour les caches
            self.update_caches_from_enriched_data()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données CSV: {e}")
            return False
    
    def update_caches_from_enriched_data(self):
        """📋 Met à jour les caches depuis les données enrichies"""
        if enriched_data is None:
            return
        
        # Cache vaccination depuis les données enrichies
        vaccination_cols = [col for col in enriched_data.columns if 'vaccination' in col or 'vax' in col]
        if vaccination_cols:
            for country in enriched_data['country_name'].unique():
                if pd.notna(country):
                    country_data = enriched_data[enriched_data['country_name'] == country][
                        ['date'] + vaccination_cols
                    ].sort_values('date')
                    self.vaccination_cache[country.strip().lower()] = country_data
        
        logger.info(f"📋 Caches mis à jour: {len(self.vaccination_cache)} pays")
    
    async def load_country_data_from_csv(self, country: str):
        """📊 Charge les données COVID pour un pays depuis le CSV enrichi"""
        try:
            if enriched_data is None:
                raise HTTPException(status_code=503, detail="Données CSV non chargées")
            
            # Filtrer par pays
            country_data = enriched_data[enriched_data['country_name'] == country].copy()
            
            if len(country_data) == 0:
                raise HTTPException(status_code=404, detail=f"Aucune donnée pour {country}")
            
            # Trier par date
            country_data = country_data.sort_values('date')
            
            # S'assurer que les colonnes essentielles existent
            if 'active' not in country_data.columns:
                country_data['active'] = (
                    country_data['confirmed'] - 
                    country_data['deaths'] - 
                    country_data['recovered']
                )
            
            logger.info(f"📊 Données CSV chargées pour {country}: {len(country_data)} points")
            return country_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur chargement CSV {country}: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    def get_vaccination_features_csv(self, country: str, target_date: datetime):
        """💉 Récupère features vaccination depuis les données CSV"""
        country_norm = country.strip().lower()
        
        features = {
            'cumulative_vaccinations': 0,
            'daily_vaccinations': 0,
            'vaccination_rate_7d': 0,
            'vaccination_acceleration': 0,
            'vaccination_coverage_est': 0,
            'vaccination_momentum': 0,
            'days_since_vax_start': 0,
        }
        
        # Recherche dans les données enrichies
        if enriched_data is not None:
            country_data = enriched_data[
                enriched_data['country_name'].str.lower() == country_norm
            ]
            
            if len(country_data) > 0:
                # Trouver la date la plus proche
                country_data = country_data.copy()
                country_data['date_diff'] = abs((country_data['date'] - target_date).dt.days)
                
                valid_dates = country_data[country_data['date_diff'] <= 30]
                
                if len(valid_dates) > 0:
                    closest_data = valid_dates.loc[valid_dates['date_diff'].idxmin()]
                    
                    # Remplir les features disponibles
                    for feature in features.keys():
                        if feature in closest_data:
                            features[feature] = float(closest_data[feature])
        
        return features
    
    def get_demographic_features_csv(self, country: str):
        """👥 Récupère features démographiques depuis les données CSV"""
        country_norm = country.strip().lower()
        
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
        
        # Recherche dans les données enrichies
        if enriched_data is not None:
            country_data = enriched_data[
                enriched_data['country_name'].str.lower() == country_norm
            ]
            
            if len(country_data) > 0:
                # Prendre les valeurs moyennes (car constantes pour un pays)
                latest_data = country_data.iloc[-1]
                
                # Remplir les features disponibles
                for feature in features.keys():
                    if feature in latest_data:
                        features[feature] = float(latest_data[feature])
        
        return features
    
    def create_features_for_prediction_csv(self, covid_df: pd.DataFrame, country: str, prediction_date: datetime):
        """🧠 Crée toutes les features pour la prédiction depuis CSV"""
        
        # Features temporelles pour la date de prédiction
        temporal_features = {
            'month_sin': np.sin(2 * np.pi * prediction_date.month / 12),
            'month_cos': np.cos(2 * np.pi * prediction_date.month / 12),
            'day_of_year': prediction_date.timetuple().tm_yday,
            'quarter': (prediction_date.month - 1) // 3 + 1,
            'week_of_year': prediction_date.isocalendar()[1],  # Fix: [1] pour récupérer la semaine
            'weekday': prediction_date.weekday(),
            'is_weekend': 1 if prediction_date.weekday() >= 5 else 0,
        }
        
        # Features vaccination
        vaccination_features = self.get_vaccination_features_csv(country, prediction_date)
        
        # Features démographiques
        demographic_features = self.get_demographic_features_csv(country)
        
        # Features d'interaction
        interaction_features = {
            'vax_per_capita': vaccination_features['cumulative_vaccinations'] / 
                            (demographic_features['population_millions'] * 1000 + 1),
            'elderly_vax_urgency': demographic_features['elderly_ratio'] * 
                                 (1 - vaccination_features['vaccination_coverage_est']),
            'demographic_covid_risk': demographic_features['demographic_vulnerability'] * 
                                    covid_df['mortality_rate'].iloc[-1] if len(covid_df) > 0 else 0,
            'vax_effectiveness_lag': vaccination_features['cumulative_vaccinations'] * 
                                   np.exp(-vaccination_features['days_since_vax_start'] / 21),
        }
        
        # Combiner toutes les features statiques
        static_features = {
            **demographic_features,
            **vaccination_features,
            **interaction_features,
            **temporal_features
        }
        
        return static_features
    
    async def predict_revolutionary_csv(self, country: str, region: str = None, 
                                      prediction_horizons: List[int] = [1, 7, 14, 30],
                                      start_date: str = None, include_uncertainty: bool = True,
                                      include_attention: bool = False):
        """🚀 Prédiction révolutionnaire multi-horizons depuis CSV"""
        try:
            if model is None:
                raise HTTPException(status_code=503, detail="Modèle non chargé")
            
            # Charger données COVID depuis CSV
            covid_df = await self.load_country_data_from_csv(country)
            
            # Date de début prédiction
            if start_date:
                prediction_start = pd.to_datetime(start_date)
            else:
                prediction_start = covid_df['date'].max() + timedelta(days=1)
            
            # Préparer séquence temporelle (30 derniers jours)
            if len(covid_df) < self.sequence_length:
                raise HTTPException(status_code=400, detail=f"Pas assez de données historiques pour {country}")
            
            # Features temporelles des 30 derniers jours
            recent_data = covid_df.tail(self.sequence_length).copy()
            
            # Créer la séquence d'entrée
            temporal_sequence = []
            for _, row in recent_data.iterrows():
                day_features = [row.get(f, 0) for f in TEMPORAL_FEATURES if f in row.index]
                temporal_sequence.append(day_features)
            
            temporal_sequence = np.array(temporal_sequence, dtype=np.float32)
            
            # Normaliser la séquence
            temporal_sequence_scaled = sequence_scaler.transform(
                temporal_sequence.reshape(-1, temporal_sequence.shape[-1])
            ).reshape(temporal_sequence.shape)
            
            # Créer features statiques
            static_features_dict = self.create_features_for_prediction_csv(covid_df, country, prediction_start)
            static_features = np.array([static_features_dict.get(f, 0) for f in STATIC_FEATURES], dtype=np.float32)
            static_features_scaled = static_scaler.transform(static_features.reshape(1, -1))[0]
            
            # Convertir en tenseurs
            temporal_tensor = torch.FloatTensor(temporal_sequence_scaled).unsqueeze(0).to(predictor.device)
            static_tensor = torch.FloatTensor(static_features_scaled).unsqueeze(0).to(predictor.device)
            
            # Prédictions pour tous les horizons
            model.eval()
            predictions_results = []
            
            with torch.no_grad():
                for horizon in prediction_horizons:
                    if horizon in [1, 7, 14, 30]:  # Horizons supportés
                        pred, uncertainty, attention_weights = model(temporal_tensor, static_tensor, target_horizon=horizon)
                        
                        # Dénormaliser
                        pred_denorm = target_scaler.inverse_transform(pred.cpu().numpy())[0]
                        uncertainty_denorm = uncertainty.cpu().numpy()[0] if include_uncertainty else None
                        
                        # Date de prédiction
                        pred_date = prediction_start + timedelta(days=horizon)
                        
                        # Créer résultat
                        result = PredictionResult(
                            date=pred_date.strftime("%Y-%m-%d"),
                            horizon_days=horizon,
                            confirmed=max(0, float(pred_denorm[0])),
                            deaths=max(0, float(pred_denorm[1])),
                            recovered=max(0, float(pred_denorm[2])),
                            active=max(0, float(pred_denorm[3]))
                        )
                        
                        # Intervalles de confiance
                        if include_uncertainty and uncertainty_denorm is not None:
                            result.confidence_intervals = {
                                'confirmed': {
                                    'lower': max(0, float(pred_denorm[0] - 1.96 * uncertainty_denorm[0])),
                                    'upper': float(pred_denorm[0] + 1.96 * uncertainty_denorm[0])
                                },
                                'deaths': {
                                    'lower': max(0, float(pred_denorm[1] - 1.96 * uncertainty_denorm[1])),
                                    'upper': float(pred_denorm[1] + 1.96 * uncertainty_denorm[1])
                                },
                                'recovered': {
                                    'lower': max(0, float(pred_denorm[2] - 1.96 * uncertainty_denorm[2])),
                                    'upper': float(pred_denorm[2] + 1.96 * uncertainty_denorm[2])
                                },
                                'active': {
                                    'lower': max(0, float(pred_denorm[3] - 1.96 * uncertainty_denorm[3])),
                                    'upper': float(pred_denorm[3] + 1.96 * uncertainty_denorm[3])
                                }
                            }
                        
                        # Score d'attention
                        if include_attention and attention_weights:
                            avg_attention = torch.stack(attention_weights).mean().item()
                            result.attention_score = float(avg_attention)
                        
                        predictions_results.append(result)
            
            # Impact vaccination
            vaccination_impact = {
                'current_coverage': static_features_dict['vaccination_coverage_est'],
                'vaccination_momentum': static_features_dict['vaccination_momentum'],
                'effectiveness_score': static_features_dict['vax_effectiveness_lag'],
                'data_available': True
            }
            
            # Facteurs démographiques
            demographic_factors = {
                'population_millions': static_features_dict['population_millions'],
                'elderly_ratio': static_features_dict['elderly_ratio'],
                'vulnerability_score': static_features_dict['demographic_vulnerability'],
                'life_expectancy': static_features_dict['life_expectancy']
            }
            
            # Confiance du modèle
            model_confidence = {
                'data_quality': min(1.0, len(covid_df) / 365),
                'vaccination_data_coverage': 1.0 if static_features_dict['cumulative_vaccinations'] > 0 else 0.3,
                'demographic_data_coverage': 0.9,
                'overall_confidence': 0.8
            }
            
            return {
                "predictions": predictions_results,
                "vaccination_impact": vaccination_impact,
                "demographic_factors": demographic_factors,
                "model_confidence": model_confidence,
                "model_info": {
                    "model_type": "COVID Revolutionary Transformer v2.1 CSV",
                    "sequence_length": self.sequence_length,
                    "prediction_start_date": prediction_start.strftime("%Y-%m-%d"),
                    "last_data_date": covid_df['date'].max().strftime("%Y-%m-%d"),
                    "data_points_used": len(covid_df),
                    "features_count": len(TEMPORAL_FEATURES) + len(STATIC_FEATURES),
                    "device": str(self.device),
                    "horizons_supported": [1, 7, 14, 30],
                    "data_source": "CSV Files"
                }
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur prédiction CSV: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Instance du prédicteur
predictor = CSVRevolutionaryCovidPredictor()

@app.on_event("startup")
async def startup_event():
    """🚀 Initialisation révolutionnaire CSV"""
    global model, sequence_scaler, static_scaler, target_scaler, model_config
    
    logger.info("🚀 Démarrage API Révolutionnaire COVID IA v2.1 CSV...")
    
    # Charger les données CSV
    csv_loaded = predictor.load_csv_data()
    if not csv_loaded:
        logger.warning("⚠️ Données CSV non chargées - API en mode dégradé")
    
    # Chargement du modèle révolutionnaire
    try:
        model_path = os.path.join(MODEL_DIR, 'covid_revolutionary_model.pth')
        config_path = os.path.join(MODEL_DIR, 'revolutionary_config.json')
        sequence_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_sequence_scaler.pkl')
        static_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_static_scaler.pkl')
        target_scaler_path = os.path.join(MODEL_DIR, 'revolutionary_target_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            # Charger configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_config = config['model_config']
                features = config['features']
            
            # Créer et charger le modèle
            model = CovidRevolutionaryTransformer(
                sequence_features=features['sequence_features'],
                static_features=features['static_features'],
                **model_config
            ).to(predictor.device)
            
            model.load_state_dict(torch.load(model_path, map_location=predictor.device))
            model.eval()
            logger.info("✅ Modèle révolutionnaire CSV chargé")
            
            # Charger les scalers
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
            logger.warning("⚠️ Modèle révolutionnaire non trouvé")
    
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
    
    logger.info("✅ API Révolutionnaire CSV prête!")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """🏥 Health check révolutionnaire CSV"""
    csv_data_available = enriched_data is not None
    model_loaded = model is not None
    
    countries_count = 0
    if csv_data_available:
        try:
            countries_count = enriched_data['country_name'].nunique()
        except:
            pass
    
    # Performance du modèle
    model_performance = None
    if model_config:
        try:
            with open(os.path.join(MODEL_DIR, 'revolutionary_config.json'), 'r') as f:
                config = json.load(f)
                if 'training_history' in config and config['training_history'].get('val_metrics'):
                    latest_metrics = config['training_history']['val_metrics'][-1]
                    model_performance = {
                        'confirmed_r2': latest_metrics.get('confirmed_r2', 0),
                        'deaths_r2': latest_metrics.get('deaths_r2', 0),
                        'overall_mae': latest_metrics.get('confirmed_mae', 0)
                    }
        except:
            pass
    
    return HealthCheck(
        status="revolutionary_csv" if all([csv_data_available, model_loaded]) else "partial",
        model_loaded=model_loaded,
        csv_data_available=csv_data_available,
        revolutionary_features_count=len(TEMPORAL_FEATURES) + len(STATIC_FEATURES),
        countries_available=countries_count,
        model_performance=model_performance,
        data_source="CSV Files"
    )

@app.post("/predict", response_model=RevolutionaryPredictionResponse)
async def predict_covid_revolutionary_csv(request: RevolutionaryPredictionRequest):
    """🚀 Prédiction COVID révolutionnaire multi-horizons depuis CSV"""
    logger.info(f"🧠 Prédiction révolutionnaire CSV pour {request.country}, horizons: {request.prediction_horizons}")
    
    try:
        result = await predictor.predict_revolutionary_csv(
            country=request.country,
            region=request.region,
            prediction_horizons=request.prediction_horizons,
            start_date=request.start_date,
            include_uncertainty=request.include_uncertainty,
            include_attention=request.include_attention
        )
        
        return RevolutionaryPredictionResponse(
            country=request.country,
            region=request.region,
            prediction_start_date=result["model_info"]["prediction_start_date"],
            model_info=result["model_info"],
            predictions=result["predictions"],
            vaccination_impact=result["vaccination_impact"],
            demographic_factors=result["demographic_factors"],
            model_confidence=result["model_confidence"],
            prediction_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur CSV: {e}")
        raise HTTPException(status_code=500, detail="Erreur serveur révolutionnaire CSV")

@app.get("/countries")
async def get_available_countries_csv():
    """📋 Liste des pays disponibles dans les CSV"""
    try:
        if enriched_data is None:
            raise HTTPException(status_code=503, detail="Données CSV non chargées")
        
        countries_list = sorted(enriched_data['country_name'].dropna().unique().tolist())
        return {
            "countries": countries_list, 
            "count": len(countries_list),
            "revolutionary_features_available": True,
            "data_source": "CSV Files"
        }
    except Exception as e:
        logger.error(f"Erreur pays CSV: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération pays CSV")

@app.get("/model/performance")
async def get_model_performance_csv():
    """📊 Performance détaillée du modèle révolutionnaire CSV"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        with open(os.path.join(MODEL_DIR, 'revolutionary_config.json'), 'r') as f:
            config = json.load(f)
        
        return {
            "model_architecture": {
                "type": "Revolutionary Transformer + LSTM (CSV)",
                "parameters": sum(p.numel() for p in model.parameters()),
                "layers": config['model_config']['n_layers'],
                "attention_heads": config['model_config']['n_heads'],
                "model_dimension": config['model_config']['d_model']
            },
            "performance_metrics": config.get('training_history', {}).get('val_metrics', [])[-5:] if config.get('training_history') else [],
            "data_sources": {
                "covid_timeseries": "CSV Files (covid_19_clean_complete_clean.csv)", 
                "vaccination_data": "CSV (cumulative-covid-vaccinations_clean.csv)", 
                "demographics": "CSV (consolidated_demographics_data.csv)",
                "features_engineered": len(TEMPORAL_FEATURES) + len(STATIC_FEATURES)
            },
            "capabilities": {
                "multi_horizon_prediction": [1, 7, 14, 30],
                "uncertainty_estimation": True,
                "attention_visualization": True,
                "vaccination_impact_analysis": True,
                "demographic_integration": True,
                "csv_native": True
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur performance CSV: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération performance")

@app.get("/vaccination/{country}")
async def get_vaccination_analysis_csv(country: str, date: str = None):
    """💉 Analyse vaccination révolutionnaire CSV pour un pays"""
    if date:
        target_date = pd.to_datetime(date)
    else:
        target_date = datetime.now()
    
    vaccination_features = predictor.get_vaccination_features_csv(country, target_date)
    demographic_features = predictor.get_demographic_features_csv(country)
    
    # Analyse avancée
    analysis = {
        "vaccination_status": {
            "cumulative_vaccinations": vaccination_features['cumulative_vaccinations'],
            "daily_rate_7d": vaccination_features['vaccination_rate_7d'],
            "momentum": vaccination_features['vaccination_momentum'],
            "coverage_estimate": vaccination_features['vaccination_coverage_est']
        },
        "demographic_context": {
            "population_millions": demographic_features['population_millions'],
            "elderly_ratio": demographic_features['elderly_ratio'],
            "vulnerability_score": demographic_features['demographic_vulnerability']
        },
        "risk_assessment": {
            "elderly_vaccination_urgency": demographic_features['elderly_ratio'] * (1 - vaccination_features['vaccination_coverage_est']),
            "vaccination_per_capita": vaccination_features['cumulative_vaccinations'] / (demographic_features['population_millions'] * 1000 + 1),
            "effectiveness_with_lag": vaccination_features['cumulative_vaccinations'] * np.exp(-vaccination_features['days_since_vax_start'] / 21)
        }
    }
    
    return {
        "country": country,
        "analysis_date": target_date.strftime('%Y-%m-%d'),
        "revolutionary_analysis": analysis,
        "data_sources_available": {
            "vaccination": vaccination_features['cumulative_vaccinations'] > 0,
            "demographics": demographic_features['population_millions'] > 0
        },
        "data_source": "CSV Files"
    }

@app.get("/csv/data-info")
async def get_csv_data_info():
    """📊 Informations sur les données CSV chargées"""
    if enriched_data is None:
        raise HTTPException(status_code=503, detail="Données CSV non chargées")
    
    try:
        return {
            "dataset_info": {
                "total_rows": len(enriched_data),
                "total_features": len(enriched_data.columns),
                "countries_count": enriched_data['country_name'].nunique(),
                "date_range": {
                    "start": enriched_data['date'].min().strftime('%Y-%m-%d'),
                    "end": enriched_data['date'].max().strftime('%Y-%m-%d'),
                    "days": (enriched_data['date'].max() - enriched_data['date'].min()).days
                }
            },
            "feature_categories": {
                "temporal_features": len(TEMPORAL_FEATURES),
                "static_features": len(STATIC_FEATURES),
                "total_revolutionary_features": len(TEMPORAL_FEATURES) + len(STATIC_FEATURES)
            },
            "data_quality": {
                "missing_values_pct": (enriched_data.isnull().sum().sum() / (len(enriched_data) * len(enriched_data.columns)) * 100),
                "countries_with_full_data": len(enriched_data.groupby('country_name').size()[enriched_data.groupby('country_name').size() > 300])
            },
            "data_source": "CSV Files - Revolutionary Pipeline"
        }
    
    except Exception as e:
        logger.error(f"Erreur info CSV: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération info CSV")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("covid_api:app", host="0.0.0.0", port=8000, reload=True)
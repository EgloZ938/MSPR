from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import joblib
import os
from pymongo import MongoClient
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="COVID-19 Simple Intelligent AI", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'covid_dashboard')
CSV_DATA_PATH = '../data/dataset_clean'

# Modèle LSTM Simple mais Intelligent
class SimpleIntelligentCovidLSTM(nn.Module):
    """Modèle LSTM intelligent mais SIMPLE avec MongoDB + Vaccination seulement"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, enriched_features=12, dropout=0.2):
        super(SimpleIntelligentCovidLSTM, self).__init__()
        
        # LSTM pour les données COVID
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Réseau pour les features enrichies (vaccination + temporel)
        self.enriched_fc = nn.Sequential(
            nn.Linear(enriched_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Réseau de fusion simple mais efficace
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # confirmed, deaths, recovered, active
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, time_series, enriched_features):
        batch_size = time_series.size(0)
        
        lstm_out, _ = self.lstm(time_series)
        lstm_features = lstm_out[:, -1, :]
        
        enriched_processed = self.enriched_fc(enriched_features)
        
        combined_features = torch.cat([lstm_features, enriched_processed], dim=1)
        output = self.fusion_fc(combined_features)
        
        return output

# Modèles Pydantic
class SimplePredictionRequest(BaseModel):
    country: str
    region: Optional[str] = None
    days_to_predict: int = 14
    start_date: Optional[str] = None
    use_vaccination_data: bool = True

class SimplePredictionResponse(BaseModel):
    country: str
    region: Optional[str]
    prediction_start_date: str
    predictions: List[Dict[str, Union[str, int, float]]]
    confidence_intervals: List[Dict[str, Union[str, int, float]]]
    vaccination_impact: Dict[str, Union[str, float]]
    model_info: Dict[str, Union[str, int, float]]
    prediction_date: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    mongodb_connected: bool
    vaccination_data_available: bool
    countries_count: int

# Variables globales
model = None
time_scaler = None
enriched_scaler = None
mongodb_client = None
db = None
vaccination_cache = {}

class SimpleIntelligentPredictor:
    def __init__(self):
        self.sequence_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🧠 Prédicteur SIMPLE + INTELLIGENT sur {self.device}")
    
    async def connect_mongodb(self):
        """Connexion à MongoDB"""
        global mongodb_client, db
        try:
            mongodb_client = MongoClient(MONGO_URI)
            db = mongodb_client[DB_NAME]
            db.command('ping')
            logger.info("✅ Connexion MongoDB réussie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion MongoDB: {e}")
            return False
    
    def load_vaccination_cache(self):
        """Charge les données de vaccination en cache"""
        global vaccination_cache
        
        try:
            vaccination_file = os.path.join(CSV_DATA_PATH, 'cumulative-covid-vaccinations_clean.csv')
            
            if os.path.exists(vaccination_file):
                logger.info(f"💉 Chargement vaccination cache...")
                
                vacc_df = pd.read_csv(vaccination_file)
                vacc_df.columns = vacc_df.columns.str.strip()
                vacc_df['date'] = pd.to_datetime(vacc_df['date'], errors='coerce')
                vacc_df['cumulative_vaccinations'] = pd.to_numeric(vacc_df['cumulative_vaccinations'], errors='coerce').fillna(0)
                vacc_df['daily_vaccinations'] = pd.to_numeric(vacc_df['daily_vaccinations'], errors='coerce').fillna(0)
                vacc_df = vacc_df.dropna(subset=['date'])
                
                # Index par pays
                vaccination_cache = {}
                for country in vacc_df['country'].unique():
                    if pd.notna(country):
                        country_vacc = vacc_df[vacc_df['country'] == country].sort_values('date')
                        vaccination_cache[country] = country_vacc
                
                logger.info(f"💉 Vaccination cache: {len(vaccination_cache)} pays")
                logger.info(f"💉 Période: {vacc_df['date'].min().strftime('%Y-%m-%d')} au {vacc_df['date'].max().strftime('%Y-%m-%d')}")
            else:
                logger.warning("⚠️ Fichier vaccination non trouvé")
                
        except Exception as e:
            logger.error(f"❌ Erreur vaccination cache: {e}")
    
    async def load_country_data(self, country: str):
        """Charge les données COVID depuis MongoDB"""
        try:
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
                {"$match": {"country.country_name": country}},
                {"$sort": {"date": 1}},
                {
                    "$project": {
                        "date": 1,
                        "confirmed": 1,
                        "deaths": 1,
                        "recovered": 1,
                        "active": 1
                    }
                }
            ]
            
            covid_data = list(db.daily_stats.aggregate(pipeline))
            
            if not covid_data:
                raise HTTPException(status_code=404, detail=f"Aucune donnée pour {country}")
            
            covid_df = pd.DataFrame(covid_data)
            covid_df['date'] = pd.to_datetime(covid_df['date'])
            covid_df = covid_df.sort_values('date')
            
            return covid_df
            
        except Exception as e:
            logger.error(f"Erreur chargement {country}: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    def get_vaccination_features(self, country: str, target_date: datetime):
        """Récupère les features de vaccination pour une date"""
        
        # Valeurs par défaut
        vaccination_features = {
            'cumulative_vaccinations': 0,
            'daily_vaccinations': 0,
            'vaccination_rate': 0
        }
        
        # Normalisation de nom
        def normalize_country(name):
            return str(name).strip().lower()
        
        country_norm = normalize_country(country)
        
        # Chercher dans le cache
        vaccination_data = None
        
        if country in vaccination_cache:
            vaccination_data = vaccination_cache[country]
        else:
            # Recherche avec normalisation
            for vacc_country, data in vaccination_cache.items():
                if normalize_country(vacc_country) == country_norm:
                    vaccination_data = data
                    break
        
        if vaccination_data is not None and len(vaccination_data) > 0:
            # Trouver la date la plus proche
            vaccination_data = vaccination_data.copy()
            vaccination_data['date_diff'] = abs((vaccination_data['date'] - target_date).dt.days)
            closest_idx = vaccination_data['date_diff'].idxmin()
            closest_vacc = vaccination_data.loc[closest_idx]
            
            vaccination_features['cumulative_vaccinations'] = float(closest_vacc['cumulative_vaccinations'])
            vaccination_features['daily_vaccinations'] = float(closest_vacc['daily_vaccinations'])
            
            # Calculer un taux basique (vaccinations par 100k habitants estimé)
            vaccination_features['vaccination_rate'] = min(
                vaccination_features['cumulative_vaccinations'] / 10000,  # Estimation basique
                100.0
            )
            
            logger.info(f"💉 Vaccination pour {country} au {target_date.strftime('%Y-%m-%d')}: {vaccination_features['cumulative_vaccinations']:,.0f}")
        
        return vaccination_features
    
    def get_simple_features(self, country: str, target_date: datetime, last_covid_data: dict = None):
        """Crée 12 features simples mais intelligentes pour prédiction"""
        
        # Features temporelles
        features = {
            'day_of_year': target_date.timetuple().tm_yday,
            'month': target_date.month,
            'quarter': (target_date.month - 1) // 3 + 1,
            'week_of_year': target_date.isocalendar().week,
            'month_sin': np.sin(2 * np.pi * target_date.month / 12),
            'month_cos': np.cos(2 * np.pi * target_date.month / 12),
            'mortality_rate': 2.5,  # Défaut
            'recovery_rate': 85.0   # Défaut
        }
        
        # Ajouter vaccination
        vaccination_features = self.get_vaccination_features(country, target_date)
        features.update(vaccination_features)
        
        # Calculer mortalité/récupération si on a des données COVID
        if last_covid_data:
            if last_covid_data['confirmed'] > 0:
                features['mortality_rate'] = (last_covid_data['deaths'] / last_covid_data['confirmed']) * 100
                features['recovery_rate'] = (last_covid_data['recovered'] / last_covid_data['confirmed']) * 100
        
        # Ordre des 12 features (comme l'entraînement)
        feature_order = [
            'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate',
            'month_sin', 'month_cos', 'day_of_year', 'month', 'quarter', 'week_of_year',
            'mortality_rate', 'recovery_rate', 'recovery_rate'  # Doublé pour faire 12
        ]
        
        return np.array([features[f] for f in feature_order], dtype=np.float32)
    
    async def predict_simple(self, country: str, region: str = None, days_to_predict: int = 14, 
                           start_date: str = None, use_vaccination_data: bool = True):
        """Prédiction simple mais intelligente"""
        try:
            if model is None:
                raise HTTPException(status_code=503, detail="Modèle non chargé")
            
            # Charger données COVID
            covid_df = await self.load_country_data(country)
            
            # Date de début prédiction
            if start_date:
                prediction_start = pd.to_datetime(start_date)
            else:
                prediction_start = covid_df['date'].max() + timedelta(days=1)
            
            # Préparer features temporelles
            time_features = covid_df[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
            
            # Normaliser
            if time_scaler is not None:
                time_features_scaled = time_scaler.transform(time_features)
            else:
                time_features_scaled = (time_features - time_features.mean(axis=0)) / (time_features.std(axis=0) + 1e-8)
            
            # Séquence pour LSTM
            if len(time_features_scaled) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(time_features_scaled), time_features_scaled.shape[1]))
                sequence = np.vstack([padding, time_features_scaled])
            else:
                sequence = time_features_scaled[-self.sequence_length:]
            
            # PRÉDICTIONS
            model.eval()
            predictions = []
            vaccination_impacts = []
            current_sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                for day in range(days_to_predict):
                    prediction_date = prediction_start + timedelta(days=day)
                    
                    # Features intelligentes pour cette date
                    last_covid = covid_df.iloc[-1].to_dict() if len(covid_df) > 0 else None
                    enriched_features = self.get_simple_features(country, prediction_date, last_covid)
                    enriched_tensor = torch.FloatTensor(enriched_features).unsqueeze(0).to(self.device)
                    
                    # Normaliser features enrichies
                    if enriched_scaler is not None:
                        enriched_features_norm = enriched_scaler.transform(enriched_features.reshape(1, -1))
                        enriched_tensor = torch.FloatTensor(enriched_features_norm).to(self.device)
                    
                    # Prédiction
                    pred = model(current_sequence, enriched_tensor)
                    predictions.append(pred.cpu().numpy()[0])
                    
                    # Impact vaccination pour cette date
                    vaccination_impacts.append({
                        'date': prediction_date.strftime('%Y-%m-%d'),
                        'vaccination_level': float(enriched_features[0]),  # cumulative_vaccinations
                        'seasonal_factor': float(enriched_features[3])     # month_sin
                    })
                    
                    # Mettre à jour séquence
                    new_point = pred.unsqueeze(1)
                    current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)
            
            # Dénormaliser
            predictions = np.array(predictions)
            if time_scaler is not None:
                predictions = time_scaler.inverse_transform(predictions)
            
            # Dates futures
            future_dates = [prediction_start + timedelta(days=i) for i in range(days_to_predict)]
            
            # Formatter résultats
            formatted_predictions = []
            last_real = covid_df.iloc[-1]
            
            for i, (date, pred) in enumerate(zip(future_dates, predictions)):
                if i == 0:
                    confirmed = max(last_real['confirmed'], int(pred[0]))
                    deaths = max(last_real['deaths'], int(pred[1]))
                    recovered = max(last_real['recovered'], int(pred[2]))
                    active = max(0, int(pred[3]))
                else:
                    prev_pred = formatted_predictions[i-1]
                    confirmed = max(prev_pred['confirmed'], int(pred[0]))
                    deaths = max(prev_pred['deaths'], int(pred[1]))
                    recovered = max(prev_pred['recovered'], int(pred[2]))
                    active = max(0, int(pred[3]))
                
                formatted_predictions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day_ahead": i + 1,
                    "confirmed": confirmed,
                    "deaths": deaths,
                    "recovered": recovered,
                    "active": active,
                    "mortality_rate": round((deaths / max(confirmed, 1)) * 100, 2)
                })
            
            # Intervalles de confiance
            confidence_intervals = []
            for i, pred in enumerate(formatted_predictions):
                uncertainty = 0.05 + (i * 0.005)  # 5% + progression
                
                confidence_intervals.append({
                    "date": pred["date"],
                    "confirmed_lower": max(0, int(pred["confirmed"] * (1 - uncertainty))),
                    "confirmed_upper": int(pred["confirmed"] * (1 + uncertainty)),
                    "deaths_lower": max(0, int(pred["deaths"] * (1 - uncertainty))),
                    "deaths_upper": int(pred["deaths"] * (1 + uncertainty))
                })
            
            # Impact vaccination moyen
            avg_vaccination_impact = {
                'average_vaccination_level': np.mean([v['vaccination_level'] for v in vaccination_impacts]),
                'vaccination_data_available': use_vaccination_data and len(vaccination_cache) > 0,
                'seasonal_variation': np.std([v['seasonal_factor'] for v in vaccination_impacts])
            }
            
            return {
                "predictions": formatted_predictions,
                "confidence_intervals": confidence_intervals,
                "vaccination_impact": avg_vaccination_impact,
                "model_info": {
                    "model_type": "Simple Intelligent LSTM",
                    "sequence_length": self.sequence_length,
                    "prediction_start_date": prediction_start.strftime("%Y-%m-%d"),
                    "last_real_date": covid_df['date'].max().strftime("%Y-%m-%d"),
                    "data_points_used": len(covid_df),
                    "features_count": 12,
                    "vaccination_integrated": use_vaccination_data,
                    "device": str(self.device)
                }
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Instance du prédicteur
predictor = SimpleIntelligentPredictor()

@app.on_event("startup")
async def startup_event():
    """Initialisation"""
    global model, time_scaler, enriched_scaler
    
    logger.info("🚀 Démarrage API Simple + Intelligente...")
    
    # MongoDB
    await predictor.connect_mongodb()
    
    # Cache vaccination
    predictor.load_vaccination_cache()
    
    # Modèle
    try:
        model_path = os.path.join('models', 'simple_covid_model.pth')
        time_scaler_path = os.path.join('models', 'simple_time_scaler.pkl')
        enriched_scaler_path = os.path.join('models', 'simple_enriched_scaler.pkl')
        
        if os.path.exists(model_path):
            config_path = os.path.join('models', 'simple_config.json')
            enriched_features = 12  # Par défaut
            
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    enriched_features = len(config.get('enriched_features', []))
            
            model = SimpleIntelligentCovidLSTM(enriched_features=enriched_features).to(predictor.device)
            model.load_state_dict(torch.load(model_path, map_location=predictor.device))
            model.eval()
            logger.info("✅ Modèle SIMPLE chargé")
            
            if os.path.exists(time_scaler_path):
                time_scaler = joblib.load(time_scaler_path)
                logger.info("✅ Time scaler chargé")
            
            if os.path.exists(enriched_scaler_path):
                enriched_scaler = joblib.load(enriched_scaler_path)
                logger.info("✅ Enriched scaler chargé")
        else:
            logger.warning("⚠️ Modèle non trouvé")
            
    except Exception as e:
        logger.error(f"❌ Erreur modèle: {e}")
    
    logger.info("✅ API Simple + Intelligente prête!")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check"""
    mongodb_connected = mongodb_client is not None
    model_loaded = model is not None
    vaccination_available = len(vaccination_cache) > 0
    
    countries_count = 0
    if mongodb_connected:
        try:
            countries_count = db.countries.count_documents({})
        except:
            pass
    
    return HealthCheck(
        status="ready" if all([mongodb_connected, model_loaded]) else "partial",
        model_loaded=model_loaded,
        mongodb_connected=mongodb_connected,
        vaccination_data_available=vaccination_available,
        countries_count=countries_count
    )

@app.post("/predict", response_model=SimplePredictionResponse)
async def predict_covid_simple(request: SimplePredictionRequest):
    """Prédiction COVID simple mais intelligente"""
    logger.info(f"🧠 Prédiction pour {request.country}, {request.days_to_predict} jours")
    
    try:
        result = await predictor.predict_simple(
            country=request.country,
            region=request.region,
            days_to_predict=request.days_to_predict,
            start_date=request.start_date,
            use_vaccination_data=request.use_vaccination_data
        )
        
        return SimplePredictionResponse(
            country=request.country,
            region=request.region,
            prediction_start_date=result["model_info"]["prediction_start_date"],
            predictions=result["predictions"],
            confidence_intervals=result["confidence_intervals"],
            vaccination_impact=result["vaccination_impact"],
            model_info=result["model_info"],
            prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail="Erreur serveur")

@app.get("/countries")
async def get_available_countries():
    """Liste des pays"""
    try:
        countries = db.countries.find({}, {"country_name": 1, "_id": 0})
        countries_list = [country["country_name"] for country in countries]
        return {"countries": countries_list, "count": len(countries_list)}
    except Exception as e:
        logger.error(f"Erreur pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération pays")

@app.get("/vaccination/{country}")
async def get_vaccination_info(country: str, date: str = None):
    """Info vaccination pour un pays"""
    if date:
        target_date = pd.to_datetime(date)
    else:
        target_date = datetime.now()
    
    vaccination_features = predictor.get_vaccination_features(country, target_date)
    
    return {
        "country": country,
        "date": target_date.strftime('%Y-%m-%d'),
        "vaccination_data": vaccination_features,
        "data_available": vaccination_features['cumulative_vaccinations'] > 0
    }

@app.get("/model/info")
async def get_model_info():
    """Info modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": "Simple Intelligent LSTM",
        "features": {
            "vaccination": "Données vaccination 2020-2025",
            "temporal": "Patterns saisonniers et cycliques",
            "covid_derived": "Taux mortalité/récupération"
        },
        "architecture": {
            "lstm_layers": 2,
            "hidden_size": 128,
            "enriched_features": 12,
            "parameters": sum(p.numel() for p in model.parameters())
        },
        "data_sources": {
            "mongodb": "COVID base 2020",
            "vaccination_csv": "Vaccination 2020-2025"
        },
        "cache_status": {
            "vaccination_countries": len(vaccination_cache),
            "mongodb_connected": mongodb_client is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
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
import glob
from pymongo import MongoClient
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="COVID-19 AI Prediction API - Hybrid", version="2.0.0")

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

# Modèle LSTM Hybride
class CovidLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, enriched_features=10, dropout=0.2):
        super(CovidLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.enriched_fc = nn.Sequential(
            nn.Linear(enriched_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
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
class PredictionRequest(BaseModel):
    country: str
    region: Optional[str] = None
    days_to_predict: int = 14
    include_demographics: bool = True

class PredictionResponse(BaseModel):
    country: str
    region: Optional[str]
    predictions: List[Dict[str, Union[str, int]]]
    confidence_intervals: List[Dict[str, Union[str, int]]]
    model_metrics: Dict[str, Union[str, int, float]]
    prediction_date: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    mongodb_connected: bool
    csv_data_available: bool

# Variables globales
model = None
time_scaler = None
enriched_scaler = None
mongodb_client = None
db = None
enrichment_cache = {}

class HybridCovidPredictor:
    def __init__(self):
        self.sequence_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")
    
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
    
    def load_enrichment_cache(self):
        """Charge les données d'enrichissement en cache"""
        global enrichment_cache
        
        try:
            # Charger les données de vaccination
            vaccination_file = os.path.join(CSV_DATA_PATH, 'cumulative-covid-vaccinations_clean.csv')
            if os.path.exists(vaccination_file):
                vacc_df = pd.read_csv(vaccination_file)
                vacc_df['date'] = pd.to_datetime(vacc_df['date'])
                
                # Créer un cache par pays
                enrichment_cache['vaccination'] = {}
                for country in vacc_df['country'].unique():
                    country_vacc = vacc_df[vacc_df['country'] == country].sort_values('date')
                    enrichment_cache['vaccination'][country] = country_vacc
                
                logger.info(f"💉 Vaccination cache: {len(enrichment_cache['vaccination'])} pays")
            
            # Charger des données démographiques simplifiées
            demo_files = glob.glob(os.path.join(CSV_DATA_PATH, "*age*clean.csv"))
            demo_files.extend(glob.glob(os.path.join(CSV_DATA_PATH, "*pooled*clean.csv")))
            
            if demo_files:
                demo_dfs = []
                for file in demo_files[:2]:  # Limiter pour éviter la surcharge
                    try:
                        df = pd.read_csv(file)
                        if 'country' in df.columns:
                            demo_dfs.append(df)
                    except:
                        continue
                
                if demo_dfs:
                    demo_df = pd.concat(demo_dfs, ignore_index=True)
                    
                    # Créer des statistiques par pays
                    demo_stats = demo_df.groupby('country').agg({
                        'cum_death_both': ['mean', 'std'] if 'cum_death_both' in demo_df.columns else lambda x: [0, 0],
                        'age_numeric': 'mean' if 'age_numeric' in demo_df.columns else lambda x: 50
                    }).reset_index()
                    
                    demo_stats.columns = ['country', 'avg_demo_deaths', 'std_demo_deaths', 'avg_age']
                    demo_stats = demo_stats.fillna(0)
                    
                    enrichment_cache['demographics'] = demo_stats.set_index('country').to_dict('index')
                    logger.info(f"👥 Démographie cache: {len(enrichment_cache['demographics'])} pays")
            
            logger.info("✅ Cache d'enrichissement chargé")
            
        except Exception as e:
            logger.error(f"⚠️  Erreur chargement cache enrichissement: {e}")
    
    async def load_country_data_from_mongodb(self, country: str):
        """Charge les données COVID d'un pays depuis MongoDB"""
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
                raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {country}")
            
            covid_df = pd.DataFrame(covid_data)
            covid_df['date'] = pd.to_datetime(covid_df['date'])
            covid_df = covid_df.sort_values('date')
            
            return covid_df
            
        except Exception as e:
            logger.error(f"Erreur chargement données MongoDB: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur chargement données: {str(e)}")
    
    def get_enriched_features(self, country: str, covid_df: pd.DataFrame):
        """Récupère les features enrichies pour un pays"""
        
        # Initialiser avec des valeurs par défaut
        enriched_features = {
            'cumulative_vaccinations': 0,
            'daily_vaccinations': 0,
            'avg_demo_deaths': 0,
            'std_demo_deaths': 0,
            'avg_age': 50,
            'day_of_year': covid_df['date'].iloc[-1].dayofyear,
            'month': covid_df['date'].iloc[-1].month,
            'quarter': (covid_df['date'].iloc[-1].month - 1) // 3 + 1,
            'confirmed_avg': covid_df['confirmed'].mean(),
            'deaths_avg': covid_df['deaths'].mean()
        }
        
        # Enrichir avec les données de vaccination si disponibles
        if 'vaccination' in enrichment_cache and country in enrichment_cache['vaccination']:
            vacc_data = enrichment_cache['vaccination'][country]
            if len(vacc_data) > 0:
                latest_vacc = vacc_data.iloc[-1]
                enriched_features['cumulative_vaccinations'] = latest_vacc['cumulative_vaccinations']
                enriched_features['daily_vaccinations'] = latest_vacc['daily_vaccinations']
        
        # Enrichir avec les données démographiques si disponibles
        if 'demographics' in enrichment_cache and country in enrichment_cache['demographics']:
            demo_data = enrichment_cache['demographics'][country]
            enriched_features.update(demo_data)
        
        return np.array(list(enriched_features.values()), dtype=np.float32)
    
    async def predict(self, country: str, region: str = None, days_to_predict: int = 14):
        """Effectue la prédiction hybride"""
        try:
            # Vérifier que le modèle est chargé
            if model is None:
                raise HTTPException(status_code=503, detail="Modèle non chargé")
            
            # Charger les données COVID du pays depuis MongoDB
            covid_df = await self.load_country_data_from_mongodb(country)
            
            # Préparer les features temporelles (COVID)
            time_features = covid_df[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
            
            # Normaliser les features temporelles
            if time_scaler is not None:
                time_features_scaled = time_scaler.transform(time_features)
            else:
                time_features_scaled = (time_features - time_features.mean(axis=0)) / (time_features.std(axis=0) + 1e-8)
            
            # Créer la séquence pour LSTM
            if len(time_features_scaled) < self.sequence_length:
                # Padding si pas assez de données
                padding = np.zeros((self.sequence_length - len(time_features_scaled), time_features_scaled.shape[1]))
                sequence = np.vstack([padding, time_features_scaled])
            else:
                sequence = time_features_scaled[-self.sequence_length:]
            
            # Récupérer les features enrichies
            enriched_features = self.get_enriched_features(country, covid_df)
            
            # Normaliser les features enrichies
            if enriched_scaler is not None:
                enriched_features = enriched_scaler.transform(enriched_features.reshape(1, -1))[0]
            
            # Convertir en tensors
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            enriched_tensor = torch.FloatTensor(enriched_features).unsqueeze(0).to(self.device)
            
            # Prédiction
            model.eval()
            predictions = []
            current_sequence = sequence_tensor.clone()
            
            with torch.no_grad():
                for day in range(days_to_predict):
                    # Prédiction pour le jour suivant
                    pred = model(current_sequence, enriched_tensor)
                    predictions.append(pred.cpu().numpy()[0])
                    
                    # Mettre à jour la séquence avec la prédiction
                    new_point = pred.unsqueeze(1)
                    current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)
            
            # Dénormaliser les prédictions
            predictions = np.array(predictions)
            if time_scaler is not None:
                predictions = time_scaler.inverse_transform(predictions)
            
            # Créer les dates futures à partir de la DERNIÈRE date réelle
            last_date = covid_df['date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            
            # Formatter les résultats avec cohérence
            formatted_predictions = []
            last_real_values = covid_df.iloc[-1]
            
            for i, (date, pred) in enumerate(zip(future_dates, predictions)):
                # Assurer la cohérence : pas de baisse drastique irréaliste
                if i == 0:
                    # Premier jour : petite variation par rapport au dernier jour réel
                    confirmed = max(last_real_values['confirmed'], int(pred[0]))
                    deaths = max(last_real_values['deaths'], int(pred[1]))
                    recovered = max(last_real_values['recovered'], int(pred[2]))
                    active = max(0, int(pred[3]))
                else:
                    # Jours suivants : évolution progressive
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
                    "active": active
                })
            
            # Intervalles de confiance réalistes
            confidence_intervals = []
            for i, pred in enumerate(formatted_predictions):
                uncertainty = 0.03 + (i * 0.005)  # Incertitude plus réaliste (3% à 10%)
                confidence_intervals.append({
                    "date": pred["date"],
                    "confirmed_lower": max(0, int(pred["confirmed"] * (1 - uncertainty))),
                    "confirmed_upper": int(pred["confirmed"] * (1 + uncertainty)),
                    "deaths_lower": max(0, int(pred["deaths"] * (1 - uncertainty))),
                    "deaths_upper": int(pred["deaths"] * (1 + uncertainty)),
                    "recovered_lower": max(0, int(pred["recovered"] * (1 - uncertainty))),
                    "recovered_upper": int(pred["recovered"] * (1 + uncertainty)),
                    "active_lower": max(0, int(pred["active"] * (1 - uncertainty))),
                    "active_upper": int(pred["active"] * (1 + uncertainty))
                })
            
            return {
                "predictions": formatted_predictions,
                "confidence_intervals": confidence_intervals,
                "model_metrics": {
                    "sequence_length": self.sequence_length,
                    "enriched_features_count": len(enriched_features),
                    "device": str(self.device),
                    "data_points_used": len(covid_df),
                    "last_real_date": last_date.strftime("%Y-%m-%d"),
                    "model_type": "Hybrid MongoDB + CSV",
                    "vaccination_data_available": 'vaccination' in enrichment_cache and country in enrichment_cache.get('vaccination', {}),
                    "demographic_data_available": 'demographics' in enrichment_cache and country in enrichment_cache.get('demographics', {})
                }
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# Instance du prédicteur
predictor = HybridCovidPredictor()

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global model, time_scaler, enriched_scaler
    
    logger.info("🚀 Démarrage de l'API IA COVID Hybride...")
    
    # Connexion MongoDB
    await predictor.connect_mongodb()
    
    # Charger le cache d'enrichissement
    predictor.load_enrichment_cache()
    
    # Charger le modèle hybride
    try:
        model_path = os.path.join('models', 'covid_lstm_model.pth')
        time_scaler_path = os.path.join('models', 'time_scaler.pkl')
        enriched_scaler_path = os.path.join('models', 'enriched_scaler.pkl')
        
        if os.path.exists(model_path):
            # Déterminer les tailles automatiquement
            config_path = os.path.join('models', 'config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    enriched_features_count = len(config.get('enriched_features', []))
            else:
                enriched_features_count = 10  # Valeur par défaut
            
            model = CovidLSTM(enriched_features=enriched_features_count).to(predictor.device)
            model.load_state_dict(torch.load(model_path, map_location=predictor.device))
            model.eval()
            logger.info("✅ Modèle hybride chargé")
            
            if os.path.exists(time_scaler_path):
                time_scaler = joblib.load(time_scaler_path)
                logger.info("✅ Time scaler chargé")
            
            if os.path.exists(enriched_scaler_path):
                enriched_scaler = joblib.load(enriched_scaler_path)
                logger.info("✅ Enriched scaler chargé")
        else:
            logger.warning("⚠️  Modèle hybride non trouvé")
            
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
    
    logger.info("✅ API IA hybride prête !")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Vérification de l'état de l'API"""
    mongodb_connected = mongodb_client is not None
    model_loaded = model is not None
    csv_data_available = len(enrichment_cache) > 0
    
    return HealthCheck(
        status="healthy" if mongodb_connected and model_loaded else "partial",
        model_loaded=model_loaded,
        mongodb_connected=mongodb_connected,
        csv_data_available=csv_data_available
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_covid_evolution(request: PredictionRequest):
    """Prédiction hybride de l'évolution COVID"""
    logger.info(f"Prédiction hybride pour {request.country}, {request.days_to_predict} jours")
    
    try:
        result = await predictor.predict(
            country=request.country,
            region=request.region,
            days_to_predict=request.days_to_predict
        )
        
        return PredictionResponse(
            country=request.country,
            region=request.region,
            predictions=result["predictions"],
            confidence_intervals=result["confidence_intervals"],
            model_metrics=result["model_metrics"],
            prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/countries")
async def get_available_countries():
    """Liste des pays disponibles"""
    try:
        countries = db.countries.find({}, {"country_name": 1, "_id": 0})
        return {"countries": [country["country_name"] for country in countries]}
    except Exception as e:
        logger.error(f"Erreur récupération pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération des pays")

@app.get("/model/info")
async def get_model_info():
    """Informations sur le modèle hybride"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    config_info = {}
    metrics_info = {}
    
    config_path = os.path.join('models', 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_info = json.load(f)
    
    metrics_path = os.path.join('models', 'metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics_info = json.load(f)
    
    return {
        "model_type": "LSTM Hybride COVID (MongoDB + CSV)",
        "architecture": "Fusion MongoDB (COVID) + CSV (enrichissement)",
        "data_sources": {
            "mongodb": "Données COVID principales (confirmed, deaths, recovered, active)",
            "csv": "Données enrichissement (vaccination, démographie)"
        },
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "enrichment_cache": {
            "vaccination_countries": len(enrichment_cache.get('vaccination', {})),
            "demographic_countries": len(enrichment_cache.get('demographics', {}))
        },
        "scalers_loaded": {
            "time_scaler": time_scaler is not None,
            "enriched_scaler": enriched_scaler is not None
        },
        "model_config": config_info,
        "model_metrics": metrics_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
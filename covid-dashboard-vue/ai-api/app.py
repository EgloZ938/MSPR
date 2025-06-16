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

app = FastAPI(title="COVID-19 AI Prediction API", version="1.0.0")

# CORS pour permettre les requêtes depuis le frontend Vue.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration MongoDB
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'covid_dashboard')

# Modèle LSTM Hybride avec PyTorch
class CovidLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, demographic_features=6, dropout=0.2):
        super(CovidLSTM, self).__init__()
        
        # LSTM pour les séries temporelles
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Couches pour les features démographiques
        self.demographic_fc = nn.Sequential(
            nn.Linear(demographic_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Couche de fusion
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
    
    def forward(self, time_series, demographic_features):
        batch_size = time_series.size(0)
        
        # LSTM pour séries temporelles
        lstm_out, _ = self.lstm(time_series)
        lstm_features = lstm_out[:, -1, :]  # Dernière sortie de la séquence
        
        # Traitement des features démographiques
        demo_features = self.demographic_fc(demographic_features)
        
        # Fusion des features
        combined_features = torch.cat([lstm_features, demo_features], dim=1)
        output = self.fusion_fc(combined_features)
        
        return output

# Modèles Pydantic pour l'API
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

# Variables globales pour le modèle et les données
model = None
scaler = None
demographic_scaler = None
mongodb_client = None
db = None

class CovidPredictor:
    def __init__(self):
        self.sequence_length = 30  # 30 jours d'historique
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")
    
    async def connect_mongodb(self):
        """Connexion à MongoDB"""
        global mongodb_client, db
        try:
            mongodb_client = MongoClient(MONGO_URI)
            db = mongodb_client[DB_NAME]
            # Test de connexion
            db.command('ping')
            logger.info("✅ Connexion MongoDB réussie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion MongoDB: {e}")
            return False
    
    async def load_data_from_mongodb(self, country: str, region: str = None):
        """Charge les données depuis MongoDB"""
        try:
            # Données de séries temporelles
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
            
            time_series_data = list(db.daily_stats.aggregate(pipeline))
            
            # Données démographiques
            demo_pipeline = [
                {"$match": {"country_name": country}},
                {
                    "$group": {
                        "_id": "$age_group",
                        "avg_mortality_rate_male": {"$avg": "$mortality_rate_male"},
                        "avg_mortality_rate_female": {"$avg": "$mortality_rate_female"},
                        "total_deaths": {"$sum": "$cum_death_both"},
                        "avg_age_numeric": {"$avg": "$age_numeric"}
                    }
                }
            ]
            
            demographic_data = list(db.age_demographics.aggregate(demo_pipeline))
            
            return time_series_data, demographic_data
        
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            return [], []
    
    def prepare_features(self, time_series_data, demographic_data):
        """Prépare les features pour le modèle"""
        # Séries temporelles
        df = pd.DataFrame(time_series_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Features temporelles
        time_features = df[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
        
        # Normalisation des séries temporelles
        if scaler is None:
            logger.warning("Scaler non chargé, utilisation de valeurs par défaut")
            time_features_scaled = (time_features - time_features.mean(axis=0)) / (time_features.std(axis=0) + 1e-8)
        else:
            time_features_scaled = scaler.transform(time_features)
        
        # Features démographiques
        if demographic_data:
            demo_df = pd.DataFrame(demographic_data)
            demographic_features = np.array([
                demo_df['avg_mortality_rate_male'].mean() if len(demo_df) > 0 else 0,
                demo_df['avg_mortality_rate_female'].mean() if len(demo_df) > 0 else 0,
                demo_df['total_deaths'].sum() if len(demo_df) > 0 else 0,
                demo_df['avg_age_numeric'].mean() if len(demo_df) > 0 else 50,
                len(demo_df),  # Nombre de groupes d'âge
                demo_df['total_deaths'].std() if len(demo_df) > 1 else 0  # Variance des décès par âge
            ], dtype=np.float32)
        else:
            # Valeurs par défaut si pas de données démographiques
            demographic_features = np.array([0.5, 0.5, 1000, 50, 10, 100], dtype=np.float32)
        
        return time_features_scaled, demographic_features, df
    
    def create_sequences(self, data, sequence_length):
        """Crée les séquences pour LSTM"""
        if len(data) < sequence_length:
            # Padding si pas assez de données
            padding = np.zeros((sequence_length - len(data), data.shape[1]))
            data = np.vstack([padding, data])
        
        return data[-sequence_length:]
    
    async def predict(self, country: str, region: str = None, days_to_predict: int = 14):
        """Effectue la prédiction"""
        try:
            # Charger les données
            time_series_data, demographic_data = await self.load_data_from_mongodb(country, region)
            
            if not time_series_data:
                raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {country}")
            
            # Préparer les features
            time_features, demo_features, df = self.prepare_features(time_series_data, demographic_data)
            
            # Créer la séquence
            sequence = self.create_sequences(time_features, self.sequence_length)
            
            # Convertir en tensors
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            demo_tensor = torch.FloatTensor(demo_features).unsqueeze(0).to(self.device)
            
            # Prédiction
            model.eval()
            predictions = []
            current_sequence = sequence_tensor.clone()
            
            with torch.no_grad():
                for day in range(days_to_predict):
                    # Prédiction pour le jour suivant
                    pred = model(current_sequence, demo_tensor)
                    predictions.append(pred.cpu().numpy()[0])
                    
                    # Mettre à jour la séquence avec la prédiction
                    new_point = pred.unsqueeze(1)
                    current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)
            
            # Dénormaliser les prédictions
            predictions = np.array(predictions)
            if scaler is not None:
                predictions = scaler.inverse_transform(predictions)
            
            # Créer les dates futures
            last_date = df['date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            
            # Formatter les résultats
            formatted_predictions = []
            for i, (date, pred) in enumerate(zip(future_dates, predictions)):
                formatted_predictions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day_ahead": i + 1,
                    "confirmed": max(0, int(pred[0])),
                    "deaths": max(0, int(pred[1])),
                    "recovered": max(0, int(pred[2])),
                    "active": max(0, int(pred[3]))
                })
            
            # Intervalles de confiance (simulation simple)
            confidence_intervals = []
            for i, pred in enumerate(formatted_predictions):
                uncertainty = 0.1 + (i * 0.02)  # Incertitude croissante
                confidence_intervals.append({
                    "date": pred["date"],
                    "confirmed_lower": max(0, int(pred["confirmed"] * (1 - uncertainty))),
                    "confirmed_upper": int(pred["confirmed"] * (1 + uncertainty)),
                    "deaths_lower": max(0, int(pred["deaths"] * (1 - uncertainty))),
                    "deaths_upper": int(pred["deaths"] * (1 + uncertainty))
                })
            
            return {
                "predictions": formatted_predictions,
                "confidence_intervals": confidence_intervals,
                "model_metrics": {
                    "sequence_length": self.sequence_length,
                    "demographic_features_count": len(demo_features),
                    "device": str(self.device),
                    "data_points_used": len(time_series_data)
                }
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Instance du prédicteur
predictor = CovidPredictor()

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global model, scaler, demographic_scaler
    
    logger.info("🚀 Démarrage de l'API IA COVID...")
    
    # Connexion MongoDB
    await predictor.connect_mongodb()
    
    # Charger le modèle (pour l'instant on crée un modèle par défaut)
    try:
        model = CovidLSTM().to(predictor.device)
        # TODO: Charger un modèle pré-entraîné si disponible
        logger.info("✅ Modèle LSTM initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
    
    logger.info("✅ API IA prête !")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Vérification de l'état de l'API"""
    mongodb_connected = mongodb_client is not None
    model_loaded = model is not None
    
    return HealthCheck(
        status="healthy" if mongodb_connected and model_loaded else "partial",
        model_loaded=model_loaded,
        mongodb_connected=mongodb_connected
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_covid_evolution(request: PredictionRequest):
    """Prédiction de l'évolution COVID pour un pays"""
    logger.info(f"Prédiction demandée pour {request.country}, {request.days_to_predict} jours")
    
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
    """Liste des pays disponibles pour prédiction"""
    try:
        countries = db.countries.find({}, {"country_name": 1, "_id": 0})
        return {"countries": [country["country_name"] for country in countries]}
    except Exception as e:
        logger.error(f"Erreur récupération pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération des pays")

@app.get("/model/info")
async def get_model_info():
    """Informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": "LSTM Hybride avec PyTorch",
        "input_features": ["confirmed", "deaths", "recovered", "active"],
        "demographic_features": ["mortality_rates", "age_distribution", "population_stats"],
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
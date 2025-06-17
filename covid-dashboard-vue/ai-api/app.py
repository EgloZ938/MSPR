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
import re
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

# Configuration
DATA_PATH = '../data/dataset_clean'

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
    data_available: bool

# Variables globales pour le modèle et les données
model = None
time_scaler = None
demo_scaler = None
csv_data = None

class CovidCSVPredictor:
    def __init__(self):
        self.sequence_length = 30  # 30 jours d'historique
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")
    
    def detect_separator(self, filepath):
        """Détecte le séparateur CSV"""
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        separators = [',', ';', '\t', '|']
        max_cols = 0
        best_sep = ','
        
        for sep in separators:
            cols = len(first_line.split(sep))
            if cols > max_cols:
                max_cols = cols
                best_sep = sep
        
        return best_sep
    
    def extract_date_from_filename(self, filename):
        """Extrait la date du nom de fichier"""
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',           # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})',           # YYYY_MM_DD
            r'(\d{2}-\d{2}-\d{4})',           # MM-DD-YYYY
            r'(\d{2}_\d{2}_\d{4})',           # MM_DD_YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    if date_str.count('-') == 2 or date_str.count('_') == 2:
                        parts = re.split(r'[-_]', date_str)
                        if len(parts[0]) == 4:  # YYYY-MM-DD
                            return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                        else:  # MM-DD-YYYY
                            return datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                except:
                    continue
        
        return None
    
    def load_csv_data(self):
        """Charge toutes les données CSV"""
        logger.info(f"📂 Chargement des données CSV depuis {DATA_PATH}")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Le dossier {DATA_PATH} n'existe pas!")
        
        csv_files = glob.glob(os.path.join(DATA_PATH, "*_clean.csv"))
        logger.info(f"📁 {len(csv_files)} fichiers CSV trouvés")
        
        all_data = []
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            
            # Extraire la date du nom de fichier
            file_date = self.extract_date_from_filename(filename)
            
            # Détecter le séparateur
            separator = self.detect_separator(csv_file)
            
            try:
                df = pd.read_csv(csv_file, sep=separator)
                
                if len(df) == 0:
                    continue
                
                # Traiter selon le type de fichier
                if 'full_grouped' in filename:
                    df = self.process_full_grouped_data(df, filename, file_date)
                elif 'country_wise_latest' in filename:
                    df = self.process_country_wise_data(df, filename, file_date)
                elif 'covid_19_clean_complete' in filename:
                    df = self.process_complete_data(df, filename, file_date)
                else:
                    continue
                
                if len(df) > 0:
                    all_data.append(df)
                
            except Exception as e:
                logger.warning(f"Erreur lecture {filename}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Aucune donnée n'a pu être chargée!")
        
        # Combiner toutes les données
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(['country', 'date'])
        
        logger.info(f"📊 {len(combined_df)} enregistrements chargés")
        logger.info(f"🏳️ Pays disponibles: {sorted(combined_df['country'].unique())}")
        
        return combined_df
    
    def process_full_grouped_data(self, df, filename, file_date):
        """Traite les données full_grouped"""
        required_cols = ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']
        
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        df = df[required_cols].copy()
        df.columns = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active']
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        for col in ['confirmed', 'deaths', 'recovered', 'active']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def process_country_wise_data(self, df, filename, file_date):
        """Traite les données country_wise_latest"""
        if 'Country/Region' not in df.columns:
            return pd.DataFrame()
        
        df['date'] = file_date if file_date else datetime.now()
        df['country'] = df['Country/Region']
        
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        for col in numeric_cols:
            if col in df.columns:
                df[col.lower()] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col.lower()] = 0
        
        return df[['country', 'date', 'confirmed', 'deaths', 'recovered', 'active']].copy()
    
    def process_complete_data(self, df, filename, file_date):
        """Traite les données covid_19_clean_complete"""
        required_cols = ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered']
        
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        grouped = df.groupby(['Country/Region', 'Date']).agg({
            'Confirmed': 'sum',
            'Deaths': 'sum', 
            'Recovered': 'sum'
        }).reset_index()
        
        grouped['Active'] = grouped['Confirmed'] - grouped['Deaths'] - grouped['Recovered']
        grouped.columns = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active']
        
        for col in ['confirmed', 'deaths', 'recovered', 'active']:
            grouped[col] = pd.to_numeric(grouped[col], errors='coerce').fillna(0)
        
        return grouped
    
    def get_country_data(self, country: str):
        """Récupère les données pour un pays spécifique"""
        if csv_data is None:
            raise HTTPException(status_code=503, detail="Données CSV non chargées")
        
        country_data = csv_data[csv_data['country'] == country].copy()
        
        if len(country_data) == 0:
            raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {country}")
        
        return country_data.sort_values('date')
    
    def prepare_features(self, country_data):
        """Prépare les features pour le modèle"""
        # Features temporelles
        time_features = country_data[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
        
        # Normalisation des séries temporelles
        if time_scaler is not None:
            time_features_scaled = time_scaler.transform(time_features)
        else:
            time_features_scaled = (time_features - time_features.mean(axis=0)) / (time_features.std(axis=0) + 1e-8)
        
        # Features démographiques (moyennes par pays)
        demographic_features = np.array([
            country_data['confirmed'].mean(),
            country_data['deaths'].mean(),
            country_data['recovered'].mean(), 
            country_data['active'].mean(),
            len(country_data),  # Nombre de points de données
            country_data['deaths'].std() if len(country_data) > 1 else 0  # Variabilité
        ], dtype=np.float32)
        
        # Normaliser les features démographiques si possible
        if demo_scaler is not None:
            demographic_features = demo_scaler.transform(demographic_features.reshape(1, -1))[0]
        
        return time_features_scaled, demographic_features
    
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
            # Vérifier que le modèle est chargé
            if model is None:
                raise HTTPException(status_code=503, detail="Modèle non chargé")
            
            # Charger les données du pays
            country_data = self.get_country_data(country)
            
            # Préparer les features
            time_features, demo_features = self.prepare_features(country_data)
            
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
            if time_scaler is not None:
                predictions = time_scaler.inverse_transform(predictions)
            
            # Créer les dates futures
            last_date = country_data['date'].iloc[-1]
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
            
            # Intervalles de confiance
            confidence_intervals = []
            for i, pred in enumerate(formatted_predictions):
                uncertainty = 0.05 + (i * 0.01)  # Incertitude croissante
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
                    "demographic_features_count": len(demo_features),
                    "device": str(self.device),
                    "data_points_used": len(country_data),
                    "last_date": last_date.strftime("%Y-%m-%d"),
                    "model_loaded": True,
                    "scalers_loaded": time_scaler is not None and demo_scaler is not None
                }
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# Instance du prédicteur
predictor = CovidCSVPredictor()

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global model, time_scaler, demo_scaler, csv_data
    
    logger.info("🚀 Démarrage de l'API IA COVID (mode CSV)...")
    
    # Charger les données CSV
    try:
        csv_data = predictor.load_csv_data()
        logger.info("✅ Données CSV chargées")
    except Exception as e:
        logger.error(f"❌ Erreur chargement CSV: {e}")
        csv_data = None
    
    # Charger le modèle pré-entraîné
    try:
        model_path = os.path.join('models', 'covid_lstm_model.pth')
        time_scaler_path = os.path.join('models', 'time_scaler.pkl')
        demo_scaler_path = os.path.join('models', 'demo_scaler.pkl')
        
        if os.path.exists(model_path):
            model = CovidLSTM().to(predictor.device)
            model.load_state_dict(torch.load(model_path, map_location=predictor.device))
            model.eval()
            logger.info("✅ Modèle LSTM chargé")
            
            if os.path.exists(time_scaler_path):
                time_scaler = joblib.load(time_scaler_path)
                logger.info("✅ Time scaler chargé")
            
            if os.path.exists(demo_scaler_path):
                demo_scaler = joblib.load(demo_scaler_path)
                logger.info("✅ Demo scaler chargé")
        else:
            logger.warning("⚠️  Modèle non trouvé")
            
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
    
    logger.info("✅ API IA prête !")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Vérification de l'état de l'API"""
    model_loaded = model is not None
    data_available = csv_data is not None
    
    return HealthCheck(
        status="healthy" if model_loaded and data_available else "partial",
        model_loaded=model_loaded,
        data_available=data_available
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
        if csv_data is None:
            raise HTTPException(status_code=503, detail="Données CSV non chargées")
        
        countries = sorted(csv_data['country'].unique().tolist())
        return {"countries": countries}
    except Exception as e:
        logger.error(f"Erreur récupération pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération des pays")

@app.get("/model/info")
async def get_model_info():
    """Informations sur le modèle"""
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
        "model_type": "LSTM Hybride COVID (CSV Mode)",
        "input_features": ["confirmed", "deaths", "recovered", "active"],
        "demographic_features": 6,
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data_source": "CSV files direct",
        "countries_available": len(csv_data['country'].unique()) if csv_data is not None else 0,
        "scalers_loaded": {
            "time_scaler": time_scaler is not None,
            "demo_scaler": demo_scaler is not None
        },
        "model_config": config_info,
        "model_metrics": metrics_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
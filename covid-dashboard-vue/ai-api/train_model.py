import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'covid_dashboard')
MODEL_DIR = 'models'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Créer le dossier de modèles
os.makedirs(MODEL_DIR, exist_ok=True)

class CovidLSTM(nn.Module):
    """Modèle LSTM hybride identique à l'API"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, demographic_features=6, dropout=0.2):
        super(CovidLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.demographic_fc = nn.Sequential(
            nn.Linear(demographic_features, 64),
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
            nn.Linear(128, 4)
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, time_series, demographic_features):
        batch_size = time_series.size(0)
        
        lstm_out, _ = self.lstm(time_series)
        lstm_features = lstm_out[:, -1, :]
        
        demo_features = self.demographic_fc(demographic_features)
        
        combined_features = torch.cat([lstm_features, demo_features], dim=1)
        output = self.fusion_fc(combined_features)
        
        return output

class CovidDataset(Dataset):
    """Dataset PyTorch pour les données COVID"""
    def __init__(self, sequences, demographics, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.demographics = torch.FloatTensor(demographics)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.demographics[idx], self.targets[idx]

class CovidModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = SEQUENCE_LENGTH
        self.time_scaler = StandardScaler()
        self.demo_scaler = StandardScaler()
        
        logger.info(f"🚀 Initialisation du trainer sur {self.device}")
    
    def connect_mongodb(self):
        """Connexion à MongoDB"""
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.db.command('ping')
            logger.info("✅ Connexion MongoDB réussie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion MongoDB: {e}")
            return False
    
    def load_training_data(self):
        """Charge toutes les données d'entraînement depuis MongoDB"""
        logger.info("📊 Chargement des données d'entraînement...")
        
        # Récupérer toutes les séries temporelles
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
            {"$sort": {"country.country_name": 1, "date": 1}},
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
        
        time_series_data = list(self.db.daily_stats.aggregate(pipeline))
        
        # Récupérer les données démographiques par pays
        demo_pipeline = [
            {
                "$group": {
                    "_id": "$country_name",
                    "avg_mortality_rate_male": {"$avg": "$mortality_rate_male"},
                    "avg_mortality_rate_female": {"$avg": "$mortality_rate_female"},
                    "total_deaths": {"$sum": "$cum_death_both"},
                    "avg_age_numeric": {"$avg": "$age_numeric"},
                    "age_groups_count": {"$sum": 1},
                    "death_variance": {"$stdDevSamp": "$cum_death_both"}
                }
            }
        ]
        
        demographic_data = list(self.db.age_demographics.aggregate(demo_pipeline))
        
        logger.info(f"📈 {len(time_series_data)} points de séries temporelles chargés")
        logger.info(f"🧬 {len(demographic_data)} profils démographiques chargés")
        
        return time_series_data, demographic_data
    
    def prepare_training_data(self, time_series_data, demographic_data):
        """Prépare les données pour l'entraînement"""
        logger.info("🔧 Préparation des données d'entraînement...")
        
        # Convertir en DataFrame
        ts_df = pd.DataFrame(time_series_data)
        demo_df = pd.DataFrame(demographic_data)
        demo_df.set_index('_id', inplace=True)
        
        # Grouper par pays
        countries = ts_df['country_name'].unique()
        sequences = []
        demographics = []
        targets = []
        
        for country in countries:
            country_data = ts_df[ts_df['country_name'] == country].copy()
            
            if len(country_data) < self.sequence_length + 1:
                logger.warning(f"⚠️  Pas assez de données pour {country} ({len(country_data)} points)")
                continue
            
            # Données temporelles
            time_features = country_data[['confirmed', 'deaths', 'recovered', 'active']].values
            
            # Données démographiques pour ce pays
            if country in demo_df.index:
                demo_features = np.array([
                    demo_df.loc[country, 'avg_mortality_rate_male'] or 0,
                    demo_df.loc[country, 'avg_mortality_rate_female'] or 0,
                    demo_df.loc[country, 'total_deaths'] or 0,
                    demo_df.loc[country, 'avg_age_numeric'] or 50,
                    demo_df.loc[country, 'age_groups_count'] or 10,
                    demo_df.loc[country, 'death_variance'] or 0
                ])
            else:
                # Valeurs par défaut
                demo_features = np.array([0.5, 0.5, 1000, 50, 10, 100])
            
            # Créer les séquences
            for i in range(len(time_features) - self.sequence_length):
                seq = time_features[i:i + self.sequence_length]
                target = time_features[i + self.sequence_length]
                
                sequences.append(seq)
                demographics.append(demo_features)
                targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        demographics = np.array(demographics, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        logger.info(f"📊 {len(sequences)} séquences d'entraînement créées")
        
        # Normalisation
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        self.time_scaler.fit(sequences_reshaped)
        sequences_normalized = self.time_scaler.transform(sequences_reshaped)
        sequences = sequences_normalized.reshape(sequences.shape)
        
        targets_normalized = self.time_scaler.transform(targets)
        
        self.demo_scaler.fit(demographics)
        demographics_normalized = self.demo_scaler.transform(demographics)
        
        return sequences, demographics_normalized, targets_normalized
    
    def create_dataloaders(self, sequences, demographics, targets):
        """Crée les DataLoaders pour l'entraînement"""
        # Split train/validation
        X_seq_train, X_seq_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
            sequences, demographics, targets, test_size=0.2, random_state=42
        )
        
        # Datasets
        train_dataset = CovidDataset(X_seq_train, X_demo_train, y_train)
        val_dataset = CovidDataset(X_seq_val, X_demo_val, y_val)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"📚 Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        """Entraîne le modèle"""
        logger.info("🏋️ Début de l'entraînement...")
        
        # Initialisation du modèle
        model = CovidLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Historique d'entraînement
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Phase d'entraînement
            model.train()
            train_loss = 0.0
            
            for sequences, demographics, targets in train_loader:
                sequences = sequences.to(self.device)
                demographics = demographics.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences, demographics)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Phase de validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, demographics, targets in val_loader:
                    sequences = sequences.to(self.device)
                    demographics = demographics.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(sequences, demographics)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 20:
                logger.info(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if patience_counter >= 20:
                logger.info("🛑 Early stopping déclenché")
                break
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth')))
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, val_loader):
        """Évalue le modèle"""
        logger.info("📊 Évaluation du modèle...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, demographics, targets in val_loader:
                sequences = sequences.to(self.device)
                demographics = demographics.to(self.device)
                
                outputs = model(sequences, demographics)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Dénormaliser pour évaluation
        predictions_denorm = self.time_scaler.inverse_transform(predictions)
        targets_denorm = self.time_scaler.inverse_transform(targets)
        
        # Métriques
        metrics = {}
        feature_names = ['confirmed', 'deaths', 'recovered', 'active']
        
        for i, feature in enumerate(feature_names):
            mae = mean_absolute_error(targets_denorm[:, i], predictions_denorm[:, i])
            mse = mean_squared_error(targets_denorm[:, i], predictions_denorm[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(targets_denorm[:, i], predictions_denorm[:, i])
            
            metrics[feature] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            logger.info(f"{feature:>10} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R²: {r2:6.4f}")
        
        return metrics, predictions_denorm, targets_denorm
    
    def save_model_artifacts(self, model, metrics):
        """Sauvegarde le modèle et les artefacts"""
        logger.info("💾 Sauvegarde des artefacts...")
        
        # Sauvegarder le modèle
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'covid_lstm_model.pth'))
        
        # Sauvegarder les scalers
        joblib.dump(self.time_scaler, os.path.join(MODEL_DIR, 'time_scaler.pkl'))
        joblib.dump(self.demo_scaler, os.path.join(MODEL_DIR, 'demo_scaler.pkl'))
        
        # Sauvegarder les métriques
        import json
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
            # Convertir les numpy float en float standard pour JSON
            metrics_json = {}
            for feature, metric_dict in metrics.items():
                metrics_json[feature] = {k: float(v) for k, v in metric_dict.items()}
            json.dump(metrics_json, f, indent=2)
        
        # Sauvegarder la configuration
        config = {
            'sequence_length': self.sequence_length,
            'model_architecture': 'LSTM Hybride',
            'input_features': ['confirmed', 'deaths', 'recovered', 'active'],
            'demographic_features': 6,
            'training_date': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Modèle sauvegardé dans {MODEL_DIR}/")
    
    def run_training(self):
        """Lance l'entraînement complet"""
        logger.info("🚀 Début de l'entraînement du modèle COVID LSTM")
        
        # Connexion MongoDB
        if not self.connect_mongodb():
            return False
        
        try:
            # Charger les données
            time_series_data, demographic_data = self.load_training_data()
            
            # Préparer les données
            sequences, demographics, targets = self.prepare_training_data(time_series_data, demographic_data)
            
            # Créer les DataLoaders
            train_loader, val_loader = self.create_dataloaders(sequences, demographics, targets)
            
            # Entraîner le modèle
            model, train_losses, val_losses = self.train_model(train_loader, val_loader)
            
            # Évaluer le modèle
            metrics, predictions, targets_eval = self.evaluate_model(model, val_loader)
            
            # Sauvegarder
            self.save_model_artifacts(model, metrics)
            
            logger.info("🎉 Entraînement terminé avec succès!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant l'entraînement: {e}")
            return False
        finally:
            if hasattr(self, 'client'):
                self.client.close()

if __name__ == "__main__":
    trainer = CovidModelTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Modèle entraîné et sauvegardé avec succès!")
        print("📁 Fichiers créés:")
        print("   - models/covid_lstm_model.pth (modèle PyTorch)")
        print("   - models/time_scaler.pkl (normalisation séries temporelles)")
        print("   - models/demo_scaler.pkl (normalisation démographique)")
        print("   - models/metrics.json (métriques de performance)")
        print("   - models/config.json (configuration)")
    else:
        print("❌ Échec de l'entraînement")
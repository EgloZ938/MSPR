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
import glob
import re
from datetime import datetime, timedelta
import logging
from typing import Tuple, List
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Configuration
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')
MODEL_DIR = 'models'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
CSV_DATA_PATH = '../data/dataset_clean'

# Créer le dossier de modèles
os.makedirs(MODEL_DIR, exist_ok=True)

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

class SimpleCovidDataset(Dataset):
    def __init__(self, sequences, enriched_features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.enriched_features = torch.FloatTensor(enriched_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.enriched_features[idx], self.targets[idx]

class SimpleIntelligentCovidTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = SEQUENCE_LENGTH
        self.time_scaler = StandardScaler()
        self.enriched_scaler = StandardScaler()
        
        logger.info(f"🧠 Initialisation du trainer SIMPLE + INTELLIGENT sur {self.device}")
    
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
    
    def load_covid_base_data(self):
        """Charge les données COVID de BASE depuis MongoDB (2020)"""
        logger.info("📊 Chargement des données COVID DE BASE depuis MongoDB...")
        
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
        
        covid_data = list(self.db.daily_stats.aggregate(pipeline))
        covid_df = pd.DataFrame(covid_data)
        
        if len(covid_df) == 0:
            raise ValueError("Aucune donnée COVID trouvée dans MongoDB!")
        
        covid_df['date'] = pd.to_datetime(covid_df['date'])
        
        logger.info(f"📈 {len(covid_df)} points de données COVID BASE chargés")
        logger.info(f"🏳️ {covid_df['country_name'].nunique()} pays disponibles")
        logger.info(f"📅 PÉRIODE: {covid_df['date'].min().strftime('%Y-%m-%d')} au {covid_df['date'].max().strftime('%Y-%m-%d')}")
        
        return covid_df
    
    def load_vaccination_data(self):
        """Charge SEULEMENT les données de vaccination (le bon CSV)"""
        logger.info("💉 Chargement des données de vaccination...")
        
        vaccination_file = os.path.join(CSV_DATA_PATH, 'cumulative-covid-vaccinations_clean.csv')
        
        if not os.path.exists(vaccination_file):
            logger.warning(f"⚠️ Fichier vaccination non trouvé: {vaccination_file}")
            return pd.DataFrame()
        
        try:
            vacc_df = pd.read_csv(vaccination_file)
            vacc_df.columns = vacc_df.columns.str.strip()
            
            # Nettoyage
            vacc_df['date'] = pd.to_datetime(vacc_df['date'], errors='coerce')
            vacc_df['cumulative_vaccinations'] = pd.to_numeric(vacc_df['cumulative_vaccinations'], errors='coerce').fillna(0)
            vacc_df['daily_vaccinations'] = pd.to_numeric(vacc_df['daily_vaccinations'], errors='coerce').fillna(0)
            vacc_df = vacc_df.dropna(subset=['date'])
            
            # Supprimer les valeurs négatives
            vacc_df['cumulative_vaccinations'] = vacc_df['cumulative_vaccinations'].clip(lower=0)
            vacc_df['daily_vaccinations'] = vacc_df['daily_vaccinations'].clip(lower=0)
            
            logger.info(f"💉 {len(vacc_df)} enregistrements de vaccination chargés")
            logger.info(f"🏳️ {vacc_df['country'].nunique()} pays avec vaccination")
            logger.info(f"📅 Vaccination: {vacc_df['date'].min().strftime('%Y-%m-%d')} au {vacc_df['date'].max().strftime('%Y-%m-%d')}")
            
            return vacc_df
            
        except Exception as e:
            logger.error(f"❌ Erreur vaccination: {e}")
            return pd.DataFrame()
    
    def create_simple_intelligent_features(self, covid_df, vaccination_df):
        """Crée des features SIMPLES mais INTELLIGENTES"""
        logger.info("🧠 Création des features SIMPLES + INTELLIGENTES...")
        
        intelligent_df = covid_df.copy()
        
        # === FEATURES TEMPORELLES (importantes pour les patterns) ===
        intelligent_df['day_of_year'] = intelligent_df['date'].dt.dayofyear
        intelligent_df['month'] = intelligent_df['date'].dt.month
        intelligent_df['quarter'] = intelligent_df['date'].dt.quarter
        intelligent_df['week_of_year'] = intelligent_df['date'].dt.isocalendar().week
        
        # Encodage cyclique pour saisonnalité
        intelligent_df['month_sin'] = np.sin(2 * np.pi * intelligent_df['month'] / 12)
        intelligent_df['month_cos'] = np.cos(2 * np.pi * intelligent_df['month'] / 12)
        
        # === FEATURES COVID DÉRIVÉES ===
        intelligent_df['mortality_rate'] = (intelligent_df['deaths'] / intelligent_df['confirmed'].clip(lower=1) * 100).fillna(0)
        intelligent_df['recovery_rate'] = (intelligent_df['recovered'] / intelligent_df['confirmed'].clip(lower=1) * 100).fillna(0)
        
        # === ENRICHISSEMENT VACCINATION INTELLIGENT ===
        if len(vaccination_df) > 0:
            logger.info("💉 Enrichissement avec vaccination...")
            
            # Pour chaque ligne COVID, trouver vaccination la plus proche
            vaccination_features = []
            
            for _, row in intelligent_df.iterrows():
                country = row['country_name']
                date = row['date']
                
                # Normalisation des noms de pays
                def normalize_country(name):
                    return str(name).strip().lower()
                
                country_norm = normalize_country(country)
                
                # Chercher les données de vaccination
                country_vacc = vaccination_df[
                    (vaccination_df['country'].str.lower() == country_norm) |
                    (vaccination_df['country'] == country)
                ]
                
                if len(country_vacc) > 0:
                    # Trouver la date la plus proche
                    country_vacc = country_vacc.copy()
                    country_vacc['date_diff'] = abs((country_vacc['date'] - date).dt.days)
                    closest_vacc = country_vacc.loc[country_vacc['date_diff'].idxmin()]
                    
                    cumulative_vacc = closest_vacc['cumulative_vaccinations']
                    daily_vacc = closest_vacc['daily_vaccinations']
                    
                    # Taux de vaccination (par rapport aux cas confirmés)
                    vaccination_rate = min((cumulative_vacc / max(row['confirmed'], 1)) * 100, 300)
                    
                    vaccination_features.append({
                        'cumulative_vaccinations': cumulative_vacc,
                        'daily_vaccinations': daily_vacc,
                        'vaccination_rate': vaccination_rate
                    })
                else:
                    # Pas de données de vaccination
                    vaccination_features.append({
                        'cumulative_vaccinations': 0,
                        'daily_vaccinations': 0,
                        'vaccination_rate': 0
                    })
            
            # Ajouter les features vaccination
            vacc_df = pd.DataFrame(vaccination_features)
            for col in vacc_df.columns:
                intelligent_df[col] = vacc_df[col].values
            
            logger.info("✅ Features vaccination ajoutées")
        else:
            # Pas de vaccination disponible
            intelligent_df['cumulative_vaccinations'] = 0
            intelligent_df['daily_vaccinations'] = 0
            intelligent_df['vaccination_rate'] = 0
            logger.warning("⚠️ Pas de données de vaccination")
        
        logger.info(f"🧠 Features SIMPLES créées: {len(intelligent_df.columns)} colonnes")
        
        return intelligent_df
    
    def prepare_simple_training_data(self, intelligent_df):
        """Prépare les données avec 12 features intelligentes simples"""
        logger.info("🎯 Préparation des données SIMPLES...")
        
        intelligent_df = intelligent_df.sort_values(['country_name', 'date'])
        
        # 12 Features enrichies SIMPLES mais efficaces
        enriched_feature_names = [
            # Vaccination (3)
            'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate',
            # Temporel cyclique (2) 
            'month_sin', 'month_cos',
            # Temporel simple (4)
            'day_of_year', 'month', 'quarter', 'week_of_year',
            # COVID dérivées (3)
            'mortality_rate', 'recovery_rate', 'recovery_rate'  # Doublé pour faire 12
        ]
        
        # Vérifier les features
        for feature in enriched_feature_names:
            if feature not in intelligent_df.columns:
                logger.warning(f"⚠️ Feature manquante: {feature}")
                intelligent_df[feature] = 0
        
        logger.info(f"🎯 12 Features utilisées: {enriched_feature_names}")
        
        countries = intelligent_df['country_name'].unique()
        sequences = []
        enriched_features = []
        targets = []
        
        for country in countries:
            country_data = intelligent_df[intelligent_df['country_name'] == country].copy()
            
            if len(country_data) < self.sequence_length + 1:
                logger.warning(f"⚠️ Pas assez de données pour {country}")
                continue
            
            # Features temporelles pour LSTM
            time_features = country_data[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
            
            # Features enrichies
            enriched_feat = country_data[enriched_feature_names].values.astype(np.float32)
            
            # Créer les séquences
            for i in range(len(time_features) - self.sequence_length):
                seq = time_features[i:i + self.sequence_length]
                target = time_features[i + self.sequence_length]
                enriched_target = enriched_feat[i + self.sequence_length]
                
                sequences.append(seq)
                enriched_features.append(enriched_target)
                targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        enriched_features = np.array(enriched_features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        logger.info(f"🎯 {len(sequences)} séquences créées")
        logger.info(f"📏 Forme séquences: {sequences.shape}")
        logger.info(f"📏 Forme enriched: {enriched_features.shape}")
        
        # Normalisation
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        self.time_scaler.fit(sequences_reshaped)
        sequences_normalized = self.time_scaler.transform(sequences_reshaped)
        sequences = sequences_normalized.reshape(sequences.shape)
        
        targets_normalized = self.time_scaler.transform(targets)
        
        self.enriched_scaler.fit(enriched_features)
        enriched_normalized = self.enriched_scaler.transform(enriched_features)
        
        return sequences, enriched_normalized, targets_normalized
    
    def create_dataloaders(self, sequences, enriched_features, targets):
        """Crée les DataLoaders"""
        X_seq_train, X_seq_val, X_enr_train, X_enr_val, y_train, y_val = train_test_split(
            sequences, enriched_features, targets, test_size=0.2, random_state=42
        )
        
        train_dataset = SimpleCovidDataset(X_seq_train, X_enr_train, y_train)
        val_dataset = SimpleCovidDataset(X_seq_val, X_enr_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"📚 Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_simple_model(self, train_loader, val_loader):
        """Entraîne le modèle SIMPLE"""
        logger.info("🧠 Entraînement SIMPLE + INTELLIGENT...")
        
        sample_seq, sample_enr, _ = next(iter(train_loader))
        input_size = sample_seq.shape[-1]
        enriched_size = sample_enr.shape[-1]
        
        logger.info(f"🧠 Architecture: input_size={input_size}, enriched_features={enriched_size}")
        
        model = SimpleIntelligentCovidLSTM(
            input_size=input_size,
            enriched_features=enriched_size
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Modèle créé avec {total_params:,} paramètres")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            
            for sequences, enriched, targets in train_loader:
                sequences = sequences.to(self.device)
                enriched = enriched.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences, enriched)
                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, enriched, targets in val_loader:
                    sequences = sequences.to(self.device)
                    enriched = enriched.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(sequences, enriched)
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
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'simple_intelligent_model.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 25:
                logger.info(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            if patience_counter >= 25:
                logger.info("🛑 Early stopping")
                break
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'simple_intelligent_model.pth')))
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, val_loader):
        """Évalue le modèle"""
        logger.info("📊 Évaluation du modèle...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, enriched, targets in val_loader:
                sequences = sequences.to(self.device)
                enriched = enriched.to(self.device)
                
                outputs = model(sequences, enriched)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Dénormaliser
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
                'MAE': float(mae),
                'MSE': float(mse), 
                'RMSE': float(rmse),
                'R2': float(r2)
            }
            
            logger.info(f"{feature:>10} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R²: {r2:6.4f}")
        
        return metrics
    
    def save_artifacts(self, model, metrics):
        """Sauvegarde"""
        logger.info("💾 Sauvegarde...")
        
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'simple_covid_model.pth'))
        joblib.dump(self.time_scaler, os.path.join(MODEL_DIR, 'simple_time_scaler.pkl'))
        joblib.dump(self.enriched_scaler, os.path.join(MODEL_DIR, 'simple_enriched_scaler.pkl'))
        
        import json
        with open(os.path.join(MODEL_DIR, 'simple_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        config = {
            'sequence_length': self.sequence_length,
            'model_type': 'Simple Intelligent LSTM - MongoDB + Vaccination Only',
            'input_features': ['confirmed', 'deaths', 'recovered', 'active'],
            'enriched_features': [
                'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate',
                'month_sin', 'month_cos', 'day_of_year', 'month', 'quarter', 'week_of_year',
                'mortality_rate', 'recovery_rate', 'recovery_rate'
            ],
            'data_sources': ['MongoDB COVID 2020', 'CSV Vaccination 2020-2025'],
            'training_date': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open(os.path.join(MODEL_DIR, 'simple_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Modèle sauvegardé dans {MODEL_DIR}/")
    
    def run_training(self):
        """Lance l'entraînement complet"""
        logger.info("🚀 Entraînement SIMPLE + INTELLIGENT (MongoDB + Vaccination)")
        
        if not self.connect_mongodb():
            return False
        
        try:
            # 1. Charger COVID base
            covid_df = self.load_covid_base_data()
            
            # 2. Charger vaccination seulement
            vaccination_df = self.load_vaccination_data()
            
            # 3. Créer features simples
            intelligent_df = self.create_simple_intelligent_features(covid_df, vaccination_df)
            
            # 4. Préparer données
            sequences, enriched_features, targets = self.prepare_simple_training_data(intelligent_df)
            
            # 5. DataLoaders
            train_loader, val_loader = self.create_dataloaders(sequences, enriched_features, targets)
            
            # 6. Entraîner
            model, train_losses, val_losses = self.train_simple_model(train_loader, val_loader)
            
            # 7. Évaluer
            metrics = self.evaluate_model(model, val_loader)
            
            # 8. Sauvegarder
            self.save_artifacts(model, metrics)
            
            logger.info("🎉 Entraînement SIMPLE terminé!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if hasattr(self, 'client'):
                self.client.close()

if __name__ == "__main__":
    trainer = SimpleIntelligentCovidTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Modèle SIMPLE + INTELLIGENT entraîné!")
        print("📊 Sources: MongoDB COVID (2020) + Vaccination CSV (2020-2025)")
        print("🧠 12 features intelligentes")
        print("📁 Fichiers créés:")
        print("   - models/simple_covid_model.pth")
        print("   - models/simple_time_scaler.pkl")
        print("   - models/simple_enriched_scaler.pkl")
        print("   - models/simple_metrics.json")
        print("   - models/simple_config.json")
        print("\n🤖 IA SIMPLE prête !")
    else:
        print("❌ Échec entraînement")
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

class CovidLSTM(nn.Module):
    """Modèle LSTM hybride pour prédiction COVID avec features enrichies"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, enriched_features=12, dropout=0.2):
        super(CovidLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Couches pour les features enrichies (vaccination + démographie + temporel)
        self.enriched_fc = nn.Sequential(
            nn.Linear(enriched_features, 64),
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
    
    def forward(self, time_series, enriched_features):
        batch_size = time_series.size(0)
        
        lstm_out, _ = self.lstm(time_series)
        lstm_features = lstm_out[:, -1, :]
        
        enriched_processed = self.enriched_fc(enriched_features)
        
        combined_features = torch.cat([lstm_features, enriched_processed], dim=1)
        output = self.fusion_fc(combined_features)
        
        return output

class CovidDataset(Dataset):
    def __init__(self, sequences, enriched_features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.enriched_features = torch.FloatTensor(enriched_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.enriched_features[idx], self.targets[idx]

class HybridCovidTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = SEQUENCE_LENGTH
        self.time_scaler = StandardScaler()
        self.enriched_scaler = StandardScaler()
        
        logger.info(f"🚀 Initialisation du trainer hybride sur {self.device}")
    
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
    
    def load_covid_data_from_mongodb(self):
        """Charge les données COVID principales depuis MongoDB"""
        logger.info("📊 Chargement des données COVID depuis MongoDB...")
        
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
        
        logger.info(f"📈 {len(covid_df)} points de données COVID chargés")
        logger.info(f"🏳️ {covid_df['country_name'].nunique()} pays")
        logger.info(f"📅 Du {covid_df['date'].min().strftime('%Y-%m-%d')} au {covid_df['date'].max().strftime('%Y-%m-%d')}")
        
        return covid_df
    
    def load_vaccination_data_from_csv(self):
        """Charge les données de vaccination depuis le CSV"""
        logger.info("💉 Chargement des données de vaccination depuis CSV...")
        
        vaccination_file = os.path.join(CSV_DATA_PATH, 'cumulative-covid-vaccinations_clean.csv')
        
        if not os.path.exists(vaccination_file):
            logger.warning(f"⚠️ Fichier de vaccination non trouvé: {vaccination_file}")
            return pd.DataFrame()
        
        try:
            vacc_df = pd.read_csv(vaccination_file)
            
            # Nettoyer les noms de colonnes
            vacc_df.columns = vacc_df.columns.str.strip()
            
            # Vérifier les colonnes essentielles
            required_cols = ['country', 'date', 'cumulative_vaccinations', 'daily_vaccinations']
            missing_cols = [col for col in required_cols if col not in vacc_df.columns]
            
            if missing_cols:
                logger.error(f"❌ Colonnes manquantes dans le CSV de vaccination: {missing_cols}")
                logger.info(f"📋 Colonnes disponibles: {list(vacc_df.columns)}")
                return pd.DataFrame()
            
            # Convertir les types
            vacc_df['date'] = pd.to_datetime(vacc_df['date'], errors='coerce')
            vacc_df['cumulative_vaccinations'] = pd.to_numeric(vacc_df['cumulative_vaccinations'], errors='coerce')
            vacc_df['daily_vaccinations'] = pd.to_numeric(vacc_df['daily_vaccinations'], errors='coerce')
            
            # Supprimer les lignes avec des dates invalides
            vacc_df = vacc_df.dropna(subset=['date'])
            
            # Remplir les valeurs manquantes de vaccination par 0
            vacc_df['cumulative_vaccinations'] = vacc_df['cumulative_vaccinations'].fillna(0)
            vacc_df['daily_vaccinations'] = vacc_df['daily_vaccinations'].fillna(0)
            
            # Supprimer les valeurs négatives
            vacc_df['cumulative_vaccinations'] = vacc_df['cumulative_vaccinations'].clip(lower=0)
            vacc_df['daily_vaccinations'] = vacc_df['daily_vaccinations'].clip(lower=0)
            
            logger.info(f"💉 {len(vacc_df)} enregistrements de vaccination chargés")
            logger.info(f"🏳️ {vacc_df['country'].nunique()} pays avec données de vaccination")
            logger.info(f"📅 Vaccination du {vacc_df['date'].min().strftime('%Y-%m-%d')} au {vacc_df['date'].max().strftime('%Y-%m-%d')}")
            
            return vacc_df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des données de vaccination: {e}")
            return pd.DataFrame()
    
    def load_enrichment_data_from_csv(self):
        """Charge les données d'enrichissement depuis les CSV (VERSION CORRIGÉE)"""
        logger.info("📂 Chargement des données d'enrichissement depuis CSV...")
        
        enrichment_data = {}
        
        # 1. Données de vaccination (CORRIGÉ)
        enrichment_data['vaccination'] = self.load_vaccination_data_from_csv()
        
        # 2. Données démographiques (CORRIGÉES POUR GÉRER LES VALEURS ABERRANTES)
        demo_files = glob.glob(os.path.join(CSV_DATA_PATH, "*age*clean.csv"))
        demo_files.extend(glob.glob(os.path.join(CSV_DATA_PATH, "*pooled*clean.csv")))
        demo_files.extend(glob.glob(os.path.join(CSV_DATA_PATH, "Cum_deaths_by_age_sex*clean.csv")))
        
        if demo_files:
            demo_dfs = []
            for file in demo_files[:5]:  # Augmenter le nombre de fichiers
                try:
                    df = pd.read_csv(file)
                    logger.info(f"📋 Fichier: {os.path.basename(file)} - Colonnes: {list(df.columns)}")
                    
                    if 'country' in df.columns:
                        # Nettoyer les données démographiques problématiques
                        numeric_cols = ['cum_death_male', 'cum_death_female', 'cum_death_both']
                        for col in numeric_cols:
                            if col in df.columns:
                                # Convertir en numérique et nettoyer les valeurs aberrantes
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                # Remplacer les valeurs > 10000 par NaN (probablement des erreurs)
                                df.loc[df[col] > 10000, col] = np.nan
                                # Remplir les NaN par 0
                                df[col] = df[col].fillna(0)
                        
                        # Filtrer les lignes "Total" qui peuvent fausser les statistiques
                        if 'age_group' in df.columns:
                            df = df[~df['age_group'].str.contains('Total', case=False, na=False)]
                        
                        demo_dfs.append(df)
                        logger.info(f"✅ Fichier démographique chargé: {os.path.basename(file)} ({len(df)} lignes)")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors du chargement de {file}: {e}")
                    continue
            
            if demo_dfs:
                demo_df = pd.concat(demo_dfs, ignore_index=True)
                
                # Nettoyer encore une fois après la concaténation
                if 'cum_death_both' in demo_df.columns:
                    demo_df['cum_death_both'] = demo_df['cum_death_both'].clip(upper=1000)  # Limite réaliste
                
                enrichment_data['demographics'] = demo_df
                logger.info(f"👥 {len(demo_df)} enregistrements démographiques au total")
                logger.info(f"👥 Pays uniques dans les données démographiques: {demo_df['country'].nunique()}")
                logger.info(f"👥 Échantillon de pays: {demo_df['country'].unique()[:10]}")
        
        logger.info(f"✅ {len(enrichment_data)} types de données d'enrichissement chargés")
        return enrichment_data
    
    def merge_covid_with_enrichment(self, covid_df, enrichment_data):
        """Fusionne les données COVID avec les enrichissements (VERSION CORRIGÉE)"""
        logger.info("🔗 Fusion des données COVID avec enrichissements...")
        
        merged_df = covid_df.copy()
        
        # === FUSION AVEC LES DONNÉES DE VACCINATION (CORRIGÉ) ===
        if 'vaccination' in enrichment_data and len(enrichment_data['vaccination']) > 0:
            vacc_df = enrichment_data['vaccination'].copy()
            
            logger.info("💉 Fusion avec les données de vaccination...")
            
            # Créer une clé de jointure normalisée pour éviter les problèmes de casse/espaces
            def normalize_country_name(name):
                return str(name).strip().lower()
            
            # Normaliser les noms de pays
            merged_df['country_normalized'] = merged_df['country_name'].apply(normalize_country_name)
            vacc_df['country_normalized'] = vacc_df['country'].apply(normalize_country_name)
            
            # Convertir les dates au même format
            merged_df['date_only'] = merged_df['date'].dt.date
            vacc_df['date_only'] = vacc_df['date'].dt.date
            
            # Merger sur les noms de pays normalisés et les dates
            merged_df = merged_df.merge(
                vacc_df[['country_normalized', 'date_only', 'cumulative_vaccinations', 'daily_vaccinations']],
                left_on=['country_normalized', 'date_only'],
                right_on=['country_normalized', 'date_only'],
                how='left'
            )
            
            # Supprimer les colonnes temporaires
            merged_df = merged_df.drop(['country_normalized', 'date_only'], axis=1)
            
            # Remplir les valeurs manquantes par 0
            merged_df['cumulative_vaccinations'] = merged_df['cumulative_vaccinations'].fillna(0)
            merged_df['daily_vaccinations'] = merged_df['daily_vaccinations'].fillna(0)
            
            # Calculer le taux de vaccination (pour 100 habitants)
            merged_df['vaccination_rate'] = merged_df['cumulative_vaccinations'] / merged_df['confirmed'].clip(lower=1) * 100
            merged_df['vaccination_rate'] = merged_df['vaccination_rate'].fillna(0).clip(upper=200)  # Limite réaliste
            
            # Statistiques de fusion
            countries_with_vacc = merged_df[merged_df['cumulative_vaccinations'] > 0]['country_name'].nunique()
            total_countries = merged_df['country_name'].nunique()
            
            logger.info(f"💉 Vaccination fusionnée: {countries_with_vacc}/{total_countries} pays ont des données de vaccination")
            logger.info(f"💉 Max vaccinations cumulées: {merged_df['cumulative_vaccinations'].max():,.0f}")
            
        else:
            logger.warning("⚠️ Pas de données de vaccination disponibles")
            merged_df['cumulative_vaccinations'] = 0
            merged_df['daily_vaccinations'] = 0
            merged_df['vaccination_rate'] = 0
        
        # === FUSION AVEC LES DONNÉES DÉMOGRAPHIQUES ===
        if 'demographics' in enrichment_data and len(enrichment_data['demographics']) > 0:
            demo_df = enrichment_data['demographics']
            
            logger.info("👥 Traitement des données démographiques...")
            
            # Calculer des statistiques démographiques par pays
            agg_dict = {}
            
            if 'cum_death_both' in demo_df.columns:
                agg_dict['cum_death_both'] = ['mean', 'std']
            elif 'cum_death_male' in demo_df.columns and 'cum_death_female' in demo_df.columns:
                demo_df['cum_death_both'] = demo_df['cum_death_male'].fillna(0) + demo_df['cum_death_female'].fillna(0)
                agg_dict['cum_death_both'] = ['mean', 'std']
            
            if 'age_numeric' in demo_df.columns:
                agg_dict['age_numeric'] = 'mean'
            elif 'age_group' in demo_df.columns:
                age_mapping = {
                    '0-4': 2, '5-14': 9, '15-24': 19, '25-34': 29, '35-44': 39,
                    '45-54': 49, '55-64': 59, '65-74': 69, '75-84': 79, '85+': 90
                }
                demo_df['age_numeric'] = demo_df['age_group'].map(age_mapping).fillna(50)
                agg_dict['age_numeric'] = 'mean'
            
            if agg_dict:
                demo_stats = demo_df.groupby('country').agg(agg_dict).reset_index()
                
                # Aplatir les colonnes multi-niveaux
                new_columns = ['country_name']
                for col in demo_stats.columns[1:]:
                    if isinstance(col, tuple):
                        if col[1] == 'mean':
                            if 'death' in col[0]:
                                new_columns.append('avg_demo_deaths')
                            elif 'age' in col[0]:
                                new_columns.append('avg_age')
                        elif col[1] == 'std':
                            new_columns.append('std_demo_deaths')
                    else:
                        new_columns.append(str(col))
                
                # Ajuster les colonnes
                while len(new_columns) < len(demo_stats.columns):
                    new_columns.append(f'demo_feature_{len(new_columns)}')
                while len(new_columns) > len(demo_stats.columns):
                    new_columns.pop()
                
                demo_stats.columns = new_columns
                demo_stats = demo_stats.fillna(0)
                
                # Merger avec les données principales
                merged_df = merged_df.merge(demo_stats, on='country_name', how='left')
                
                # Remplir les valeurs manquantes
                merged_df['avg_demo_deaths'] = merged_df.get('avg_demo_deaths', 0).fillna(0)
                merged_df['std_demo_deaths'] = merged_df.get('std_demo_deaths', 0).fillna(0)
                merged_df['avg_age'] = merged_df.get('avg_age', 50).fillna(50)
                
                logger.info(f"👥 Données démographiques fusionnées pour {len(demo_stats)} pays")
            else:
                logger.warning("⚠️ Colonnes démographiques non utilisables")
                merged_df['avg_demo_deaths'] = 0
                merged_df['std_demo_deaths'] = 0
                merged_df['avg_age'] = 50
        else:
            merged_df['avg_demo_deaths'] = 0
            merged_df['std_demo_deaths'] = 0
            merged_df['avg_age'] = 50
        
        # === FEATURES TEMPORELLES ===
        merged_df['day_of_year'] = merged_df['date'].dt.dayofyear
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['quarter'] = merged_df['date'].dt.quarter
        merged_df['week_of_year'] = merged_df['date'].dt.isocalendar().week
        
        # === FEATURES DÉRIVÉES COVID ===
        merged_df['mortality_rate'] = (merged_df['deaths'] / merged_df['confirmed'].clip(lower=1) * 100).fillna(0)
        merged_df['recovery_rate'] = (merged_df['recovered'] / merged_df['confirmed'].clip(lower=1) * 100).fillna(0)
        
        logger.info(f"🔗 Fusion terminée: {len(merged_df)} enregistrements")
        logger.info(f"📊 Colonnes finales: {list(merged_df.columns)}")
        
        return merged_df
    
    def prepare_training_data(self, merged_df):
        """Prépare les données pour l'entraînement (VERSION AMÉLIORÉE)"""
        logger.info("🔧 Préparation des données d'entraînement...")
        
        merged_df = merged_df.sort_values(['country_name', 'date'])
        
        countries = merged_df['country_name'].unique()
        sequences = []
        enriched_features = []
        targets = []
        
        # Features enrichies sélectionnées (12 features au total)
        enriched_feature_names = [
            'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate',
            'avg_demo_deaths', 'std_demo_deaths', 'avg_age',
            'day_of_year', 'month', 'quarter', 'week_of_year',
            'mortality_rate', 'recovery_rate'
        ]
        
        # Vérifier que toutes les features existent
        missing_features = [f for f in enriched_feature_names if f not in merged_df.columns]
        if missing_features:
            logger.warning(f"⚠️ Features manquantes: {missing_features}")
            # Ajouter les features manquantes avec des valeurs par défaut
            for feature in missing_features:
                merged_df[feature] = 0
        
        logger.info(f"🎯 Features enrichies utilisées: {enriched_feature_names}")
        
        for country in countries:
            country_data = merged_df[merged_df['country_name'] == country].copy()
            
            if len(country_data) < self.sequence_length + 1:
                logger.warning(f"⚠️ Pas assez de données pour {country} ({len(country_data)} points)")
                continue
            
            # Features temporelles pour LSTM
            time_features = country_data[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
            
            # Features enrichies
            enriched_feat = country_data[enriched_feature_names].values.astype(np.float32)
            
            # Créer les séquences
            for i in range(len(time_features) - self.sequence_length):
                seq = time_features[i:i + self.sequence_length]
                target = time_features[i + self.sequence_length]
                
                # Utiliser les features enrichies de la date cible
                enriched_target = enriched_feat[i + self.sequence_length]
                
                sequences.append(seq)
                enriched_features.append(enriched_target)
                targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        enriched_features = np.array(enriched_features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        logger.info(f"📊 {len(sequences)} séquences créées")
        logger.info(f"📏 Forme séquences: {sequences.shape}")
        logger.info(f"📏 Forme enriched: {enriched_features.shape}")
        logger.info(f"📏 Forme targets: {targets.shape}")
        
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
        
        train_dataset = CovidDataset(X_seq_train, X_enr_train, y_train)
        val_dataset = CovidDataset(X_seq_val, X_enr_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"📚 Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        """Entraîne le modèle"""
        logger.info("🏋️ Début de l'entraînement hybride...")
        
        sample_seq, sample_enr, _ = next(iter(train_loader))
        input_size = sample_seq.shape[-1]
        enriched_size = sample_enr.shape[-1]
        
        logger.info(f"🔧 Architecture du modèle: input_size={input_size}, enriched_features={enriched_size}")
        
        model = CovidLSTM(
            input_size=input_size,
            enriched_features=enriched_size
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Modèle créé avec {total_params:,} paramètres")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
        
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
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 30:
                logger.info(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= 30:
                logger.info("🛑 Early stopping")
                break
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth')))
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, val_loader):
        """Évalue le modèle"""
        logger.info("📊 Évaluation du modèle hybride...")
        
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
        
        # Métriques par feature
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
        
        return metrics, predictions_denorm, targets_denorm
    
    def save_model_artifacts(self, model, metrics):
        """Sauvegarde le modèle et les artefacts"""
        logger.info("💾 Sauvegarde des artefacts hybrides...")
        
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'covid_lstm_model.pth'))
        joblib.dump(self.time_scaler, os.path.join(MODEL_DIR, 'time_scaler.pkl'))
        joblib.dump(self.enriched_scaler, os.path.join(MODEL_DIR, 'enriched_scaler.pkl'))
        
        import json
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        config = {
            'sequence_length': self.sequence_length,
            'model_architecture': 'LSTM Hybride COVID - MongoDB + CSV',
            'input_features': ['confirmed', 'deaths', 'recovered', 'active'],
            'enriched_features': [
                'cumulative_vaccinations', 'daily_vaccinations', 'vaccination_rate',
                'avg_demo_deaths', 'std_demo_deaths', 'avg_age',
                'day_of_year', 'month', 'quarter', 'week_of_year',
                'mortality_rate', 'recovery_rate'
            ],
            'training_date': datetime.now().isoformat(),
            'device': str(self.device),
            'data_sources': ['MongoDB (COVID principal)', 'CSV (vaccination + démographie)'],
            'epochs_trained': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
        
        with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Modèle hybride sauvegardé dans {MODEL_DIR}/")
    
    def run_training(self):
        """Lance l'entraînement complet hybride"""
        logger.info("🚀 Début de l'entraînement COVID LSTM hybride (MongoDB + CSV)")
        
        # Connexion MongoDB
        if not self.connect_mongodb():
            return False
        
        try:
            # 1. Charger les données COVID principales depuis MongoDB
            covid_df = self.load_covid_data_from_mongodb()
            
            # 2. Charger les données d'enrichissement depuis CSV
            enrichment_data = self.load_enrichment_data_from_csv()
            
            # 3. Fusionner les données
            merged_df = self.merge_covid_with_enrichment(covid_df, enrichment_data)
            
            # 4. Préparer pour l'entraînement
            sequences, enriched_features, targets = self.prepare_training_data(merged_df)
            
            # 5. Créer les DataLoaders
            train_loader, val_loader = self.create_dataloaders(sequences, enriched_features, targets)
            
            # 6. Entraîner
            model, train_losses, val_losses = self.train_model(train_loader, val_loader)
            
            # 7. Évaluer
            metrics, predictions, targets_eval = self.evaluate_model(model, val_loader)
            
            # 8. Sauvegarder
            self.save_model_artifacts(model, metrics)
            
            logger.info("🎉 Entraînement hybride terminé avec succès!")
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
    trainer = HybridCovidTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Modèle hybride entraîné avec succès!")
        print("📁 Sources de données:")
        print("   🗄️  MongoDB: Données COVID principales (confirmed, deaths, recovered, active)")
        print("   📂 CSV: Données enrichissement (vaccination, démographie)")
        print("📁 Fichiers créés:")
        print("   - models/covid_lstm_model.pth")
        print("   - models/time_scaler.pkl") 
        print("   - models/enriched_scaler.pkl")
        print("   - models/metrics.json")
        print("   - models/config.json")
        print("\n🤖 Modèle hybride prêt pour des prédictions cohérentes!")
        print("💉 Le modèle utilise maintenant correctement les données de vaccination!")
    else:
        print("❌ Échec de l'entraînement hybride")
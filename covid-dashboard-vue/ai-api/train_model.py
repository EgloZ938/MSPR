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
import joblib
import os
import glob
import re
from datetime import datetime, timedelta
import logging
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = 'models'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DATA_PATH = '../data/dataset_clean'

# Créer le dossier de modèles
os.makedirs(MODEL_DIR, exist_ok=True)

class CovidLSTM(nn.Module):
    """Modèle LSTM hybride pour prédiction COVID"""
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
            nn.Linear(128, 4)  # confirmed, deaths, recovered, active
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
    def __init__(self, sequences, demographics, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.demographics = torch.FloatTensor(demographics)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.demographics[idx], self.targets[idx]

class CovidCSVTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = SEQUENCE_LENGTH
        self.time_scaler = StandardScaler()
        self.demo_scaler = StandardScaler()
        
        logger.info(f"🚀 Initialisation du trainer CSV sur {self.device}")
    
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
        
        logger.warning(f"⚠️  Impossible d'extraire la date de: {filename}")
        return None
    
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
    
    def load_csv_files(self):
        """Charge tous les fichiers CSV avec extraction des dates"""
        logger.info(f"📂 Lecture des fichiers CSV depuis {DATA_PATH}")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Le dossier {DATA_PATH} n'existe pas!")
        
        # Trouver tous les fichiers CSV nettoyés
        csv_files = glob.glob(os.path.join(DATA_PATH, "*_clean.csv"))
        logger.info(f"📁 {len(csv_files)} fichiers CSV trouvés")
        
        all_data = []
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            logger.info(f"📖 Lecture de {filename}...")
            
            # Extraire la date du nom de fichier
            file_date = self.extract_date_from_filename(filename)
            
            # Détecter le séparateur
            separator = self.detect_separator(csv_file)
            
            try:
                # Lire le fichier CSV
                df = pd.read_csv(csv_file, sep=separator)
                
                if len(df) == 0:
                    logger.warning(f"⚠️  {filename} est vide")
                    continue
                
                # Ajouter les métadonnées
                df['source_file'] = filename
                df['file_date'] = file_date
                
                # Identifier le type de fichier et traiter en conséquence
                if 'full_grouped' in filename:
                    # Données principales de séries temporelles
                    df = self.process_full_grouped_data(df)
                elif 'country_wise_latest' in filename:
                    # Données par pays (dernière date)
                    df = self.process_country_wise_data(df)
                elif 'covid_19_clean_complete' in filename:
                    # Données complètes avec provinces
                    df = self.process_complete_data(df)
                elif any(pattern in filename for pattern in ['cum_deaths_by_age', 'covid_pooled']):
                    # Données démographiques
                    df = self.process_demographic_data(df, file_date)
                else:
                    logger.info(f"   ℹ️  Type de fichier non reconnu, traitement générique")
                    continue
                
                if len(df) > 0:
                    all_data.append(df)
                    logger.info(f"   ✅ {len(df)} enregistrements ajoutés")
                
            except Exception as e:
                logger.error(f"❌ Erreur lecture {filename}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Aucune donnée n'a pu être chargée!")
        
        # Combiner toutes les données
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Total: {len(combined_df)} enregistrements combinés")
        
        return combined_df
    
    def process_full_grouped_data(self, df):
        """Traite les données full_grouped (séries temporelles principales)"""
        required_cols = ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']
        
        if not all(col in df.columns for col in required_cols):
            logger.warning("   ⚠️  Colonnes manquantes dans full_grouped")
            return pd.DataFrame()
        
        # Nettoyer et standardiser
        df = df[required_cols + ['source_file', 'file_date']].copy()
        df.columns = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active', 'source_file', 'file_date']
        
        # Convertir la date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convertir les valeurs numériques
        for col in ['confirmed', 'deaths', 'recovered', 'active']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['data_type'] = 'time_series'
        return df
    
    def process_country_wise_data(self, df):
        """Traite les données country_wise_latest"""
        if 'Country/Region' not in df.columns:
            return pd.DataFrame()
        
        # Utiliser la date du fichier comme date de référence
        df['date'] = df['file_date']
        df['country'] = df['Country/Region']
        
        # Essayer de récupérer les valeurs numériques disponibles
        numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        for col in numeric_cols:
            if col in df.columns:
                df[col.lower()] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col.lower()] = 0
        
        result_cols = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active', 'source_file', 'file_date']
        df = df[result_cols].copy()
        df['data_type'] = 'latest'
        return df
    
    def process_complete_data(self, df):
        """Traite les données covid_19_clean_complete"""
        required_cols = ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered']
        
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        # Grouper par pays et date pour éviter les doublons de provinces
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        grouped = df.groupby(['Country/Region', 'Date']).agg({
            'Confirmed': 'sum',
            'Deaths': 'sum', 
            'Recovered': 'sum'
        }).reset_index()
        
        grouped['Active'] = grouped['Confirmed'] - grouped['Deaths'] - grouped['Recovered']
        grouped['source_file'] = df['source_file'].iloc[0]
        grouped['file_date'] = df['file_date'].iloc[0]
        
        # Standardiser les noms de colonnes
        grouped.columns = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active', 'source_file', 'file_date']
        
        # Convertir les valeurs numériques
        for col in ['confirmed', 'deaths', 'recovered', 'active']:
            grouped[col] = pd.to_numeric(grouped[col], errors='coerce').fillna(0)
        
        grouped['data_type'] = 'complete'
        return grouped
    
    def process_demographic_data(self, df, file_date):
        """Traite les données démographiques par âge"""
        if 'country' not in df.columns or 'age_group' not in df.columns:
            return pd.DataFrame()
        
        # Utiliser la date du fichier ou du champ death_reference_date
        if 'death_reference_date' in df.columns:
            df['date'] = pd.to_datetime(df['death_reference_date'], errors='coerce')
        else:
            df['date'] = file_date
        
        # Nettoyer les données démographiques
        df = df.dropna(subset=['country', 'age_group'])
        
        # Filtrer les groupes d'âge valides
        valid_age_groups = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']
        df = df[df['age_group'].isin(valid_age_groups)]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Agréger par pays et date
        numeric_cols = ['cum_death_male', 'cum_death_female', 'cum_death_both']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Grouper par pays et date
        agg_dict = {}
        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = 'sum'
        
        if not agg_dict:
            return pd.DataFrame()
        
        grouped = df.groupby(['country', 'date']).agg(agg_dict).reset_index()
        
        # Calculer les métriques COVID standard
        grouped['deaths'] = grouped.get('cum_death_both', 0)
        grouped['confirmed'] = grouped['deaths'] * 20  # Estimation approximative
        grouped['recovered'] = grouped['confirmed'] * 0.95  # Estimation approximative  
        grouped['active'] = grouped['confirmed'] - grouped['deaths'] - grouped['recovered']
        
        grouped['source_file'] = df['source_file'].iloc[0]
        grouped['file_date'] = df['file_date'].iloc[0]
        grouped['data_type'] = 'demographic'
        
        # Garder seulement les colonnes standardisées
        result_cols = ['country', 'date', 'confirmed', 'deaths', 'recovered', 'active', 'source_file', 'file_date', 'data_type']
        return grouped[result_cols]
    
    def prepare_training_data(self, df):
        """Prépare les données pour l'entraînement"""
        logger.info("🔧 Préparation des données d'entraînement...")
        
        # Filtrer et nettoyer
        df = df.dropna(subset=['country', 'date'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Trier par pays et date
        df = df.sort_values(['country', 'date'])
        
        countries = df['country'].unique()
        sequences = []
        demographics = []
        targets = []
        
        for country in countries:
            country_data = df[df['country'] == country].copy()
            
            if len(country_data) < self.sequence_length + 1:
                logger.warning(f"⚠️  Pas assez de données pour {country} ({len(country_data)} points)")
                continue
            
            # Trier par date
            country_data = country_data.sort_values('date')
            
            # Features temporelles pour LSTM
            time_features = country_data[['confirmed', 'deaths', 'recovered', 'active']].values.astype(np.float32)
            
            # Features démographiques (moyennes par pays)
            demo_features = np.array([
                country_data['confirmed'].mean(),
                country_data['deaths'].mean(),
                country_data['recovered'].mean(), 
                country_data['active'].mean(),
                len(country_data),  # Nombre de points de données
                country_data['deaths'].std() if len(country_data) > 1 else 0  # Variabilité
            ], dtype=np.float32)
            
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
        
        logger.info(f"📊 {len(sequences)} séquences créées")
        logger.info(f"📏 Forme séquences: {sequences.shape}")
        logger.info(f"📏 Forme démographiques: {demographics.shape}")
        logger.info(f"📏 Forme targets: {targets.shape}")
        
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
        """Crée les DataLoaders"""
        # Split train/validation
        X_seq_train, X_seq_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
            sequences, demographics, targets, test_size=0.2, random_state=42
        )
        
        train_dataset = CovidDataset(X_seq_train, X_demo_train, y_train)
        val_dataset = CovidDataset(X_seq_val, X_demo_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"📚 Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        """Entraîne le modèle"""
        logger.info("🏋️ Début de l'entraînement...")
        
        model = CovidLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Training
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
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
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 20:
                logger.info(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if patience_counter >= 20:
                logger.info("🛑 Early stopping")
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
        
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'covid_lstm_model.pth'))
        joblib.dump(self.time_scaler, os.path.join(MODEL_DIR, 'time_scaler.pkl'))
        joblib.dump(self.demo_scaler, os.path.join(MODEL_DIR, 'demo_scaler.pkl'))
        
        import json
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
            metrics_json = {}
            for feature, metric_dict in metrics.items():
                metrics_json[feature] = {k: float(v) for k, v in metric_dict.items()}
            json.dump(metrics_json, f, indent=2)
        
        config = {
            'sequence_length': self.sequence_length,
            'model_architecture': 'LSTM Hybride COVID - Lecture CSV',
            'input_features': ['confirmed', 'deaths', 'recovered', 'active'],
            'demographic_features': 6,
            'training_date': datetime.now().isoformat(),
            'device': str(self.device),
            'data_source': 'CSV files direct'
        }
        
        with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Modèle sauvegardé dans {MODEL_DIR}/")
    
    def run_training(self):
        """Lance l'entraînement complet"""
        logger.info("🚀 Début de l'entraînement COVID LSTM (lecture CSV)")
        
        try:
            # Charger les données CSV
            df = self.load_csv_files()
            
            # Préparer les données
            sequences, demographics, targets = self.prepare_training_data(df)
            
            # Créer les DataLoaders
            train_loader, val_loader = self.create_dataloaders(sequences, demographics, targets)
            
            # Entraîner
            model, train_losses, val_losses = self.train_model(train_loader, val_loader)
            
            # Évaluer
            metrics, predictions, targets_eval = self.evaluate_model(model, val_loader)
            
            # Sauvegarder
            self.save_model_artifacts(model, metrics)
            
            logger.info("🎉 Entraînement terminé avec succès!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    trainer = CovidCSVTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Modèle entraîné avec succès à partir des fichiers CSV!")
        print("📁 Fichiers créés:")
        print("   - models/covid_lstm_model.pth")
        print("   - models/time_scaler.pkl") 
        print("   - models/demo_scaler.pkl")
        print("   - models/metrics.json")
        print("   - models/config.json")
        print("\n🤖 Le modèle peut maintenant prédire l'évolution COVID!")
    else:
        print("❌ Échec de l'entraînement")
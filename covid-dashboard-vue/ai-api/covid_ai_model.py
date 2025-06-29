import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
import joblib
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidTransformerDataset(Dataset):
    """Dataset optimisé pour le modèle Transformer"""
    
    def __init__(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.static_features = torch.FloatTensor(static_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.static_features[idx], self.targets[idx]

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Attention multi-tête optimisée"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Transformation linéaire et reshape
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calcul de l'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Application de l'attention
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(out), attention

class TransformerBlock(nn.Module):
    """Bloc Transformer complet"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attention_weights

class CovidRevolutionaryTransformer(nn.Module):
    """
    🚀 MODÈLE RÉVOLUTIONNAIRE OPTIMISÉ: Transformer Hybride pour COVID
    
    Architecture innovante combinant:
    - Transformer pour capturer les dépendances long-terme
    - LSTM pour la temporalité séquentielle 
    - Features statiques (démographie, vaccination)
    - Multi-output SIMULTANÉ (cas, morts, guérisons) pour TOUS les horizons
    - ⚡ OPTIMISÉ: Un seul forward pass pour tous les horizons
    """
    
    def __init__(self, 
                 sequence_features: int = 15,  # Features temporelles COVID
                 static_features: int = 25,    # Features démographiques + vaccination
                 d_model: int = 256,           # Dimension du modèle
                 n_heads: int = 8,             # Têtes d'attention
                 n_layers: int = 6,            # Couches Transformer
                 d_ff: int = 1024,             # Dimension feed-forward
                 dropout: float = 0.1,
                 prediction_horizons: List[int] = [1, 7, 14, 30]):  # Horizons de prédiction
        
        super().__init__()
        
        self.d_model = d_model
        self.prediction_horizons = prediction_horizons
        self.n_outputs = 3  # confirmed, deaths, recovered, active
        self.n_horizons = len(prediction_horizons)
        
        # 1. Embedding des séquences temporelles
        self.sequence_embedding = nn.Linear(sequence_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 3. LSTM pour capture séquentielle additionnelle
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model//2, 
            num_layers=2, 
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Traitement des features statiques
        self.static_processor = nn.Sequential(
            nn.Linear(static_features, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 5. Fusion et prédiction
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + d_model//4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 🚀 6. OPTIMISATION: Tête unique pour TOUS les horizons simultanément
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, self.n_outputs * self.n_horizons),  # Tous les horizons d'un coup
            nn.ReLU()  # Garantir des prédictions positives
        )
        
        # 7. Couche d'incertitude (pour intervalles de confiance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, self.n_outputs * self.n_horizons),  # Tous les horizons d'un coup
            nn.Softplus()  # Garantir des valeurs positives pour l'écart-type
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"🧠 Modèle révolutionnaire OPTIMISÉ créé:")
        logger.info(f"   - Dimension: {d_model}")
        logger.info(f"   - Couches Transformer: {n_layers}")
        logger.info(f"   - Têtes d'attention: {n_heads}")
        logger.info(f"   - Horizons de prédiction: {prediction_horizons}")
        logger.info(f"   - 🚀 OPTIMISATION: Forward unique pour tous les horizons")
        logger.info(f"   - Paramètres total: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sequences, static_features, target_horizon=None):
        batch_size, seq_len, _ = sequences.shape
        
        # 1. Embedding des séquences + encodage positionnel
        embedded_seq = self.sequence_embedding(sequences)
        embedded_seq = self.positional_encoding(embedded_seq.transpose(0, 1)).transpose(0, 1)
        
        # 2. Transformer layers
        transformer_out = embedded_seq
        attention_weights = []
        
        for transformer_layer in self.transformer_layers:
            transformer_out, attn_weights = transformer_layer(transformer_out)
            attention_weights.append(attn_weights)
        
        # 3. LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(transformer_out)
        
        # 4. Extraction de la représentation finale (moyenne simple)
        temporal_representation = lstm_out.mean(dim=1)
        
        # 5. Traitement des features statiques
        static_representation = self.static_processor(static_features)
        
        # 6. Fusion
        fused_representation = torch.cat([temporal_representation, static_representation], dim=-1)
        fused_out = self.fusion_layer(fused_representation)
        
        # 🚀 7. OPTIMISATION: Prédictions pour TOUS les horizons simultanément
        all_predictions = self.prediction_head(fused_out)  # [batch, n_outputs * n_horizons]
        all_uncertainty = self.uncertainty_head(fused_out)  # [batch, n_outputs * n_horizons]
        
        # Reshape pour séparer horizons et outputs
        predictions = all_predictions.view(batch_size, self.n_horizons, self.n_outputs)  # [batch, horizons, outputs]
        uncertainty = all_uncertainty.view(batch_size, self.n_horizons, self.n_outputs)  # [batch, horizons, outputs]
        
        # Si un horizon spécifique est demandé
        if target_horizon is not None and target_horizon != -1:
            horizon_idx = self.prediction_horizons.index(target_horizon)
            return predictions[:, horizon_idx, :], uncertainty[:, horizon_idx, :], attention_weights
        
        # Sinon retourner toutes les prédictions
        return predictions, uncertainty, attention_weights

class CovidRevolutionaryTrainer:
    """Entraîneur révolutionnaire OPTIMISÉ pour le modèle Transformer"""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sequence_scaler = RobustScaler()
        self.static_scaler = StandardScaler()
        self.target_scaler = RobustScaler()
        
        logger.info(f"🚀 Trainer OPTIMISÉ initialisé sur {self.device}")
    
    def prepare_revolutionary_dataset(self, enriched_df: pd.DataFrame, 
                                    sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Préparation révolutionnaire des données"""
        logger.info("🎯 Préparation du dataset révolutionnaire...")
        
        # Features pour séquences temporelles (COVID + temporelles)
        temporal_features = [
            'confirmed', 'deaths', 'recovered', 'active',
            'new_cases', 'new_deaths', 'new_recovered',
            'new_cases_ma7', 'new_deaths_ma7',
            'growth_rate', 'mortality_rate', 'recovery_rate', 'trend_7d',
            'month_sin', 'month_cos'
        ]
        
        # Features statiques (démographiques + vaccination + interaction)
        static_features = [
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
        
        # Cibles à prédire
        target_features = ['confirmed', 'deaths', 'recovered']
        
        # Vérifier la présence des features
        available_temporal = [f for f in temporal_features if f in enriched_df.columns]
        available_static = [f for f in static_features if f in enriched_df.columns]
        
        logger.info(f"📊 Features temporelles: {len(available_temporal)}/{len(temporal_features)}")
        logger.info(f"📊 Features statiques: {len(available_static)}/{len(static_features)}")
        
        countries = enriched_df['country_name'].unique()
        sequences = []
        static_arrays = []
        targets = []
        
        for country in countries:
            country_data = enriched_df[enriched_df['country_name'] == country].sort_values('date')
            
            if len(country_data) < sequence_length + 30:  # Au moins 30 jours pour prédiction
                continue
            
            # Données temporelles
            temporal_data = country_data[available_temporal].values.astype(np.float32)
            
            # Remplir les NaN
            temporal_data = np.nan_to_num(temporal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Données statiques (moyenne par pays car constantes)
            static_data = country_data[available_static].fillna(0).mean().values.astype(np.float32)
            
            # Données cibles
            target_data = country_data[target_features].values.astype(np.float32)
            
            # Créer les séquences
            for i in range(len(temporal_data) - sequence_length - 30):
                # Séquence d'entrée
                seq = temporal_data[i:i + sequence_length]
                
                # Features statiques (constantes pour chaque échantillon)
                static = static_data
                
                # Cibles futures (1, 7, 14, 30 jours)
                future_targets = {}
                for horizon in [1, 7, 14, 30]:
                    if i + sequence_length + horizon - 1 < len(target_data):
                        # Prendre seulement confirmed, deaths, recovered (pas active)
                        target_values = target_data[i + sequence_length + horizon - 1][:3]
                        future_targets[horizon] = target_values

                if len(future_targets) == 4:  # S'assurer qu'on a tous les horizons
                    sequences.append(seq)
                    static_arrays.append(static)
                    targets.append([future_targets[h] for h in [1, 7, 14, 30]])
        
        sequences = np.array(sequences, dtype=np.float32)
        static_arrays = np.array(static_arrays, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        logger.info(f"🎯 Dataset créé:")
        logger.info(f"   - Séquences: {sequences.shape}")
        logger.info(f"   - Features statiques: {static_arrays.shape}")
        logger.info(f"   - Cibles: {targets.shape}")
        
        return sequences, static_arrays, targets
    
    def create_dataloaders(self, sequences: np.ndarray, static_features: np.ndarray, 
                          targets: np.ndarray, batch_size: int = 32, 
                          val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Crée les DataLoaders avec split temporel"""
        
        # Split temporel (plus réaliste pour séries temporelles)
        split_idx = int(len(sequences) * (1 - val_split))
        
        # Normalisation sur les données d'entraînement uniquement
        seq_train = sequences[:split_idx]
        seq_val = sequences[split_idx:]
        
        static_train = static_features[:split_idx]
        static_val = static_features[split_idx:]
        
        targets_train = targets[:split_idx]
        targets_val = targets[split_idx:]
        
        # Normalisation
        seq_train_scaled = self.sequence_scaler.fit_transform(
            seq_train.reshape(-1, seq_train.shape[-1])
        ).reshape(seq_train.shape)
        seq_val_scaled = self.sequence_scaler.transform(
            seq_val.reshape(-1, seq_val.shape[-1])
        ).reshape(seq_val.shape)
        
        static_train_scaled = self.static_scaler.fit_transform(static_train)
        static_val_scaled = self.static_scaler.transform(static_val)
        
        targets_train_scaled = self.target_scaler.fit_transform(
            targets_train.reshape(-1, targets_train.shape[-1])
        ).reshape(targets_train.shape)
        targets_val_scaled = self.target_scaler.transform(
            targets_val.reshape(-1, targets_val.shape[-1])
        ).reshape(targets_val.shape)
        
        # Datasets
        train_dataset = CovidTransformerDataset(seq_train_scaled, static_train_scaled, targets_train_scaled)
        val_dataset = CovidTransformerDataset(seq_val_scaled, static_val_scaled, targets_val_scaled)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"📚 DataLoaders créés:")
        logger.info(f"   - Train: {len(train_dataset)} échantillons")
        logger.info(f"   - Validation: {len(val_dataset)} échantillons")
        
        return train_loader, val_loader
    
    def train_revolutionary_model(self, train_loader: DataLoader, val_loader: DataLoader,
                                epochs: int = 100, learning_rate: float = 1e-4) -> Dict:
        """🚀 Entraînement révolutionnaire OPTIMISÉ du modèle"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT RÉVOLUTIONNAIRE OPTIMISÉ")
        
        # Déterminer les dimensions
        sample_seq, sample_static, sample_targets = next(iter(train_loader))
        sequence_features = sample_seq.shape[-1]
        static_features = sample_static.shape[-1]
        
        # Créer le modèle
        self.model = CovidRevolutionaryTransformer(
            sequence_features=sequence_features,
            static_features=static_features,
            **self.config
        ).to(self.device)
        
        # Optimiseur et scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate*5, epochs=epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.1
        )
        
        # Loss function (combinaison MSE + MAE)
        def combined_loss(pred, target, uncertainty=None):
            """Loss améliorée avec cohérence mathématique"""
            batch_size, n_horizons, n_outputs = pred.shape  # n_outputs = 3 maintenant
            
            # Loss de base
            mse_loss = nn.MSELoss()(pred, target)
            mae_loss = nn.L1Loss()(pred, target)
            
            # Pondération par horizon (moins de poids sur horizons lointains)
            horizon_weights = torch.tensor([1.0, 0.8, 0.6, 0.4], device=pred.device)  # [1j, 7j, 14j, 30j]
            horizon_weights = horizon_weights.view(1, n_horizons, 1).expand(batch_size, n_horizons, n_outputs)
            
            # Loss pondérée
            weighted_mse = torch.mean(horizon_weights * (pred - target) ** 2)
            weighted_mae = torch.mean(horizon_weights * torch.abs(pred - target))
            
            # Contraintes de cohérence (deaths et recovered ne peuvent pas dépasser confirmed)
            confirmed = pred[:, :, 0]  # cas confirmés
            deaths = pred[:, :, 1]     # décès
            recovered = pred[:, :, 2]  # guérisons
            
            # Pénalité si deaths > confirmed ou recovered > confirmed
            deaths_penalty = torch.mean(torch.relu(deaths - confirmed))
            recovered_penalty = torch.mean(torch.relu(recovered - confirmed))
            
            # Pénalité si deaths + recovered > confirmed
            total_penalty = torch.mean(torch.relu(deaths + recovered - confirmed))
            
            # Loss totale
            coherence_penalty = deaths_penalty + recovered_penalty + total_penalty
            total_loss = weighted_mse + weighted_mae + 0.2 * coherence_penalty
            
            # Loss avec incertitude
            if uncertainty is not None:
                # Stabiliser l'incertitude pour les 3 outputs
                uncertainty = torch.clamp(uncertainty, min=1e-6)
                nll_loss = 0.5 * torch.log(2 * np.pi * uncertainty) + \
                        0.5 * ((pred - target) ** 2) / uncertainty
                total_loss += 0.1 * nll_loss.mean()
            
            return total_loss
        
        # Historique d'entraînement
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, static, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                static = static.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 🚀 OPTIMISATION: UN SEUL forward pass pour TOUS les horizons
                all_predictions, all_uncertainty, _ = self.model(sequences, static, target_horizon=-1)
                
                # all_predictions shape: [batch, horizons, outputs] = [batch, 4, 4]
                # targets shape: [batch, horizons, outputs] = [batch, 4, 4]
                
                # Calculer la loss sur tous les horizons simultanément
                total_loss = combined_loss(all_predictions, targets, all_uncertainty)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += total_loss.item()
                
                # Log de progression toutes les 100 batches
                if batch_idx % 100 == 0:
                    logger.info(f"   Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets_list = []
            
            with torch.no_grad():
                for sequences, static, targets in val_loader:
                    sequences = sequences.to(self.device)
                    static = static.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 🚀 OPTIMISATION: UN SEUL forward pass pour validation aussi
                    all_predictions, all_uncertainty, _ = self.model(sequences, static, target_horizon=-1)
                    
                    loss = combined_loss(all_predictions, targets, all_uncertainty)
                    val_loss += loss.item()
                    
                    # Garder les prédictions à 7 jours (index 1) pour les métriques
                    val_predictions.append(all_predictions[:, 1, :].cpu().numpy())  # Horizon 7 jours
                    val_targets_list.append(targets[:, 1, :].cpu().numpy())
            
            # Calcul des métriques
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Métriques détaillées sur validation
            val_metrics = {}
            if val_predictions:
                val_pred_array = np.vstack(val_predictions)
                val_target_array = np.vstack(val_targets_list)
                
                for i, metric_name in enumerate(['confirmed', 'deaths', 'recovered']):
                    mae = mean_absolute_error(val_target_array[:, i], val_pred_array[:, i])
                    mape = mean_absolute_percentage_error(val_target_array[:, i], val_pred_array[:, i])
                    r2 = r2_score(val_target_array[:, i], val_pred_array[:, i])
                    val_metrics[f'{metric_name}_mae'] = mae
                    val_metrics[f'{metric_name}_mape'] = mape
                    val_metrics[f'{metric_name}_r2'] = r2
            
            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/covid_revolutionary_model.pth')
            else:
                patience_counter += 1
            
            # Historique
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Log des métriques
            logger.info(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            if val_metrics:
                logger.info(f"   Val R²: confirmed={val_metrics.get('confirmed_r2', 0):.3f}, deaths={val_metrics.get('deaths_r2', 0):.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping à l'epoch {epoch}")
                break
        
        # Charger le meilleur modèle
        self.model.load_state_dict(torch.load('models/covid_revolutionary_model.pth'))
        
        logger.info("🎉 ENTRAÎNEMENT OPTIMISÉ TERMINÉ AVEC SUCCÈS!")
        return history
    
    def save_artifacts(self, history: Dict):
        """Sauvegarde tous les artefacts"""
        logger.info("💾 Sauvegarde des artefacts...")
        
        os.makedirs('models', exist_ok=True)
        
        # Scalers
        joblib.dump(self.sequence_scaler, 'models/revolutionary_sequence_scaler.pkl')
        joblib.dump(self.static_scaler, 'models/revolutionary_static_scaler.pkl')
        joblib.dump(self.target_scaler, 'models/revolutionary_target_scaler.pkl')
        
        # Configuration
        config = {
            'model_config': self.config,
            'training_history': history,
            'model_type': 'CovidRevolutionaryTransformer_OPTIMIZED',
            'prediction_horizons': [1, 7, 14, 30],
            'features': {
                'sequence_features': self.model.sequence_embedding.in_features,
                'static_features': self.model.static_processor[0].in_features,
                'output_features': self.model.n_outputs
            },
            'training_date': datetime.now().isoformat(),
            'device': str(self.device),
            'optimization': 'Single forward pass for all horizons'
        }
        
        with open('models/revolutionary_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Visualisations
        self.plot_training_history(history)
        
        logger.info("✅ Tous les artefacts sauvegardés!")
    
    def plot_training_history(self, history: Dict):
        """Visualise l'historique d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0,0].plot(history['train_loss'], label='Train Loss')
        axes[0,0].plot(history['val_loss'], label='Validation Loss')
        axes[0,0].set_title('Loss Evolution (OPTIMIZED)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # R² scores
        if history['val_metrics']:
            metrics_df = pd.DataFrame(history['val_metrics'])
            for col in ['confirmed_r2', 'deaths_r2', 'recovered_r2', 'active_r2']:
                if col in metrics_df.columns:
                    axes[0,1].plot(metrics_df[col], label=col.replace('_r2', ''))
            axes[0,1].set_title('R² Scores')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # MAE
            for col in ['confirmed_mae', 'deaths_mae', 'recovered_mae', 'active_mae']:
                if col in metrics_df.columns:
                    axes[1,0].plot(metrics_df[col], label=col.replace('_mae', ''))
            axes[1,0].set_title('Mean Absolute Error')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # MAPE
            for col in ['confirmed_mape', 'deaths_mape', 'recovered_mape', 'active_mape']:
                if col in metrics_df.columns:
                    axes[1,1].plot(metrics_df[col], label=col.replace('_mape', ''))
            axes[1,1].set_title('Mean Absolute Percentage Error')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Configuration du modèle révolutionnaire
    model_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'prediction_horizons': [1, 7, 14, 30]
    }
    
    # Entraînement
    trainer = CovidRevolutionaryTrainer(model_config)
    
    # Charger les données enrichies
    enriched_data = pd.read_csv('../data/dataset_clean/enriched_covid_dataset.csv')
    
    # Préparer les données
    sequences, static_features, targets = trainer.prepare_revolutionary_dataset(enriched_data)
    
    # Créer les DataLoaders
    train_loader, val_loader = trainer.create_dataloaders(sequences, static_features, targets)
    
    # Entraîner le modèle
    history = trainer.train_revolutionary_model(train_loader, val_loader, epochs=100)
    
    # Sauvegarder
    trainer.save_artifacts(history)
    
    print("\n🎉 MODÈLE RÉVOLUTIONNAIRE OPTIMISÉ ENTRAÎNÉ ET SAUVEGARDÉ!")
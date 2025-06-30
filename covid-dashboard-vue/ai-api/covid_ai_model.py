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

class CovidLongTermDataset(Dataset):
    """Dataset optimisé pour prédictions LONG TERME"""
    
    def __init__(self, sequences: np.ndarray, static_features: np.ndarray, 
                 targets: np.ndarray, horizons: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.static_features = torch.FloatTensor(static_features)
        self.targets = torch.FloatTensor(targets)
        self.horizons = torch.LongTensor(horizons)  # Encodage des horizons
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.static_features[idx], self.targets[idx], self.horizons[idx]

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
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
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
        attn_out, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attention_weights

class CovidRevolutionaryLongTermTransformer(nn.Module):
    """
    🚀 MODÈLE RÉVOLUTIONNAIRE LONG TERME v2.1
    
    NOUVEAUTÉS RÉVOLUTIONNAIRES :
    - Prédictions multi-temporelles : 1j → 5 ans
    - Logique conditionnelle vaccination intégrée
    - Impact démographique intelligent
    - Horizons adaptatifs selon contexte
    """
    
    def __init__(self, 
                 sequence_features: int = 15,
                 static_features: int = 35,  # Plus de features démographiques + vaccination
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 prediction_horizons: List[int] = [1, 7, 14, 30, 90, 180, 365, 730, 1825]):
        
        super().__init__()
        
        self.d_model = d_model
        self.prediction_horizons = prediction_horizons
        self.n_outputs = 3  # confirmed, deaths, recovered
        self.n_horizons = len(prediction_horizons)
        
        # 1. Embedding des séquences temporelles
        self.sequence_embedding = nn.Linear(sequence_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 3. LSTM pour séquences
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model//2, 
            num_layers=2, 
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Processeur features statiques ÉTENDU
        self.static_processor = nn.Sequential(
            nn.Linear(static_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 🚀 5. ENCODEUR HORIZON (nouveau!)
        self.horizon_embedding = nn.Embedding(len(prediction_horizons), d_model//4)
        
        # 6. Couche de fusion ÉTENDUE
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + d_model//2 + d_model//4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 🧠 7. TÊTES SPÉCIALISÉES PAR HORIZON
        self.short_term_head = nn.Sequential(  # 1j, 7j, 14j, 30j
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, self.n_outputs),
        )
        
        self.medium_term_head = nn.Sequential(  # 90j, 180j
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, self.n_outputs),
        )
        
        self.long_term_head = nn.Sequential(  # 365j, 730j, 1825j
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, self.n_outputs),
        )
        
        # 8. Têtes d'incertitude par horizon
        self.short_uncertainty = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, self.n_outputs),
            nn.Softplus()
        )
        
        self.medium_uncertainty = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, self.n_outputs),
            nn.Softplus()
        )
        
        self.long_uncertainty = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, self.n_outputs),
            nn.Softplus()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"🚀 Modèle RÉVOLUTIONNAIRE LONG TERME créé:")
        logger.info(f"   - Horizons: {prediction_horizons}")
        logger.info(f"   - Court terme: {prediction_horizons[:4]}")
        logger.info(f"   - Moyen terme: {prediction_horizons[4:6]}")
        logger.info(f"   - Long terme: {prediction_horizons[6:]}")
        logger.info(f"   - Paramètres: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sequences, static_features, target_horizon=None):
        batch_size, seq_len, _ = sequences.shape
        
        # 1. Embedding séquences + encodage positionnel
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
        temporal_representation = lstm_out.mean(dim=1)
        
        # 4. Features statiques
        static_representation = self.static_processor(static_features)
        
        # 🚀 5. LOGIQUE PAR HORIZON
        if target_horizon is not None and target_horizon in self.prediction_horizons:
            horizon_idx = self.prediction_horizons.index(target_horizon)
            horizon_embed = self.horizon_embedding(torch.tensor([horizon_idx], device=sequences.device))
            horizon_embed = horizon_embed.expand(batch_size, -1)
            
            # Fusion avec encodage horizon
            fused_representation = torch.cat([temporal_representation, static_representation, horizon_embed], dim=-1)
            fused_out = self.fusion_layer(fused_representation)
            
            # Sélection tête selon horizon
            if target_horizon <= 30:  # Court terme
                predictions = self.short_term_head(fused_out)
                uncertainty = self.short_uncertainty(fused_out)
            elif target_horizon <= 180:  # Moyen terme
                predictions = self.medium_term_head(fused_out)
                uncertainty = self.medium_uncertainty(fused_out)
            else:  # Long terme
                predictions = self.long_term_head(fused_out)
                uncertainty = self.long_uncertainty(fused_out)
            
            return predictions, uncertainty, attention_weights
        
        else:
            # 🚀 TOUS LES HORIZONS (mode entraînement)
            all_predictions = []
            all_uncertainty = []
            
            for i, horizon in enumerate(self.prediction_horizons):
                horizon_embed = self.horizon_embedding(torch.tensor([i], device=sequences.device))
                horizon_embed = horizon_embed.expand(batch_size, -1)
                
                fused_representation = torch.cat([temporal_representation, static_representation, horizon_embed], dim=-1)
                fused_out = self.fusion_layer(fused_representation)
                
                # Sélection tête
                if horizon <= 30:
                    pred = self.short_term_head(fused_out)
                    unc = self.short_uncertainty(fused_out)
                elif horizon <= 180:
                    pred = self.medium_term_head(fused_out)
                    unc = self.medium_uncertainty(fused_out)
                else:
                    pred = self.long_term_head(fused_out)
                    unc = self.long_uncertainty(fused_out)
                
                all_predictions.append(pred)
                all_uncertainty.append(unc)
            
            # Stack toutes les prédictions
            predictions = torch.stack(all_predictions, dim=1)  # [batch, horizons, outputs]
            uncertainty = torch.stack(all_uncertainty, dim=1)  # [batch, horizons, outputs]
            
            return predictions, uncertainty, attention_weights

class CovidRevolutionaryLongTermTrainer:
    """Entraîneur révolutionnaire pour modèle LONG TERME"""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sequence_scaler = RobustScaler()
        self.static_scaler = StandardScaler()
        self.target_scaler = RobustScaler()
        
        # 🚀 HORIZONS RÉVOLUTIONNAIRES
        self.prediction_horizons = [1, 7, 14, 30, 90, 180, 365, 730, 1825]
        
        logger.info(f"🚀 Trainer LONG TERME initialisé sur {self.device}")
        logger.info(f"📅 Horizons: {self.prediction_horizons}")
    
    def prepare_longterm_dataset(self, enriched_df: pd.DataFrame, 
                                sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Préparation dataset RÉVOLUTIONNAIRE LONG TERME"""
        logger.info("🎯 Préparation dataset LONG TERME...")
        
        # Features temporelles COVID
        temporal_features = [
            'confirmed', 'deaths', 'recovered', 'active',
            'new_cases', 'new_deaths', 'new_recovered',
            'new_cases_ma7', 'new_deaths_ma7',
            'growth_rate', 'mortality_rate', 'recovery_rate', 'trend_7d',
            'month_sin', 'month_cos'
        ]
        
        # 🚀 FEATURES STATIQUES ÉTENDUES
        static_features = [
            # Démographiques
            'population_millions', 'birth_rate', 'mortality_rate', 'life_expectancy',
            'infant_mortality_rate', 'fertility_rate', 'growth_rate', 'elderly_ratio',
            'covid_vulnerability', 'demographic_resilience', 'age_mortality_factor',
            
            # Vaccination
            'has_vaccination', 'coverage_percent', 'protection_factor',
            'case_reduction_factor', 'mortality_reduction_factor', 'vaccination_momentum',
            
            # Temporelles étendues
            'pandemic_year', 'pandemic_phase', 'seasonal_factor',
            'day_of_year', 'quarter', 'week_of_year', 'weekday', 'is_weekend',
            
            # Interactions
            'demographic_covid_severity', 'country_resilience_score',
            'vaccination_effectiveness_adjusted', 'predicted_mortality_factor', 'epidemic_phase'
        ]
        
        target_features = ['confirmed', 'deaths', 'recovered']
        
        # Vérifier features disponibles
        available_temporal = [f for f in temporal_features if f in enriched_df.columns]
        available_static = [f for f in static_features if f in enriched_df.columns]
        
        logger.info(f"📊 Features temporelles: {len(available_temporal)}")
        logger.info(f"📊 Features statiques: {len(available_static)}")
        
        countries = enriched_df['country_name'].unique()
        sequences = []
        static_arrays = []
        targets = []
        horizons = []
        
        max_horizon = max(self.prediction_horizons)
        
        for country in countries:
            country_data = enriched_df[enriched_df['country_name'] == country].sort_values('date')
            
            if len(country_data) < sequence_length + max_horizon:
                continue
            
            # Données temporelles
            temporal_data = country_data[available_temporal].values.astype(np.float32)
            temporal_data = np.nan_to_num(temporal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Données statiques
            static_data = country_data[available_static].fillna(0).mean().values.astype(np.float32)
            
            # Données cibles
            target_data = country_data[target_features].values.astype(np.float32)
            
            # 🚀 CRÉER ÉCHANTILLONS POUR TOUS LES HORIZONS
            for i in range(len(temporal_data) - sequence_length - max_horizon):
                # Séquence d'entrée
                seq = temporal_data[i:i + sequence_length]
                static = static_data
                
                # 🎯 CIBLES POUR TOUS LES HORIZONS
                horizon_targets = []
                valid_horizons = []
                
                for horizon in self.prediction_horizons:
                    target_idx = i + sequence_length + horizon - 1
                    if target_idx < len(target_data):
                        target_values = target_data[target_idx][:3]  # confirmed, deaths, recovered
                        horizon_targets.append(target_values)
                        valid_horizons.append(horizon)
                
                # Ne garder que si on a au moins les horizons courts
                if len(valid_horizons) >= 4:  # Au moins 1j, 7j, 14j, 30j
                    # Créer un échantillon par horizon disponible
                    for j, horizon in enumerate(valid_horizons):
                        sequences.append(seq)
                        static_arrays.append(static)
                        targets.append(horizon_targets[j])
                        horizons.append(self.prediction_horizons.index(horizon))
        
        sequences = np.array(sequences, dtype=np.float32)
        static_arrays = np.array(static_arrays, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        horizons = np.array(horizons, dtype=np.int64)
        
        logger.info(f"🎯 Dataset LONG TERME créé:")
        logger.info(f"   - Séquences: {sequences.shape}")
        logger.info(f"   - Features statiques: {static_arrays.shape}")
        logger.info(f"   - Cibles: {targets.shape}")
        logger.info(f"   - Horizons: {horizons.shape}")
        
        return sequences, static_arrays, targets, horizons
    
    def create_longterm_dataloaders(self, sequences: np.ndarray, static_features: np.ndarray, 
                                   targets: np.ndarray, horizons: np.ndarray,
                                   batch_size: int = 32, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Crée DataLoaders pour LONG TERME"""
        
        # Split temporel
        split_idx = int(len(sequences) * (1 - val_split))
        
        # Normalisation
        seq_train_scaled = self.sequence_scaler.fit_transform(
            sequences[:split_idx].reshape(-1, sequences.shape[-1])
        ).reshape(sequences[:split_idx].shape)
        seq_val_scaled = self.sequence_scaler.transform(
            sequences[split_idx:].reshape(-1, sequences.shape[-1])
        ).reshape(sequences[split_idx:].shape)
        
        static_train_scaled = self.static_scaler.fit_transform(static_features[:split_idx])
        static_val_scaled = self.static_scaler.transform(static_features[split_idx:])
        
        targets_train_scaled = self.target_scaler.fit_transform(targets[:split_idx])
        targets_val_scaled = self.target_scaler.transform(targets[split_idx:])
        
        # Datasets
        train_dataset = CovidLongTermDataset(
            seq_train_scaled, static_train_scaled, targets_train_scaled, horizons[:split_idx]
        )
        val_dataset = CovidLongTermDataset(
            seq_val_scaled, static_val_scaled, targets_val_scaled, horizons[split_idx:]
        )
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"📚 DataLoaders LONG TERME créés:")
        logger.info(f"   - Train: {len(train_dataset)} échantillons")
        logger.info(f"   - Validation: {len(val_dataset)} échantillons")
        
        return train_loader, val_loader
    
    def train_longterm_model(self, train_loader: DataLoader, val_loader: DataLoader,
                            epochs: int = 100, learning_rate: float = 1e-4) -> Dict:
        """🚀 Entraînement RÉVOLUTIONNAIRE LONG TERME"""
        logger.info("🚀 DÉMARRAGE ENTRAÎNEMENT LONG TERME RÉVOLUTIONNAIRE")
        
        # Créer le modèle
        sample_seq, sample_static, sample_targets, sample_horizons = next(iter(train_loader))
        sequence_features = sample_seq.shape[-1]
        static_features = sample_static.shape[-1]
        
        self.model = CovidRevolutionaryLongTermTransformer(
            sequence_features=sequence_features,
            static_features=static_features,
            prediction_horizons=self.prediction_horizons,
            **self.config
        ).to(self.device)
        
        # Optimiseur et scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate*3, epochs=epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.1
        )
        
        # 🧠 LOSS FUNCTION RÉVOLUTIONNAIRE MULTI-HORIZON
        def revolutionary_loss(pred, target, horizon_idx, uncertainty=None):
            """Loss adaptée aux différents horizons"""
            
            # Loss de base
            mse_loss = nn.MSELoss()(pred, target)
            mae_loss = nn.L1Loss()(pred, target)
            
            # 🎯 PONDÉRATION SELON HORIZON
            horizon_weights = {
                0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7,  # Court terme (1j, 7j, 14j, 30j)
                4: 0.6, 5: 0.5,                    # Moyen terme (90j, 180j)
                6: 0.4, 7: 0.3, 8: 0.2             # Long terme (1an, 2ans, 5ans)
            }
            
            # Poids selon horizon
            weight = torch.tensor([horizon_weights.get(h.item(), 0.5) for h in horizon_idx], device=pred.device)
            weight = weight.view(-1, 1)
            
            # Loss pondérée
            weighted_mse = torch.mean(weight * (pred - target) ** 2)
            weighted_mae = torch.mean(weight * torch.abs(pred - target))
            
            # 🧠 CONTRAINTES COHÉRENCE (plus importantes pour long terme)
            confirmed = pred[:, 0]
            deaths = pred[:, 1] 
            recovered = pred[:, 2]
            
            coherence_penalty = (
                torch.mean(torch.relu(deaths - confirmed)) +
                torch.mean(torch.relu(recovered - confirmed)) +
                torch.mean(torch.relu(deaths + recovered - confirmed))
            )
            
            # Plus de pénalité pour horizons longs
            long_term_mask = (horizon_idx >= 6).float()  # 1 an et plus
            coherence_weight = 0.1 + 0.3 * long_term_mask.mean()
            
            total_loss = weighted_mse + weighted_mae + coherence_weight * coherence_penalty
            
            # Loss incertitude
            if uncertainty is not None:
                uncertainty = torch.clamp(uncertainty, min=1e-6)
                nll_loss = 0.5 * torch.log(2 * np.pi * uncertainty) + \
                          0.5 * ((pred - target) ** 2) / uncertainty
                total_loss += 0.1 * nll_loss.mean()
            
            return total_loss
        
        # Historique
        history = {
            'train_loss': [],
            'val_loss': [],
            'horizon_metrics': {h: [] for h in self.prediction_horizons}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, static, targets, horizon_indices) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                static = static.to(self.device)
                targets = targets.to(self.device)
                horizon_indices = horizon_indices.to(self.device)
                
                optimizer.zero_grad()
                
                # 🚀 PRÉDICTION PAR HORIZON SPÉCIFIQUE
                batch_predictions = []
                batch_uncertainties = []
                
                # Grouper par horizon pour efficacité
                unique_horizons = torch.unique(horizon_indices)
                
                for horizon_idx in unique_horizons:
                    mask = horizon_indices == horizon_idx
                    if mask.sum() > 0:
                        horizon_seq = sequences[mask]
                        horizon_static = static[mask]
                        horizon_targets = targets[mask]
                        
                        # Prédiction pour cet horizon
                        horizon_value = self.prediction_horizons[horizon_idx.item()]
                        pred, unc, _ = self.model(horizon_seq, horizon_static, target_horizon=horizon_value)
                        
                        # Calculer loss pour cet horizon
                        horizon_loss = revolutionary_loss(pred, horizon_targets, 
                                                        horizon_indices[mask], unc)
                        
                        # Backpropagation
                        if batch_idx == 0 or horizon_idx == unique_horizons[0]:
                            total_loss = horizon_loss
                        else:
                            total_loss += horizon_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += total_loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            horizon_predictions = {h: {'pred': [], 'target': []} for h in self.prediction_horizons}
            
            with torch.no_grad():
                for sequences, static, targets, horizon_indices in val_loader:
                    sequences = sequences.to(self.device)
                    static = static.to(self.device)
                    targets = targets.to(self.device)
                    horizon_indices = horizon_indices.to(self.device)
                    
                    # Validation par horizon
                    unique_horizons = torch.unique(horizon_indices)
                    
                    for horizon_idx in unique_horizons:
                        mask = horizon_indices == horizon_idx
                        if mask.sum() > 0:
                            horizon_seq = sequences[mask]
                            horizon_static = static[mask]
                            horizon_targets = targets[mask]
                            
                            horizon_value = self.prediction_horizons[horizon_idx.item()]
                            pred, unc, _ = self.model(horizon_seq, horizon_static, target_horizon=horizon_value)
                            
                            loss = revolutionary_loss(pred, horizon_targets, horizon_indices[mask], unc)
                            val_loss += loss.item()
                            
                            # Stocker pour métriques
                            horizon_predictions[horizon_value]['pred'].append(pred.cpu().numpy())
                            horizon_predictions[horizon_value]['target'].append(horizon_targets.cpu().numpy())
            
            # Calcul métriques par horizon
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Métriques détaillées
            for horizon in self.prediction_horizons:
                if horizon_predictions[horizon]['pred']:
                    pred_array = np.vstack(horizon_predictions[horizon]['pred'])
                    target_array = np.vstack(horizon_predictions[horizon]['target'])
                    
                    # R² pour confirmed (métrique principale)
                    r2_confirmed = r2_score(target_array[:, 0], pred_array[:, 0])
                    history['horizon_metrics'][horizon].append(r2_confirmed)
            
            # Sauvegarde meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/covid_revolutionary_longterm_model.pth')
            else:
                patience_counter += 1
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Log
            logger.info(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            # Log métriques par horizon (échantillon)
            sample_horizons = [1, 30, 365, 1825]  # Court/moyen/long terme
            metrics_log = []
            for h in sample_horizons:
                if h in self.prediction_horizons and history['horizon_metrics'][h]:
                    r2 = history['horizon_metrics'][h][-1]
                    metrics_log.append(f"{h}j: R²={r2:.3f}")
            logger.info(f"   Métriques: {' | '.join(metrics_log)}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping à l'epoch {epoch}")
                break
        
        # Charger meilleur modèle
        self.model.load_state_dict(torch.load('models/covid_revolutionary_longterm_model.pth'))
        
        logger.info("🎉 ENTRAÎNEMENT LONG TERME TERMINÉ AVEC SUCCÈS!")
        return history
    
    def save_longterm_artifacts(self, history: Dict):
        """Sauvegarde artefacts LONG TERME"""
        logger.info("💾 Sauvegarde artefacts LONG TERME...")
        
        os.makedirs('models', exist_ok=True)
        
        # Scalers
        joblib.dump(self.sequence_scaler, 'models/revolutionary_longterm_sequence_scaler.pkl')
        joblib.dump(self.static_scaler, 'models/revolutionary_longterm_static_scaler.pkl')
        joblib.dump(self.target_scaler, 'models/revolutionary_longterm_target_scaler.pkl')
        
        # Configuration complète
        config = {
            'model_config': self.config,
            'training_history': history,
            'model_type': 'CovidRevolutionaryLongTermTransformer_v2.1',
            'prediction_horizons': self.prediction_horizons,
            'horizon_categories': {
                'short_term': [1, 7, 14, 30],      # Court terme
                'medium_term': [90, 180],          # Moyen terme  
                'long_term': [365, 730, 1825]     # Long terme
            },
            'features': {
                'sequence_features': self.model.sequence_embedding.in_features,
                'static_features': self.model.static_processor[0].in_features,
                'output_features': self.model.n_outputs
            },
            'training_date': datetime.now().isoformat(),
            'device': str(self.device),
            'optimization': 'Multi-horizon adaptive predictions'
        }
        
        with open('models/revolutionary_longterm_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Visualisations par horizon
        self.plot_longterm_training_history(history)
        
        logger.info("✅ Artefacts LONG TERME sauvegardés!")
    
    def plot_longterm_training_history(self, history: Dict):
        """Visualise historique MULTI-HORIZON"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Loss globale
        axes[0,0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0,0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0,0].set_title('Loss Evolution (Multi-Horizon)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Métriques par catégorie d'horizon
        short_term = [1, 7, 14, 30]
        medium_term = [90, 180]
        long_term = [365, 730, 1825]
        
        # Court terme
        for horizon in short_term:
            if horizon in history['horizon_metrics'] and history['horizon_metrics'][horizon]:
                axes[0,1].plot(history['horizon_metrics'][horizon], 
                             label=f'{horizon}j', alpha=0.8)
        axes[0,1].set_title('Performance Court Terme (R²)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Moyen terme
        for horizon in medium_term:
            if horizon in history['horizon_metrics'] and history['horizon_metrics'][horizon]:
                axes[1,0].plot(history['horizon_metrics'][horizon], 
                             label=f'{horizon}j', alpha=0.8)
        axes[1,0].set_title('Performance Moyen Terme (R²)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Long terme
        for horizon in long_term:
            if horizon in history['horizon_metrics'] and history['horizon_metrics'][horizon]:
                axes[1,1].plot(history['horizon_metrics'][horizon], 
                             label=f'{horizon//365}an{"s" if horizon//365>1 else ""}', alpha=0.8)
        axes[1,1].set_title('Performance Long Terme (R²)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history_multihorizon.png', dpi=300, bbox_inches='tight')
        plt.close()

# Classe pour compatibilité avec l'ancien système
class CovidRevolutionaryTrainer(CovidRevolutionaryLongTermTrainer):
    """🔄 WRAPPER pour compatibilité - Délègue vers le nouveau système MULTI-HORIZON"""
    
    def __init__(self, model_config: Dict):
        super().__init__(model_config)
        logger.info("🔄 Mode compatibilité activé - Utilise le système MULTI-HORIZON")
    
    def prepare_revolutionary_dataset(self, enriched_df: pd.DataFrame, 
                                    sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrapper pour compatibilité avec l'ancienne API"""
        sequences, static_features, targets, horizons = self.prepare_longterm_dataset(enriched_df, sequence_length)
        
        # Pour compatibilité, on retourne seulement les données pour horizons courts
        short_mask = horizons < 4  # Indices 0,1,2,3 = horizons 1,7,14,30
        
        return (sequences[short_mask], 
                static_features[short_mask], 
                targets[short_mask])
    
    def create_dataloaders(self, sequences: np.ndarray, static_features: np.ndarray, 
                          targets: np.ndarray, batch_size: int = 32, 
                          val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Wrapper pour compatibilité"""
        # Créer des horizons factices pour les données courtes
        horizons = np.random.choice([0, 1, 2, 3], size=len(sequences))  # Horizons courts aléatoires
        
        return self.create_longterm_dataloaders(sequences, static_features, targets, horizons, batch_size, val_split)
    
    def train_revolutionary_model(self, train_loader: DataLoader, val_loader: DataLoader,
                                epochs: int = 100, learning_rate: float = 1e-4) -> Dict:
        """Wrapper pour compatibilité"""
        return self.train_longterm_model(train_loader, val_loader, epochs, learning_rate)
    
    def save_artifacts(self, history: Dict):
        """Wrapper pour compatibilité"""
        self.save_longterm_artifacts(history)

if __name__ == "__main__":
    # Test avec configuration MULTI-HORIZON
    model_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1
    }
    
    # Test du nouveau système
    trainer = CovidRevolutionaryLongTermTrainer(model_config)
    
    print("\n🎯 SYSTÈME RÉVOLUTIONNAIRE MULTI-HORIZON PRÊT!")
    print("📅 Horizons supportés:")
    print("   📈 Court terme: 1j, 7j, 14j, 30j")
    print("   📊 Moyen terme: 90j (3 mois), 180j (6 mois)")
    print("   🚀 Long terme: 365j (1 an), 730j (2 ans), 1825j (5 ans)")
    print("\n💡 Le modèle s'adapte automatiquement selon l'horizon demandé!")
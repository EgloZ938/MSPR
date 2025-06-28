#!/usr/bin/env python3
"""
🚨 DIAGNOSTIC URGENT - DEBUGGING TRAINER
"""
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test de chargement des données"""
    print("🔍 TEST 1: Chargement des données")
    start = time.time()
    
    try:
        # Test pipeline CSV
        sys.path.append('.')
        from covid_data_pipeline import CSVCovidDataPipeline
        
        pipeline = CSVCovidDataPipeline('../data/dataset_clean')
        
        # Test vaccination
        print("   💉 Test vaccination...")
        vax_df = pipeline.load_vaccination_data()
        print(f"   ✅ Vaccination: {len(vax_df)} lignes")
        
        # Test démographie  
        print("   👥 Test démographie...")
        demo_df = pipeline.load_demographics_data()
        print(f"   ✅ Démographie: {len(demo_df)} lignes")
        
        # Test COVID
        print("   🦠 Test COVID...")
        covid_df = pipeline.load_covid_timeseries_from_csv()
        print(f"   ✅ COVID: {len(covid_df)} lignes")
        
        elapsed = time.time() - start
        print(f"   ⏱️ Temps: {elapsed:.2f}s")
        
        if elapsed > 30:
            print("   🚨 PROBLÈME: Chargement trop lent!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_model_creation():
    """Test de création du modèle"""
    print("\n🔍 TEST 2: Création du modèle")
    start = time.time()
    
    try:
        from covid_ai_model import CovidRevolutionaryTransformer
        
        model = CovidRevolutionaryTransformer(
            sequence_features=15,
            static_features=25,
            d_model=64,  # Plus petit pour test
            n_heads=2,
            n_layers=2
        )
        
        elapsed = time.time() - start
        print(f"   ✅ Modèle créé en {elapsed:.2f}s")
        print(f"   📊 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_forward_pass():
    """Test de forward pass"""
    print("\n🔍 TEST 3: Forward pass")
    start = time.time()
    
    try:
        from covid_ai_model import CovidRevolutionaryTransformer
        
        model = CovidRevolutionaryTransformer(
            sequence_features=15,
            static_features=25,
            d_model=64,
            n_heads=2,
            n_layers=2
        )
        
        # Données test
        batch_size = 4
        seq_len = 30
        
        sequences = torch.randn(batch_size, seq_len, 15)
        static = torch.randn(batch_size, 25)
        
        print("   🧠 Test forward...")
        output, uncertainty, attention = model(sequences, static, target_horizon=7)
        
        elapsed = time.time() - start
        print(f"   ✅ Forward pass en {elapsed:.2f}s")
        print(f"   📊 Output shape: {output.shape}")
        
        if elapsed > 10:
            print("   🚨 PROBLÈME: Forward trop lent!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """Test du DataLoader"""
    print("\n🔍 TEST 4: DataLoader")
    start = time.time()
    
    try:
        from covid_ai_model import CovidTransformerDataset
        from torch.utils.data import DataLoader
        
        # Données factices
        n_samples = 100
        sequences = np.random.randn(n_samples, 30, 15).astype(np.float32)
        static = np.random.randn(n_samples, 25).astype(np.float32)
        targets = np.random.randn(n_samples, 4, 4).astype(np.float32)
        
        dataset = CovidTransformerDataset(sequences, static, targets)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print("   📦 Test itération DataLoader...")
        
        # Test première batch
        batch_sequences, batch_static, batch_targets = next(iter(dataloader))
        
        elapsed = time.time() - start
        print(f"   ✅ DataLoader en {elapsed:.2f}s")
        print(f"   📊 Batch shapes: {batch_sequences.shape}, {batch_static.shape}, {batch_targets.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_training_step():
    """Test d'une étape d'entraînement"""
    print("\n🔍 TEST 5: Étape d'entraînement")
    start = time.time()
    
    try:
        from covid_ai_model import CovidRevolutionaryTransformer
        import torch.optim as optim
        import torch.nn as nn
        
        model = CovidRevolutionaryTransformer(
            sequence_features=15,
            static_features=25,
            d_model=64,
            n_heads=2,
            n_layers=2
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Une batch factice
        sequences = torch.randn(4, 30, 15)
        static = torch.randn(4, 25)
        targets = torch.randn(4, 4, 4)
        
        print("   🎯 Test étape training...")
        
        # Forward
        model.train()
        optimizer.zero_grad()
        
        # Test pour horizon 7
        pred, uncertainty, _ = model(sequences, static, target_horizon=7)
        target_horizon = targets[:, 1, :]  # Index 1 = 7 jours
        
        # Loss
        loss = nn.MSELoss()(pred, target_horizon)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start
        print(f"   ✅ Étape training en {elapsed:.2f}s")
        print(f"   📉 Loss: {loss.item():.4f}")
        
        if elapsed > 30:
            print("   🚨 PROBLÈME: Étape trop lente!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Diagnostic complet"""
    print("🚨 DIAGNOSTIC URGENT COVID IA")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_model_creation, 
        test_forward_pass,
        test_dataloader,
        test_training_step
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            
            if not result:
                print(f"\n🚨 ARRÊT: {test.__name__} a échoué!")
                break
                
        except KeyboardInterrupt:
            print("\n⚠️ Diagnostic interrompu")
            break
        except Exception as e:
            print(f"\n❌ Erreur critique dans {test.__name__}: {e}")
            break
    
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DIAGNOSTIC:")
    
    test_names = [
        "Chargement données",
        "Création modèle", 
        "Forward pass",
        "DataLoader",
        "Étape training"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    if all(results):
        print("\n🤔 BIZARRE: Tous les tests passent individuellement...")
        print("   Le problème est probablement dans la boucle d'entraînement!")
    else:
        print(f"\n🎯 PROBLÈME IDENTIFIÉ au test: {test_names[results.index(False)]}")

if __name__ == "__main__":
    main()
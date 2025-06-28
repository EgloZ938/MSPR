import pandas as pd
import os

def clean_vaccination_data():
    """Nettoie les données de vaccination pour l'IA"""
    
    input_file = '../data/dataset/cumulative-covid-vaccinations.csv'
    output_file = '../data/dataset_clean/cumulative-covid-vaccinations_clean.csv'
    
    print(f"🧹 Nettoyage des données de vaccination...")
    
    try:
        # Lire le fichier
        df = pd.read_csv(input_file)
        print(f"📊 {len(df)} lignes chargées")
        
        # Vérifier les colonnes
        print(f"📋 Colonnes: {list(df.columns)}")
        
        # Nettoyer les noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Colonnes attendues
        expected_cols = ['Entity', 'Code', 'Day', 'COVID-19 doses (cumulative)']
        
        # Vérifier que toutes les colonnes sont présentes
        if not all(col in df.columns for col in expected_cols):
            print(f"❌ Colonnes manquantes. Trouvées: {list(df.columns)}")
            return False
        
        # Filtrer les colonnes nécessaires
        df = df[expected_cols].copy()
        
        # Renommer pour standardiser
        df.columns = ['country', 'country_code', 'date', 'cumulative_vaccinations']
        
        # Nettoyer les données
        print("🔧 Nettoyage des données...")
        
        # Supprimer les lignes avec des pays vides ou des régions (pas de code pays)
        df = df.dropna(subset=['country', 'country_code'])
        df = df[df['country_code'].str.len() == 3]  # Codes pays ISO à 3 lettres
        
        # Supprimer les lignes avec des entités comme "Africa", "World", etc.
        invalid_entities = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'World']
        df = df[~df['country'].isin(invalid_entities)]
        
        # Convertir la date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convertir les vaccinations en numérique
        df['cumulative_vaccinations'] = pd.to_numeric(df['cumulative_vaccinations'], errors='coerce')
        df = df.dropna(subset=['cumulative_vaccinations'])
        
        # Supprimer les valeurs négatives
        df = df[df['cumulative_vaccinations'] >= 0]
        
        # Trier par pays et date
        df = df.sort_values(['country', 'date'])
        
        # Calculer les vaccinations quotidiennes
        df['daily_vaccinations'] = df.groupby('country')['cumulative_vaccinations'].diff().fillna(0)
        df['daily_vaccinations'] = df['daily_vaccinations'].clip(lower=0)  # Pas de valeurs négatives
        
        # Supprimer les doublons
        df = df.drop_duplicates(subset=['country', 'date'])
        
        # Statistiques finales
        print(f"✅ Données nettoyées:")
        print(f"   📊 {len(df)} enregistrements")
        print(f"   🏳️ {df['country'].nunique()} pays")
        print(f"   📅 Du {df['date'].min().strftime('%Y-%m-%d')} au {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"   💉 Total vaccinations: {df['cumulative_vaccinations'].max():,.0f}")
        
        # Sauvegarder
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"💾 Sauvegardé: {output_file}")
        
        # Afficher quelques exemples
        print("\n📋 Exemples de données:")
        sample_countries = ['France', 'Germany', 'Italy', 'Spain', 'United States']
        for country in sample_countries:
            country_data = df[df['country'] == country]
            if len(country_data) > 0:
                latest = country_data.iloc[-1]
                print(f"   {country}: {latest['cumulative_vaccinations']:,.0f} doses au {latest['date'].strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    clean_vaccination_data()
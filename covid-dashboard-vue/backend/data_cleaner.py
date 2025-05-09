import os
import pandas as pd

# Dossiers pour les fichiers bruts et nettoyés
RAW_DATA_DIR = '../data/dataset'
CLEAN_DATA_DIR = '../data/dataset_clean'

# Colonnes autorisées pour chaque fichier
ALLOWED_COLUMNS = {
    'country_wise_latest.csv': ['Country/Region', 'WHO Region'],
    'full_grouped.csv': ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],
    'worldometer_data.csv': ['Country/Region', 'Continent', 'Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop', 'Deaths/1M pop', 'Serious,Critical'],
    'covid_19_clean_complete.csv': ['Country/Region', 'Province/State', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active'],
    'usa_county_wise.csv': ['FIPS', 'Admin2', 'Province_State', 'Lat', 'Long_', 'Date', 'Confirmed', 'Deaths']
}

def clean_file(file_name):
    """Nettoie un fichier CSV spécifique."""
    raw_path = os.path.join(RAW_DATA_DIR, file_name)
    clean_path = os.path.join(CLEAN_DATA_DIR, file_name.replace('.csv', '_clean.csv'))

    print(f"Nettoyage du fichier : {file_name}")
    df = pd.read_csv(raw_path)

    # Remplir les valeurs manquantes
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

 # Filtrer les colonnes autorisées
    if file_name in ALLOWED_COLUMNS:
        df = df[ALLOWED_COLUMNS[file_name]]

    # Supprimer les doublons
    df = df.drop_duplicates()

    print(f"Fichier nettoyé - Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
    df.to_csv(clean_path, index=False)

def clean_all_files():
    """Nettoie tous les fichiers dans le dossier RAW_DATA_DIR."""
    if not os.path.exists(CLEAN_DATA_DIR):
        os.makedirs(CLEAN_DATA_DIR)

    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith('.csv'):
            clean_file(file_name)

if __name__ == "__main__":
    clean_all_files()
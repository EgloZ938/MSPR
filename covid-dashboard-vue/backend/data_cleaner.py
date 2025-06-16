# import os
# import pandas as pd

# # Dossiers pour les fichiers bruts et nettoyés
# RAW_DATA_DIR = '../data/dataset'
# CLEAN_DATA_DIR = '../data/dataset_clean'

# def get_base_filename(filename):
#     return '_'.join(filename.split('_')[:-1])



# # Colonnes autorisées pour chaque fichier
# ALLOWED_COLUMNS = {
#     'country_wise_latest.csv': ['Country/Region', 'WHO Region'],
#     'full_grouped.csv': ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],
#     'worldometer_data.csv': ['Country/Region', 'Continent', 'Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop', 'Deaths/1M pop', 'Serious,Critical'],
#     'covid_19_clean_complete.csv': ['Country/Region', 'Province/State', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active'],
#     'usa_county_wise.csv': ['FIPS', 'Admin2', 'Province_State', 'Lat', 'Long_', 'Date', 'Confirmed', 'Deaths'],
#      'covid_pooled': ['region','country','age_group','pop_date','pop_male','pop_female','pop_both','death_reference_date','death_reference_date_type','cum_death_male','cum_death_female','cum_death_unknown','cum_death_both']
# }


# def clean_data(filepath):
#     # Lire le fichier
#     df = pd.read_csv(filepath)
    
#     # Obtenir le nom du fichier sans le chemin
#     filename = os.path.basename(filepath)
    
#     # Obtenir le nom de base pour les fichiers comme covid_pooled_XXXX
#     base_name = get_base_filename(filename)
    
#     # Vérifier si le fichier est autorisé
#     if filename in ALLOWED_COLUMNS:
#         # Pour les fichiers avec nom exact
#         expected_columns = ALLOWED_COLUMNS[filename]
#     elif base_name in ALLOWED_COLUMNS:
#         # Pour les fichiers comme covid_pooled_XXXX
#         expected_columns = ALLOWED_COLUMNS[base_name]
#     else:
#         raise ValueError(f"Fichier non autorisé: {filename}")

#     # Vérifier les colonnes
#     for col in expected_columns:
#         if col not in df.columns:
#             raise ValueError(f"Colonne manquante: {col}")



# def clean_file(file_name):
#     """Nettoie un fichier CSV spécifique."""
#     raw_path = os.path.join(RAW_DATA_DIR, file_name)
#     clean_path = os.path.join(CLEAN_DATA_DIR, file_name.replace('.csv', '_clean.csv'))

#     print(f"Nettoyage du fichier : {file_name}")
#     df = pd.read_csv(raw_path)

#     # Remplir les valeurs manquantes
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#     non_numeric_cols = df.select_dtypes(exclude=['number']).columns
#     df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

#  # Filtrer les colonnes autorisées
#     if file_name in ALLOWED_COLUMNS:
#         df = df[ALLOWED_COLUMNS[file_name]]

#     # Supprimer les doublons
#     df = df.drop_duplicates()

#     print(f"Fichier nettoyé - Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
#     df.to_csv(clean_path, index=False)

# def clean_all_files():
#     """Nettoie tous les fichiers dans le dossier RAW_DATA_DIR."""
#     if not os.path.exists(CLEAN_DATA_DIR):
#         os.makedirs(CLEAN_DATA_DIR)

#     for file_name in os.listdir(RAW_DATA_DIR):
#         if file_name.endswith('.csv'):
#             clean_file(file_name)

# if __name__ == "__main__":
#     clean_all_files()


import os
import pandas as pd

# Dossiers pour les fichiers bruts et nettoyés
RAW_DATA_DIR = '../data/dataset'
CLEAN_DATA_DIR = '../data/dataset_clean'

def get_base_filename(filename):
    return '_'.join(filename.split('_')[:-1])

# Colonnes autorisées pour chaque fichier
ALLOWED_COLUMNS = {
    'country_wise_latest.csv': ['Country/Region', 'WHO Region'],
    'full_grouped.csv': ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],
    'worldometer_data.csv': ['Country/Region', 'Continent', 'Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop', 'Deaths/1M pop', 'Serious,Critical'],
    'covid_19_clean_complete.csv': ['Country/Region', 'Province/State', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active'],
    'usa_county_wise.csv': ['FIPS', 'Admin2', 'Province_State', 'Lat', 'Long_', 'Date', 'Confirmed', 'Deaths'],
     'covid_pooled': ['region','country','age_group','pop_date','pop_male','pop_female','pop_both','death_reference_date','death_reference_date_type','cum_death_male','cum_death_female','cum_death_both'],
     'Cum_deaths_by_age_sex':  ['region','country','age_group','death_reference_date','death_reference_date_type','cum_death_male','cum_death_female','cum_death_both'],
     'covid_pooled_AS':  ['region','country','age_group','death_reference_date','death_reference_date_type','cum_death_male','cum_death_female','cum_death_both'],
     'population-density.csv':  ['Entity','Code','Year','Population density']
}

def clean_file(file_name):
    """Nettoie un fichier CSV spécifique avec validation complète."""
    raw_path = os.path.join(RAW_DATA_DIR, file_name)
    clean_path = os.path.join(CLEAN_DATA_DIR, file_name.replace('.csv', '_clean.csv'))

    print(f"Nettoyage du fichier : {file_name}")
    
    # === LOGIQUE DE clean_data : VALIDATION ===
    # Lire le fichier pour validation
    df = pd.read_csv(raw_path)
    
    # Obtenir le nom de base pour les fichiers comme covid_pooled_XXXX
    base_name = get_base_filename(file_name)
    
    # Vérifier si le fichier est autorisé
    if file_name in ALLOWED_COLUMNS:
        # Pour les fichiers avec nom exact
        expected_columns = ALLOWED_COLUMNS[file_name]
    elif base_name in ALLOWED_COLUMNS:
        # Pour les fichiers comme covid_pooled_XXXX
        expected_columns = ALLOWED_COLUMNS[base_name]
    else:
        raise ValueError(f"Fichier non autorisé: {file_name}")

    # Vérifier que toutes les colonnes requises sont présentes
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col} dans le fichier {file_name}")
    
    print(f"✓ Validation réussie pour {file_name}")
    
    # === LOGIQUE DE clean_file : NETTOYAGE ===
    # Remplir les valeurs manquantes
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

    # Filtrer les colonnes autorisées (utilise expected_columns déjà déterminé)
    df = df[expected_columns]

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
            try:
                clean_file(file_name)
            except ValueError as e:
                print(f" Erreur avec {file_name}: {e}")
            except Exception as e:
                print(f" Erreur inattendue avec {file_name}: {e}")

if __name__ == "__main__":
    clean_all_files()



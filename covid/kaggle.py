import os
import pandas as pd

destination = 'dataset'
clean_destination = 'dataset_clean'

allowed_columns = {
    'country_wise_latest.csv': ['Country/Region', 'WHO Region'],
    'full_grouped.csv': ['Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],
    'worldometer_data.csv': ['Country/Region', 'Continent', 'Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop', 'Deaths/1M pop', 'Serious,Critical'],
    'covid_19_clean_complete.csv': ['Country/Region', 'Province/State', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active'],
    'usa_county_wise.csv': ['FIPS', 'Admin2', 'Province_State', 'Lat', 'Long_', 'Date', 'Confirmed', 'Deaths']
}

liste = os.listdir(destination)

for element in liste:
    if element.endswith('.csv'):
        path = os.path.join(destination, element)
        
        clean_filename = element.replace('.csv', '_clean.csv')
        clean_path = os.path.join(clean_destination, clean_filename)
        
        print(f"Nettoyage du fichier : {element}")
        
        df = pd.read_csv(path)
        
        print(f"Avant nettoyage - Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")

        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
        
        if element in allowed_columns:
            df = df[allowed_columns[element]]
                
        df_clean = df.drop_duplicates()
        
        print(f"Après nettoyage - Lignes : {df_clean.shape[0]}, Colonnes : {df_clean.shape[1]}")
        
        df_clean.to_csv(clean_path, index=False)
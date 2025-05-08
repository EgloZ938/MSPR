import os
import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from datetime import datetime
from data_cleaner import clean_all_files

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la connexion à la base de données
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def insert_who_regions(conn, df):
    """Insertion des régions WHO"""
    with conn.cursor() as cur:
        for region in pd.unique(df['WHO Region']):
            cur.execute("""
                INSERT INTO who_regions (region_name)
                VALUES (%s)
                ON CONFLICT (region_name) DO NOTHING
                RETURNING id
            """, (region,))
        conn.commit()

def insert_countries(conn, df_countries, df_worldometer):
    """Insertion des pays"""
    with conn.cursor() as cur:
        # Récupération du mapping des régions WHO
        cur.execute("SELECT region_name, id FROM who_regions")
        who_regions_map = dict(cur.fetchall())
        
        # Fusion et insertion des données pays
        for _, row in df_countries.iterrows():
            worldometer_data = df_worldometer[
                df_worldometer['Country/Region'] == row['Country/Region']
            ].iloc[0] if len(df_worldometer[df_worldometer['Country/Region'] == row['Country/Region']]) > 0 else None
            
            cur.execute("""
                INSERT INTO countries (country_name, continent, population, who_region_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (country_name) DO UPDATE
                SET continent = EXCLUDED.continent,
                    population = EXCLUDED.population,
                    who_region_id = EXCLUDED.who_region_id
                RETURNING id
            """, (
                row['Country/Region'],
                worldometer_data['Continent'] if worldometer_data is not None else None,
                worldometer_data['Population'] if worldometer_data is not None else None,
                who_regions_map.get(row['WHO Region'])
            ))
        conn.commit()

def insert_provinces(conn, df_provinces):
    """Insertion des provinces/états"""
    with conn.cursor() as cur:
        # Récupération du mapping des pays
        cur.execute("SELECT country_name, id FROM countries")
        countries_map = dict(cur.fetchall())
        
        # Insertion des provinces
        for _, row in df_provinces.iterrows():
            if pd.notna(row['Province/State']):  # On vérifie que la province n'est pas NaN
                country_id = countries_map.get(row['Country/Region'])
                if country_id:
                    cur.execute("""
                        INSERT INTO provinces (province_name, country_id, latitude, longitude)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (province_name, country_id) DO UPDATE
                        SET latitude = EXCLUDED.latitude,
                            longitude = EXCLUDED.longitude
                        RETURNING id
                    """, (
                        row['Province/State'],
                        country_id,
                        row['Lat'],
                        row['Long']
                    ))
        conn.commit()

def insert_us_counties(conn, df_counties):
    """Insertion des comtés US"""
    with conn.cursor() as cur:
        # Récupération du mapping des provinces (états US)
        cur.execute("""
            SELECT p.id, p.province_name 
            FROM provinces p 
            JOIN countries c ON p.country_id = c.id 
            WHERE c.country_name = 'US'
        """)
        states_map = dict(cur.fetchall())
        
        # Insertion des comtés
        for _, row in df_counties.drop_duplicates(subset=['FIPS', 'Admin2', 'Province_State']).iterrows():
            if pd.notna(row['FIPS']):  # On vérifie que le FIPS n'est pas NaN
                state_id = states_map.get(row['Province_State'])
                if state_id:
                    cur.execute("""
                        INSERT INTO us_counties (county_name, state_id, fips, latitude, longitude)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (fips) DO UPDATE
                        SET county_name = EXCLUDED.county_name,
                            latitude = EXCLUDED.latitude,
                            longitude = EXCLUDED.longitude
                        RETURNING id
                    """, (
                        row['Admin2'],
                        state_id,
                        str(int(row['FIPS'])) if pd.notna(row['FIPS']) else None,
                        row['Lat'],
                        row['Long_']
                    ))
        conn.commit()

def insert_daily_stats(conn, df_full):
    """Insertion des statistiques quotidiennes par pays"""
    with conn.cursor() as cur:
        # Récupération du mapping des pays
        cur.execute("SELECT country_name, id FROM countries")
        countries_map = dict(cur.fetchall())
        
        # Insertion des données quotidiennes
        for _, row in df_full.iterrows():
            country_id = countries_map.get(row['Country/Region'])
            if country_id:
                cur.execute("""
                    INSERT INTO daily_stats (
                        country_id, date, confirmed, deaths, recovered,
                        active, new_cases, new_deaths, new_recovered
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (country_id, date) DO UPDATE
                    SET confirmed = EXCLUDED.confirmed,
                        deaths = EXCLUDED.deaths,
                        recovered = EXCLUDED.recovered,
                        active = EXCLUDED.active,
                        new_cases = EXCLUDED.new_cases,
                        new_deaths = EXCLUDED.new_deaths,
                        new_recovered = EXCLUDED.new_recovered
                """, (
                    country_id,
                    row['Date'],
                    row['Confirmed'],
                    row['Deaths'],
                    row['Recovered'],
                    row['Active'],
                    row['New cases'],
                    row['New deaths'],
                    row['New recovered']
                ))
        conn.commit()

def insert_province_stats(conn, df_complete):
    """Insertion des statistiques quotidiennes par province"""
    with conn.cursor() as cur:
        # Récupération du mapping des provinces
        cur.execute("""
            SELECT p.id, p.province_name, c.country_name 
            FROM provinces p 
            JOIN countries c ON p.country_id = c.id
        """)
        provinces_map = {(row[2], row[1]): row[0] for row in cur.fetchall()}
        
        # Insertion des données par province
        for _, row in df_complete.iterrows():
            if pd.notna(row['Province/State']):
                province_id = provinces_map.get((row['Country/Region'], row['Province/State']))
                if province_id:
                    cur.execute("""
                        INSERT INTO province_stats (
                            province_id, date, confirmed, deaths, recovered, active
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (province_id, date) DO UPDATE
                        SET confirmed = EXCLUDED.confirmed,
                            deaths = EXCLUDED.deaths,
                            recovered = EXCLUDED.recovered,
                            active = EXCLUDED.active
                    """, (
                        province_id,
                        row['Date'],
                        row['Confirmed'],
                        row['Deaths'],
                        row['Recovered'],
                        row['Active']
                    ))
        conn.commit()

def insert_county_stats(conn, df_counties):
    """Insertion des statistiques par comté US"""
    with conn.cursor() as cur:
        # Récupération du mapping des comtés
        cur.execute("SELECT id, fips FROM us_counties")
        counties_map = dict(cur.fetchall())
        
        # Insertion des données par comté
        for _, row in df_counties.iterrows():
            if pd.notna(row['FIPS']):
                county_id = counties_map.get(str(int(row['FIPS'])) if pd.notna(row['FIPS']) else None)
                if county_id:
                    cur.execute("""
                        INSERT INTO county_stats (
                            county_id, date, confirmed, deaths
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (county_id, date) DO UPDATE
                        SET confirmed = EXCLUDED.confirmed,
                            deaths = EXCLUDED.deaths
                    """, (
                        county_id,
                        row['Date'],
                        row['Confirmed'],
                        row['Deaths']
                    ))
        conn.commit()

def insert_country_details(conn, df_worldometer):
    """Insertion des détails par pays"""
    with conn.cursor() as cur:
        # Récupération du mapping des pays
        cur.execute("SELECT country_name, id FROM countries")
        countries_map = dict(cur.fetchall())
        
        current_date = datetime.now().date()
        
        # Insertion des données détaillées
        for _, row in df_worldometer.iterrows():
            country_id = countries_map.get(row['Country/Region'])
            if country_id:
                cur.execute("""
                    INSERT INTO country_details (
                        country_id, total_tests, tests_per_million,
                        cases_per_million, deaths_per_million,
                        serious_critical, last_updated
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (country_id, last_updated) DO UPDATE
                    SET total_tests = EXCLUDED.total_tests,
                        tests_per_million = EXCLUDED.tests_per_million,
                        cases_per_million = EXCLUDED.cases_per_million,
                        deaths_per_million = EXCLUDED.deaths_per_million,
                        serious_critical = EXCLUDED.serious_critical
                """, (
                    country_id,
                    row['TotalTests'],
                    row['Tests/1M pop'],
                    row['Tot Cases/1M pop'],
                    row['Deaths/1M pop'],
                    row['Serious,Critical'],
                    current_date
                ))
        conn.commit()

def main():
    """Fonction principale d'importation des données"""
    try:
        print("Nettoyage des fichiers CSV...")
        clean_all_files()  # Nettoyer les fichiers avant l'importation


        # Connexion à la base de données
        print("Connexion à la base de données...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Lecture des fichiers CSV
        print("Lecture des fichiers CSV...")
        df_country_wise = pd.read_csv('../data/dataset_clean/country_wise_latest_clean.csv')
        df_full_grouped = pd.read_csv('../data/dataset_clean/full_grouped_clean.csv')
        df_worldometer = pd.read_csv('../data/dataset_clean/worldometer_data_clean.csv')
        df_complete = pd.read_csv('../data/dataset_clean/covid_19_clean_complete_clean.csv')
        df_counties = pd.read_csv('../data/dataset_clean/usa_county_wise_clean.csv')
        
        # Insertion des données
        print("Insertion des régions WHO...")
        insert_who_regions(conn, df_country_wise)
        
        print("Insertion des pays...")
        insert_countries(conn, df_country_wise, df_worldometer)
        
        print("Insertion des provinces...")
        insert_provinces(conn, df_complete)
        
        print("Insertion des comtés US...")
        insert_us_counties(conn, df_counties)
        
        print("Insertion des statistiques quotidiennes par pays...")
        insert_daily_stats(conn, df_full_grouped)
        
        print("Insertion des statistiques par province...")
        insert_province_stats(conn, df_complete)
        
        print("Insertion des statistiques par comté US...")
        insert_county_stats(conn, df_counties)
        
        print("Insertion des détails par pays...")
        insert_country_details(conn, df_worldometer)
        
        print("Import des données terminé avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'importation des données: {str(e)}")
        raise e
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
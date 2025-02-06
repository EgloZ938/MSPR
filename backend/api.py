from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Charger les données depuis le fichier CSV
covid_data = pd.read_csv('covid/covid19.csv')  # Remplacez par le chemin exact vers votre fichier CSV

@app.route('/api/covid/summary', methods=['GET'])
def get_covid_summary():
    # Vérifiez si les colonnes nécessaires existent
    required_columns = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered']
    for column in required_columns:
        if column not in covid_data.columns:
            return jsonify({'error': f'La colonne "{column}" est manquante dans le fichier CSV.'}), 400

    # Regrouper les données par pays et sommer les valeurs
    summary = covid_data.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

    # Préparer les données au format JSON
    response_data = {
        'countries': summary['Country/Region'].tolist(),
        'confirmed': summary['Confirmed'].tolist(),
        'deaths': summary['Deaths'].tolist(),
        'recovered': summary['Recovered'].tolist()
    }
    return jsonify(response_data)

@app.route('/api/covid/ai-analysis', methods=['GET'])
def ai_analysis():
    # Simuler une analyse IA
    return jsonify({'result': 'Analyse IA : Les cas confirmés augmentent rapidement dans certaines régions clés.'})

if __name__ == '__main__':
    app.run(debug=True)

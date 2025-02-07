const apiBaseUrl = 'http://127.0.0.1:5000/api';

// Déclarer les variables globales pour stocker les graphiques
let confirmedChart = null;
let deathsChart = null;
let recoveredChart = null;

async function fetchData(endpoint) {
    const response = await fetch(`${apiBaseUrl}/${endpoint}`);
    if (!response.ok) {
        console.error("Erreur lors de la récupération des données :", response.statusText);
        return {};
    }
    return response.json();
}

async function loadCharts() {
    try {
        const covidData = await fetchData('covid/summary');
        console.log("Données reçues :", covidData);

        if (!covidData.countries || !covidData.confirmed || !covidData.deaths || !covidData.recovered) {
            console.error("Les données sont incomplètes !");
            return;
        }

        // Détruire les anciens graphiques s'ils existent
        if (confirmedChart) confirmedChart.destroy();
        if (deathsChart) deathsChart.destroy();
        if (recoveredChart) recoveredChart.destroy();

        // Graphique des cas confirmés par pays
        confirmedChart = new Chart(document.getElementById('covid-confirmed-cases').getContext('2d'), {
            type: 'bar',
            data: {
                labels: covidData.countries,
                datasets: [{
                    label: 'Cas Confirmés',
                    data: covidData.confirmed,
                    backgroundColor: '#43a047'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'Cas Confirmés par Pays' }
                }
            }
        });

        // Graphique des décès par pays
        deathsChart = new Chart(document.getElementById('covid-deaths').getContext('2d'), {
            type: 'bar',
            data: {
                labels: covidData.countries,
                datasets: [{
                    label: 'Décès',
                    data: covidData.deaths,
                    backgroundColor: '#e53935'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'Décès par Pays' }
                }
            }
        });

        // Graphique des récupérations par pays
        recoveredChart = new Chart(document.getElementById('covid-recovered').getContext('2d'), {
            type: 'bar',
            data: {
                labels: covidData.countries,
                datasets: [{
                    label: 'Récupérations',
                    data: covidData.recovered,
                    backgroundColor: '#1e88e5'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'Récupérations par Pays' }
                }
            }
        });
    } catch (error) {
        console.error("Erreur lors de la création des graphiques :", error);
    }
}

// Charger les graphiques au démarrage
loadCharts();
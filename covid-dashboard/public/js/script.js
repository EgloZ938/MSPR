
// Variables pour la section régionale
let regionChart = null;
let regionData = null;
let regionConfig = {
    datasets: {},
    colors: {}
};
let countryChart = null;
let countryData = null;
let selectedCountry = "France";
let countryConfig = {
    datasets: {
        confirmed: true,
        deaths: true,
        recovered: true,
        active: true
    },
    colors: {
        confirmed: '#1a73e8',
        deaths: '#dc3545',
        recovered: '#28a745',
        active: '#ffc107'
    }
};

// Les couleurs par défaut pour les régions
const defaultRegionColors = {
    'Europe': '#1a73e8',
    'Americas': '#dc3545',
    'Western Pacific': '#28a745',
    'Eastern Mediterranean': '#ffc107',
    'Africa': '#6f42c1',
    'South-East Asia': '#fd7e14'
};

// Variables globales
let worldChart = null;
let chartData = null;
let chartConfig = {
    datasets: {
        confirmed: true,
        deaths: true,
        recovered: true,
        active: true
    },
    colors: {
        confirmed: '#1a73e8',
        deaths: '#dc3545',
        recovered: '#28a745',
        active: '#ffc107'
    }
};

// Fonction pour formater les nombres
function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
}

// Fonction pour obtenir les options du graphique
function getChartOptions() {
    const scaleType = document.getElementById('scaleType').value;

    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            zoom: {
                zoom: {
                    wheel: {
                        enabled: true,
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'xy'
                },
                pan: {
                    enabled: true,
                    mode: 'xy'
                }
            },
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 15,
                    color: '#666'
                },
                onClick: function (e, legendItem, legend) {
                    const index = legendItem.datasetIndex;
                    const ci = legend.chart;
                    const meta = ci.getDatasetMeta(index);

                    meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                    ci.update();

                    // Mise à jour des checkboxes
                    const datasetId = ci.data.datasets[index].id;
                    document.getElementById(`toggle${datasetId}`).checked = !meta.hidden;
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(255,255,255,0.9)',
                titleColor: '#666',
                bodyColor: '#666',
                borderColor: 'rgba(0,0,0,0.1)',
                borderWidth: 1,
                padding: 10,
                callbacks: {
                    label: function (context) {
                        return `${context.dataset.label}: ${formatNumber(context.raw)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                type: scaleType,
                beginAtZero: true,
                grid: {
                    color: 'rgba(0,0,0,0.1)'
                },
                ticks: {
                    color: '#666',
                    callback: function (value) {
                        if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                        if (value >= 1000) return (value / 1000).toFixed(0) + 'k';
                        return value;
                    }
                }
            },
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: '#666',
                    maxRotation: 45,
                    minRotation: 45,
                    maxTicksLimit: 12
                }
            }
        },
        animation: {
            duration: 750,
            easing: 'easeInOutQuart'
        }
    };
}

// Chargement des statistiques globales
async function loadGlobalStats() {
    try {
        const response = await fetch('/api/global-stats');
        const data = await response.json();

        document.getElementById('confirmed').textContent = formatNumber(data.total_confirmed);
        document.getElementById('deaths').textContent = formatNumber(data.total_deaths);
        document.getElementById('recovered').textContent = formatNumber(data.total_recovered);
        document.getElementById('active').textContent = formatNumber(data.total_active);
    } catch (error) {
        console.error('Erreur:', error);
        showError('Erreur lors du chargement des statistiques globales');
    }
}

// Mise à jour du graphique
async function updateChart() {
    try {
        toggleLoading(true);
        const response = await fetch('/api/global-timeline');
        let data = await response.json();

        // Stocker les données pour l'export CSV
        chartData = data;

        const chartType = document.getElementById('chartType').value;
        const dataFormat = document.getElementById('dataFormat').value;

        // Traitement des données selon le format
        data = processData(data, dataFormat);

        const ctx = document.getElementById('worldChart').getContext('2d');

        if (worldChart) {
            worldChart.destroy();
        }

        const datasets = [];

        // Création des datasets en fonction des sélections
        if (chartConfig.datasets.confirmed) {
            datasets.push(createDataset('Confirmed', data.map(item => item.confirmed), chartConfig.colors.confirmed));
        }
        if (chartConfig.datasets.deaths) {
            datasets.push(createDataset('Deaths', data.map(item => item.deaths), chartConfig.colors.deaths));
        }
        if (chartConfig.datasets.recovered) {
            datasets.push(createDataset('Recovered', data.map(item => item.recovered), chartConfig.colors.recovered));
        }
        if (chartConfig.datasets.active) {
            datasets.push(createDataset('Active', data.map(item => item.active), chartConfig.colors.active));
        }

        worldChart = new Chart(ctx, {
            type: getChartType(chartType),
            data: {
                labels: data.map(item => new Date(item.date).toLocaleDateString()),
                datasets: datasets
            },
            options: getChartOptions()
        });
    } catch (error) {
        console.error('Erreur:', error);
        showError('Erreur lors de la mise à jour du graphique');
    } finally {
        toggleLoading(false);
    }
}

// Fonction helper pour créer un dataset
function createDataset(label, data, color) {
    const chartType = document.getElementById('chartType').value;
    return {
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: chartType === 'bar' ? `${color}88` : `${color}22`,
        fill: chartType === 'area',
        tension: 0.4,
        id: label.toLowerCase(),
        pointRadius: chartType === 'scatter' ? 4 : 0,
        pointHoverRadius: 6
    };
}

// Toggle dataset visibility
function toggleDataset(datasetName) {
    chartConfig.datasets[datasetName] = !chartConfig.datasets[datasetName];
    updateChart();
}

// Reset zoom
function resetZoom() {
    if (worldChart) {
        worldChart.resetZoom();
    }
}

// Download chart as image
function downloadChart() {
    if (worldChart) {
        const link = document.createElement('a');
        link.download = 'covid-chart.png';
        link.href = worldChart.toBase64Image();
        link.click();
    }
}



// Export data as CSV
function exportData() {
    if (!chartData) return;

    const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']];
    chartData.forEach(item => {
        rows.push([
            new Date(item.date).toLocaleDateString(),
            item.confirmed,
            item.deaths,
            item.recovered,
            item.active
        ]);
    });

    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "covid_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Fonction pour traiter les données selon le format choisi
function processData(data, format) {
    switch (format) {
        case 'percentage':
            const totals = data.map(item => item.confirmed + item.deaths + item.recovered + item.active);
            return data.map((item, i) => ({
                ...item,
                confirmed: (item.confirmed / totals[i]) * 100,
                deaths: (item.deaths / totals[i]) * 100,
                recovered: (item.recovered / totals[i]) * 100,
                active: (item.active / totals[i]) * 100
            }));
        case 'daily':
            return data.map((item, index) => {
                if (index === 0) return item;
                return {
                    ...item,
                    confirmed: item.confirmed - data[index - 1].confirmed,
                    deaths: item.deaths - data[index - 1].deaths,
                    recovered: item.recovered - data[index - 1].recovered,
                    active: item.active - data[index - 1].active
                };
            });
        case 'weekly':
            return calculateMovingAverage(data, 7);
        case 'monthly':
            return calculateMovingAverage(data, 30);
        default:
            return data;
    }
}

// Fonction pour calculer la moyenne mobile
function calculateMovingAverage(data, window) {
    return data.map((item, index, array) => {
        if (index < window - 1) return item;

        const slice = array.slice(index - window + 1, index + 1);
        return {
            ...item,
            confirmed: slice.reduce((sum, curr) => sum + curr.confirmed, 0) / window,
            deaths: slice.reduce((sum, curr) => sum + curr.deaths, 0) / window,
            recovered: slice.reduce((sum, curr) => sum + curr.recovered, 0) / window,
            active: slice.reduce((sum, curr) => sum + curr.active, 0) / window
        };
    });
}

// Fonction pour afficher les erreurs
function showError(message) {
    const container = document.querySelector('.container');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.style.cssText = `
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        text-align: center;
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    `;
    errorDiv.textContent = message;
    container.insertBefore(errorDiv, container.firstChild);
    setTimeout(() => errorDiv.remove(), 5000);
}

// Fonction pour obtenir le type de graphique
function getChartType(selectedType) {
    switch (selectedType) {
        case 'area':
            return 'line';
        case 'mixed':
            return 'bar';
        default:
            return selectedType;
    }
}

/**
 * Zoom in sur le graphique mondial
 */
function zoomIn() {
    if (worldChart) {
        worldChart.zoom(1.2); // Zoom de 20%
    }
}

/**
 * Zoom out sur le graphique mondial
 */
function zoomOut() {
    if (worldChart) {
        worldChart.zoom(0.8); // Zoom out de 20%
    }
}

/**
 * Zoom in sur le graphique régional
 */
function zoomInRegion() {
    if (regionChart) {
        regionChart.zoom(1.2); // Zoom de 20%
    }
}

/**
 * Zoom out sur le graphique régional
 */
function zoomOutRegion() {
    if (regionChart) {
        regionChart.zoom(0.8); // Zoom out de 20%
    }
}

// Fonction pour mettre à jour le graphique régional
async function updateRegionChart() {
    try {
        toggleLoading(true);

        // Récupération des données
        const response = await fetch('/api/region-stats');
        let data = await response.json();

        // Stocker les données pour l'export CSV
        regionData = data;

        const chartType = document.getElementById('regionChartType').value;
        const dataFormat = document.getElementById('regionDataFormat').value;
        const scaleType = document.getElementById('regionScaleType').value;

        // Obtenir la liste unique des régions
        const regions = [...new Set(data.map(item => item.region_name))];

        // Initialiser regionConfig pour les nouvelles régions
        regions.forEach(region => {
            if (regionConfig.datasets[region] === undefined) {
                regionConfig.datasets[region] = true;
            }
            if (regionConfig.colors[region] === undefined) {
                regionConfig.colors[region] = defaultRegionColors[region] ||
                    '#' + Math.floor(Math.random() * 16777215).toString(16); // Couleur aléatoire si non définie
            }
        });

        // Regrouper les données par date pour faciliter le calcul des variations quotidiennes
        let groupedByDate = {};
        data.forEach(item => {
            if (!groupedByDate[item.date]) {
                groupedByDate[item.date] = {};
            }
            groupedByDate[item.date][item.region_name] = item;
        });

        // Trier les dates
        const sortedDates = Object.keys(groupedByDate).sort((a, b) => new Date(a) - new Date(b));

        // Traitement des données selon le format sélectionné
        if (dataFormat === 'daily') {
            for (let i = 1; i < sortedDates.length; i++) {
                const currentDate = sortedDates[i];
                const previousDate = sortedDates[i - 1];

                regions.forEach(region => {
                    const current = groupedByDate[currentDate][region];
                    const previous = groupedByDate[previousDate][region];

                    if (current && previous) {
                        current.confirmed = current.confirmed - previous.confirmed;
                        current.deaths = current.deaths - previous.deaths;
                        current.recovered = current.recovered - previous.recovered;
                        current.active = current.active - previous.active;
                    }
                });
            }
        }

        // Réorganiser les données pour le graphique
        const datasets = [];
        regions.forEach(region => {
            if (regionConfig.datasets[region]) {
                const regionData = sortedDates.map(date => {
                    return groupedByDate[date][region] ? groupedByDate[date][region].confirmed : 0;
                });

                datasets.push({
                    label: region,
                    data: regionData,
                    borderColor: regionConfig.colors[region],
                    backgroundColor: chartType === 'bar' ?
                        `${regionConfig.colors[region]}88` :
                        `${regionConfig.colors[region]}22`,
                    fill: chartType === 'area',
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
        });

        const ctx = document.getElementById('regionChart').getContext('2d');

        if (regionChart) {
            regionChart.destroy();
        }

        // Créer le graphique
        regionChart = new Chart(ctx, {
            type: getChartType(chartType),
            data: {
                labels: sortedDates.map(date => new Date(date).toLocaleDateString()),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: false,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'xy'
                        },
                        pan: {
                            enabled: true,
                            mode: 'xy'
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            color: '#666'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(255,255,255,0.9)',
                        titleColor: '#666',
                        bodyColor: '#666',
                        borderColor: 'rgba(0,0,0,0.1)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function (context) {
                                return `${context.dataset.label}: ${formatNumber(context.raw)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        type: scaleType,
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        ticks: {
                            color: '#666',
                            callback: function (value) {
                                if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                if (value >= 1000) return (value / 1000).toFixed(0) + 'k';
                                return value;
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#666',
                            maxRotation: 45,
                            minRotation: 45,
                            maxTicksLimit: 12
                        }
                    }
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });

        // Mettre à jour les toggles et les couleurs
        updateRegionToggles(regions);
        updateRegionColors(regions);

    } catch (error) {
        console.error('Erreur:', error);
        showError('Erreur lors de la mise à jour du graphique régional');
    } finally {
        toggleLoading(false);
    }
}

// Fonction pour traiter les données régionales
function processRegionData(data, format) {
    switch (format) {
        case 'daily':
            return data.map((item, index, arr) => {
                if (index === 0) return item;
                const prevItem = arr.find(prev =>
                    prev.region_name === item.region_name &&
                    new Date(prev.date) < new Date(item.date)
                );
                return {
                    ...item,
                    confirmed: prevItem ? item.confirmed - prevItem.confirmed : item.confirmed,
                    deaths: prevItem ? item.deaths - prevItem.deaths : item.deaths,
                    recovered: prevItem ? item.recovered - prevItem.recovered : item.recovered
                };
            });
        default:
            return data;
    }
}

// Fonction pour mettre à jour les toggles des régions
function updateRegionToggles(regions) {
    const container = document.getElementById('regionToggles');
    container.innerHTML = '';

    regions.forEach(region => {
        const label = document.createElement('label');
        label.className = 'toggle-item';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = `toggle${region.replace(/\s+/g, '')}`;
        input.checked = !!regionConfig.datasets[region];
        input.onchange = () => toggleRegion(region);

        const span = document.createElement('span');
        span.textContent = region;

        label.appendChild(input);
        label.appendChild(span);
        container.appendChild(label);
    });
}


// Fonction pour mettre à jour les sélecteurs de couleur
function updateRegionColors(regions) {
    const container = document.getElementById('regionColors');
    container.innerHTML = '';

    regions.forEach(region => {
        const colorOption = document.createElement('div');
        colorOption.className = 'color-option';

        const input = document.createElement('input');
        input.type = 'color';
        input.id = `color${region.replace(/\s+/g, '')}`;
        input.value = regionConfig.colors[region] || defaultRegionColors[region] || '#000000';
        input.onchange = (e) => updateRegionColor(region, e.target.value);

        const span = document.createElement('span');
        span.textContent = region;

        colorOption.appendChild(input);
        colorOption.appendChild(span);
        container.appendChild(colorOption);
    });
}

// Fonction pour basculer la visibilité d'une région
function toggleRegion(region) {
    regionConfig.datasets[region] = !regionConfig.datasets[region];
    document.getElementById(`toggle${region.replace(/\s+/g, '')}`).checked = regionConfig.datasets[region];
    updateRegionChart();
}

// Fonction pour mettre à jour la couleur d'une région
function updateRegionColor(region, color) {
    regionConfig.colors[region] = color;
    updateRegionChart();
}

// Fonction pour réinitialiser le zoom
function resetRegionZoom() {
    if (regionChart) {
        regionChart.resetZoom();
    }
}

// Fonction pour télécharger le graphique régional
function downloadRegionChart() {
    if (regionChart) {
        const link = document.createElement('a');
        link.download = 'region-chart.png';
        link.href = regionChart.toBase64Image();
        link.click();
    }
}

// Fonction pour exporter les données régionales
function exportRegionData() {
    if (!regionData) return;

    const rows = [['Date', 'Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']];
    regionData.forEach(item => {
        rows.push([
            new Date(item.date).toLocaleDateString(),
            item.region_name,
            item.confirmed,
            item.deaths,
            item.recovered,
            item.active
        ]);
    });

    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "region_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Fonction pour changer de vue
function switchView(view) {
    try {
        // Masquer toutes les stats cards si on n'est pas dans la vue mondiale
        const statsGrid = document.querySelector('.stats-grid');
        if (statsGrid) {
            statsGrid.style.display = view === 'mondial' ? 'grid' : 'none';
        }

        // Cacher toutes les sections
        document.querySelectorAll('.visualization-section').forEach(section => {
            section.style.display = 'none';
        });

        // Afficher la section sélectionnée
        const selectedSection = document.getElementById(view);
        if (selectedSection) {
            selectedSection.style.display = 'block';
        }

        // Mettre à jour la navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        const activeLink = document.querySelector(`[href="#${view}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Charger les données appropriées
        switch (view) {
            case 'mondial':
                loadGlobalStats();
                updateChart();
                break;
            case 'regions':
                // S'assurer que les éléments nécessaires existent avant de charger
                if (document.getElementById('regionChart')) {
                    updateRegionChart();
                }
                break;
            case 'pays':
                // Initialisation de la vue pays sans filtre de date
                loadCountries();
                updateCountryChart();
                break;
            case 'correlation':
                // À implémenter plus tard
                break;
            case 'tendances':
                // À implémenter plus tard
                break;
        }
    } catch (error) {
        console.error('Erreur lors du changement de vue:', error);
        showError('Erreur lors du changement de vue');
    }
}

// Fonction pour afficher/masquer le message de chargement
function toggleLoading(show) {
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        if (show) {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading';
            loadingDiv.textContent = 'Chargement des données...';
            container.appendChild(loadingDiv);
        } else {
            const loading = container.querySelector('.loading');
            if (loading) loading.remove();
        }
    });
}

// Initialisation
document.addEventListener('DOMContentLoaded', () => {
    loadGlobalStats();
    updateChart();

    // Event listeners pour les changements de couleur
    ['confirmed', 'deaths', 'recovered', 'active'].forEach(dataset => {
        document.getElementById(`${dataset}Color`).addEventListener('change', (e) => {
            chartConfig.colors[dataset] = e.target.value;
            updateChart();
        });
    });
});

// Chargement de la liste des pays
async function loadCountries() {
    try {
        const response = await fetch('/api/countries');
        const countries = await response.json();

        const selectElement = document.getElementById('countrySelect');
        selectElement.innerHTML = '';

        countries.forEach(country => {
            const option = document.createElement('option');
            option.value = country.country_name;
            option.textContent = country.country_name;
            selectElement.appendChild(option);
        });

        // Sélectionner la France par défaut
        selectElement.value = selectedCountry;

        // Initialiser la recherche
        initCountrySearch();
    } catch (error) {
        console.error('Erreur lors du chargement des pays:', error);
        showError('Erreur lors du chargement de la liste des pays');
    }
}

// Initialisation de la fonctionnalité de recherche
function initCountrySearch() {
    const searchInput = document.getElementById('countrySearch');
    const selectElement = document.getElementById('countrySelect');

    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const options = selectElement.options;

        for (let i = 0; i < options.length; i++) {
            const countryName = options[i].textContent.toLowerCase();
            if (countryName.includes(searchTerm)) {
                options[i].style.display = '';
            } else {
                options[i].style.display = 'none';
            }
        }
    });
}

// Chargement des statistiques pour un pays spécifique
async function loadCountryStats() {
    try {
        const response = await fetch(`/api/country-timeline/${selectedCountry}`);
        const data = await response.json();

        // Stocker les données pour l'export CSV
        countryData = data;

        // Afficher les dernières statistiques
        if (data.length > 0) {
            const latestStats = data[data.length - 1];
            document.getElementById('countryConfirmed').textContent = formatNumber(latestStats.confirmed);
            document.getElementById('countryDeaths').textContent = formatNumber(latestStats.deaths);
            document.getElementById('countryRecovered').textContent = formatNumber(latestStats.recovered);
            document.getElementById('countryActive').textContent = formatNumber(latestStats.active);
            document.getElementById('countryMortalityRate').textContent = latestStats.mortality_rate.toFixed(2) + '%';
        }
    } catch (error) {
        console.error('Erreur lors du chargement des statistiques du pays:', error);
        showError(`Erreur lors du chargement des statistiques pour ${selectedCountry}`);
    }
}

// Mise à jour du graphique du pays
async function updateCountryChart() {
    try {
        toggleLoading(true);

        // Récupérer le pays sélectionné
        const selectElement = document.getElementById('countrySelect');
        selectedCountry = selectElement.value;

        // Charger les statistiques pour ce pays
        await loadCountryStats();

        if (!countryData || countryData.length === 0) {
            showError(`Aucune donnée disponible pour ${selectedCountry}`);
            return;
        }

        const chartType = document.getElementById('countryChartType').value;
        const dataFormat = document.getElementById('countryDataFormat').value;

        // Traitement des données selon le format
        let processedData = countryData;
        if (dataFormat === 'daily') {
            processedData = countryData.map((item, index) => {
                if (index === 0) return item;
                return {
                    ...item,
                    confirmed: item.confirmed - countryData[index - 1].confirmed,
                    deaths: item.deaths - countryData[index - 1].deaths,
                    recovered: item.recovered - countryData[index - 1].recovered,
                    active: item.active - countryData[index - 1].active
                };
            });
        }

        const ctx = document.getElementById('countryChart').getContext('2d');

        if (countryChart) {
            countryChart.destroy();
        }

        // Configuration spécifique pour les graphiques de type camembert/anneau
        if (chartType === 'pie' || chartType === 'doughnut') {
            // Pour les graphiques circulaires, on utilise seulement la dernière date
            const latestData = processedData[processedData.length - 1];

            const labels = [];
            const data = [];
            const backgroundColors = [];

            if (countryConfig.datasets.confirmed) {
                labels.push('Cas confirmés');
                data.push(latestData.confirmed);
                backgroundColors.push(countryConfig.colors.confirmed);
            }
            if (countryConfig.datasets.deaths) {
                labels.push('Décès');
                data.push(latestData.deaths);
                backgroundColors.push(countryConfig.colors.deaths);
            }
            if (countryConfig.datasets.recovered) {
                labels.push('Guéris');
                data.push(latestData.recovered);
                backgroundColors.push(countryConfig.colors.recovered);
            }
            if (countryConfig.datasets.active) {
                labels.push('Cas actifs');
                data.push(latestData.active);
                backgroundColors.push(countryConfig.colors.active);
            }

            countryChart = new Chart(ctx, {
                type: chartType,
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: backgroundColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    const value = context.raw;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return `${context.label}: ${formatNumber(value)} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        } else {
            // Configuration pour les graphiques chronologiques (ligne, barres)
            const datasets = [];

            if (countryConfig.datasets.confirmed) {
                datasets.push(createDataset('Confirmed', processedData.map(item => item.confirmed), countryConfig.colors.confirmed, chartType));
            }
            if (countryConfig.datasets.deaths) {
                datasets.push(createDataset('Deaths', processedData.map(item => item.deaths), countryConfig.colors.deaths, chartType));
            }
            if (countryConfig.datasets.recovered) {
                datasets.push(createDataset('Recovered', processedData.map(item => item.recovered), countryConfig.colors.recovered, chartType));
            }
            if (countryConfig.datasets.active) {
                datasets.push(createDataset('Active', processedData.map(item => item.active), countryConfig.colors.active, chartType));
            }

            countryChart = new Chart(ctx, {
                type: getChartType(chartType),
                data: {
                    labels: processedData.map(item => new Date(item.date).toLocaleDateString()),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        zoom: {
                            zoom: {
                                wheel: {
                                    enabled: false,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'xy'
                            },
                            pan: {
                                enabled: true,
                                mode: 'xy'
                            }
                        },
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function (context) {
                                    return `${context.dataset.label}: ${formatNumber(context.raw)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function (value) {
                                    if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                    if (value >= 1000) return (value / 1000).toFixed(0) + 'k';
                                    return value;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Charger le classement des pays
        loadCountryRanking('confirmed');

    } catch (error) {
        console.error('Erreur lors de la mise à jour du graphique du pays:', error);
        showError(`Erreur lors de la mise à jour du graphique pour ${selectedCountry}`);
    } finally {
        toggleLoading(false);
    }
}

// Toggle dataset visibility pour le pays
function toggleCountryDataset(datasetName) {
    countryConfig.datasets[datasetName] = !countryConfig.datasets[datasetName];
    updateCountryChart();
}

// Fonctions de zoom pour le graphique du pays
function zoomInCountry() {
    if (countryChart) {
        const zoomOptions = countryChart.options.plugins.zoom.zoom;
        zoomOptions.wheel.enabled = false;

        const centerX = countryChart.chartArea.width / 2;
        const centerY = countryChart.chartArea.height / 2;
        countryChart.pan({ x: 0, y: 0 }, 'none', 'default');
        countryChart.zoom(1.2, 'xy', { x: centerX, y: centerY });
    }
}

function zoomOutCountry() {
    if (countryChart) {
        const zoomOptions = countryChart.options.plugins.zoom.zoom;
        zoomOptions.wheel.enabled = false;

        const centerX = countryChart.chartArea.width / 2;
        const centerY = countryChart.chartArea.height / 2;
        countryChart.pan({ x: 0, y: 0 }, 'none', 'default');
        countryChart.zoom(0.8, 'xy', { x: centerX, y: centerY });
    }
}

function resetCountryZoom() {
    if (countryChart) {
        countryChart.resetZoom();
    }
}

// Télécharger le graphique comme image
function downloadCountryChart() {
    if (countryChart) {
        const link = document.createElement('a');
        link.download = `covid-${selectedCountry.toLowerCase()}-chart.png`;
        link.href = countryChart.toBase64Image();
        link.click();
    }
}

// Exporter les données en CSV
function exportCountryData() {
    if (!countryData) return;

    const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate']];
    countryData.forEach(item => {
        rows.push([
            new Date(item.date).toLocaleDateString(),
            item.confirmed,
            item.deaths,
            item.recovered,
            item.active,
            item.mortality_rate.toFixed(2)
        ]);
    });

    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `covid-${selectedCountry.toLowerCase()}-data.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Chargement du classement des pays
async function loadCountryRanking(metric) {
    try {
        // Activer l'onglet correspondant
        document.querySelectorAll('.ranking-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`.ranking-tab[onclick="updateRanking('${metric}')"]`).classList.add('active');

        // Charger les données
        const response = await fetch('/api/top-countries');
        const countries = await response.json();

        const rankingContainer = document.getElementById('countryRanking');
        rankingContainer.innerHTML = '';

        // Trier par métrique choisie
        let sortedCountries;
        if (metric === 'confirmed') {
            sortedCountries = countries.sort((a, b) => b.confirmed - a.confirmed);
        } else if (metric === 'deaths') {
            sortedCountries = countries.sort((a, b) => b.deaths - a.deaths);
        } else if (metric === 'mortality') {
            sortedCountries = countries.sort((a, b) => b.mortality_rate - a.mortality_rate);
        }

        // Afficher le top 20
        sortedCountries.slice(0, 20).forEach((country, index) => {
            const item = document.createElement('div');
            item.className = 'ranking-item';

            let valueToShow;
            if (metric === 'confirmed') {
                valueToShow = formatNumber(country.confirmed);
            } else if (metric === 'deaths') {
                valueToShow = formatNumber(country.deaths);
            } else if (metric === 'mortality') {
                valueToShow = country.mortality_rate.toFixed(2) + '%';
            }

            // Highlight du pays sélectionné
            if (country.country_region === selectedCountry) {
                item.style.backgroundColor = 'rgba(26, 115, 232, 0.1)';
                item.style.fontWeight = 'bold';
            }

            item.innerHTML = `
                <span class="rank">${index + 1}</span>
                <span class="country">${country.country_region}</span>
                <span class="value">${valueToShow}</span>
            `;

            // Cliquer sur un pays le sélectionne
            item.addEventListener('click', () => {
                document.getElementById('countrySelect').value = country.country_region;
                selectedCountry = country.country_region;
                updateCountryChart();
            });

            rankingContainer.appendChild(item);
        });
    } catch (error) {
        console.error('Erreur lors du chargement du classement:', error);
    }
}

// Mettre à jour le classement des pays
function updateRanking(metric) {
    loadCountryRanking(metric);
}
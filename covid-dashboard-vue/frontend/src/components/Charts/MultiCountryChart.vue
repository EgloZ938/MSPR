<template>
    <div>
        <ChartOptions :chartType="chartType" :dataFormat="dataFormat" :scaleType="scaleType"
            :datasets="chartConfig.datasets" :colors="chartConfig.colors" @toggle-dataset="toggleDataset"
            @update-color="updateColor" @chart-type-change="updateChartType" @data-format-change="updateDataFormat"
            @scale-type-change="updateScaleType" />

        <ChartControls @zoom-in="zoomIn" @zoom-out="zoomOut" @reset-zoom="resetZoom" @download="downloadChart"
            @export="exportData" />

        <div class="chart-container">
            <canvas id="multiCountryChart" ref="multiCountryChartRef"></canvas>
        </div>
        
        <!-- Légende des pays -->
        <div class="countries-legend">
            <div v-for="(country, index) in selectedCountries" :key="country" class="country-legend-item"
                 :style="{ borderColor: countryColors[country] || defaultColors[index % defaultColors.length] }">
                {{ country }}
            </div>
        </div>
        
        <!-- Mini Loader -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref, watch, computed, onMounted, onBeforeUnmount } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartOptions from './ChartOptions.vue';
import ChartControls from './ChartControls.vue';
import MiniLoader from '../MiniLoader.vue';
import axios from 'axios';

Chart.register(zoomPlugin);

const props = defineProps({
    selectedCountries: {
        type: Array,
        required: true,
        default: () => ['France']
    },
    dataMetric: {
        type: String,
        default: 'confirmed'
    }
});

const emit = defineEmits(['toggle-loading', 'show-error']);

const multiCountryChart = ref(null);
const multiCountryChartRef = ref(null);
const countriesData = ref({});
const isCoolingDown = ref(false);
const isDestroyed = ref(false);

// Couleurs pour les différents pays
const defaultColors = [
    '#1a73e8',  // bleu
    '#dc3545',  // rouge
    '#28a745',  // vert
    '#fd7e14',  // orange
    '#6f42c1',  // violet
    '#6c757d'   // gris
];

const countryColors = ref({});

// Variables réactives pour le graphique
const chartConfig = ref({
    datasets: {
        confirmed: true,
        deaths: true,
        recovered: false,
        active: false
    },
    colors: {
        confirmed: '#1a73e8',
        deaths: '#dc3545',
        recovered: '#28a745',
        active: '#ffc107'
    }
});

const chartType = ref('line');
const dataFormat = ref('raw');
const scaleType = ref('linear');

// Surveillance des changements de pays sélectionnés
watch(() => props.selectedCountries, (newCountries) => {
    if (!isDestroyed.value) {
        // Attribuer des couleurs aux pays s'ils n'en ont pas déjà
        newCountries.forEach((country, index) => {
            if (!countryColors.value[country]) {
                countryColors.value[country] = defaultColors[index % defaultColors.length];
            }
        });
        
        // Mettre à jour le graphique
        updateMultiCountryChart();
    }
}, { deep: true, immediate: true });

function activateCooldown(duration = 800) {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, duration);
}

async function updateMultiCountryChart() {
    if (isDestroyed.value || isCoolingDown.value) return;
    
    try {
        activateCooldown(800);
        emit('toggle-loading', true);

        // Charger les données pour chaque pays si elles ne sont pas déjà chargées
        for (const country of props.selectedCountries) {
            if (!countriesData.value[country]) {
                await loadCountryData(country);
            }
        }

        if (isDestroyed.value) return;

        // Vérifier si la référence existe
        if (!multiCountryChartRef.value) {
            console.error("La référence multiCountryChartRef est null");
            emit('show-error', 'Erreur lors de la mise à jour du graphique - référence manquante');
            return;
        }

        const ctx = multiCountryChartRef.value.getContext('2d');

        if (multiCountryChart.value) {
            multiCountryChart.value.destroy();
        }
        
        if (isDestroyed.value) return;

        // Créer les datasets pour le graphique
        const datasets = [];
        
        // Pour chaque pays, ajouter un dataset pour chaque métrique active
        props.selectedCountries.forEach(country => {
            if (!countriesData.value[country]) return;
            
            const countryData = processData(countriesData.value[country], dataFormat.value);
            const countryColor = countryColors.value[country];
            
            if (chartConfig.value.datasets.confirmed) {
                datasets.push({
                    label: `${country} - Cas confirmés`,
                    data: countryData.map(item => item.confirmed),
                    borderColor: countryColor,
                    backgroundColor: `${countryColor}22`,
                    fill: chartType.value === 'area',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
            
            if (chartConfig.value.datasets.deaths) {
                const deathsColor = getDarkerColor(countryColor);
                datasets.push({
                    label: `${country} - Décès`,
                    data: countryData.map(item => item.deaths),
                    borderColor: deathsColor,
                    backgroundColor: `${deathsColor}22`,
                    fill: chartType.value === 'area',
                    tension: 0.4,
                    borderWidth: 2,
                    borderDash: [5, 5],  // Style pointillé pour les décès
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
            
            if (chartConfig.value.datasets.recovered) {
                datasets.push({
                    label: `${country} - Guéris`,
                    data: countryData.map(item => item.recovered),
                    borderColor: chartConfig.value.colors.recovered,
                    backgroundColor: `${chartConfig.value.colors.recovered}22`,
                    fill: chartType.value === 'area',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
            
            if (chartConfig.value.datasets.active) {
                datasets.push({
                    label: `${country} - Cas actifs`,
                    data: countryData.map(item => item.active),
                    borderColor: chartConfig.value.colors.active,
                    backgroundColor: `${chartConfig.value.colors.active}22`,
                    fill: chartType.value === 'area',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
        });

        // Créer le graphique
        if (!isDestroyed.value && datasets.length > 0) {
            // Trouver toutes les dates uniques de tous les pays
            const allDates = new Set();
            props.selectedCountries.forEach(country => {
                if (countriesData.value[country]) {
                    countriesData.value[country].forEach(item => {
                        allDates.add(item.date);
                    });
                }
            });
            
            // Convertir en tableau et trier
            const sortedDates = [...allDates].sort((a, b) => new Date(a) - new Date(b));
            
            multiCountryChart.value = new Chart(ctx, {
                type: getChartType(chartType.value),
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
                                label: (context) => {
                                    return `${context.dataset.label}: ${formatNumber(context.raw)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            type: scaleType.value,
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            },
                            ticks: {
                                color: '#666',
                                callback: (value) => {
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
        }

    } catch (error) {
        console.error('Erreur lors de la mise à jour du graphique multi-pays:', error);
        if (!isDestroyed.value) {
            emit('show-error', 'Erreur lors de la mise à jour du graphique de comparaison des pays');
        }
    } finally {
        if (!isDestroyed.value) {
            emit('toggle-loading', false);
        }
    }
}

async function loadCountryData(country) {
    try {
        const response = await axios.get(`/api/country-timeline/${country}`);
        countriesData.value[country] = response.data;
    } catch (error) {
        console.error(`Erreur lors du chargement des données pour ${country}:`, error);
        if (!isDestroyed.value) {
            emit('show-error', `Erreur lors du chargement des données pour ${country}`);
        }
    }
}

function processData(data, format) {
    switch (format) {
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
        default:
            return data;
    }
}

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

function toggleDataset(datasetName) {
    if (isCoolingDown.value) return;
    
    activateCooldown(500);
    chartConfig.value.datasets[datasetName] = !chartConfig.value.datasets[datasetName];
    updateMultiCountryChart();
}

function updateColor({ datasetName, color }) {
    if (isCoolingDown.value) return;
    
    activateCooldown(500);
    chartConfig.value.colors[datasetName] = color;
    updateMultiCountryChart();
}

function updateChartType(type) {
    if (isCoolingDown.value) return;
    
    activateCooldown(800);
    chartType.value = type;
    updateMultiCountryChart();
}

function updateDataFormat(format) {
    if (isCoolingDown.value) return;
    
    activateCooldown(800);
    dataFormat.value = format;
    updateMultiCountryChart();
}

function updateScaleType(type) {
    if (isCoolingDown.value) return;
    
    activateCooldown(500);
    scaleType.value = type;
    updateMultiCountryChart();
}

function zoomIn() {
    if (isCoolingDown.value || !multiCountryChart.value) return;
    
    activateCooldown(300);
    multiCountryChart.value.zoom(1.2);
}

function zoomOut() {
    if (isCoolingDown.value || !multiCountryChart.value) return;
    
    activateCooldown(300);
    multiCountryChart.value.zoom(0.8);
}

function resetZoom() {
    if (isCoolingDown.value || !multiCountryChart.value) return;
    
    activateCooldown(300);
    multiCountryChart.value.resetZoom();
}

function downloadChart() {
    if (isCoolingDown.value || !multiCountryChart.value) return;
    
    activateCooldown(1000);
    const link = document.createElement('a');
    link.download = `covid-multi-country-comparison.png`;
    link.href = multiCountryChart.value.toBase64Image();
    link.click();
}

function exportData() {
    if (isCoolingDown.value) return;
    
    activateCooldown(1000);
    
    // Créer les en-têtes du CSV
    const headers = ['Date'];
    props.selectedCountries.forEach(country => {
        headers.push(`${country} - Confirmés`);
        headers.push(`${country} - Décès`);
        headers.push(`${country} - Guéris`);
        headers.push(`${country} - Actifs`);
    });
    
    // Trouver toutes les dates uniques de tous les pays
    const allDates = new Set();
    props.selectedCountries.forEach(country => {
        if (countriesData.value[country]) {
            countriesData.value[country].forEach(item => {
                allDates.add(item.date);
            });
        }
    });
    
    // Convertir en tableau et trier
    const sortedDates = [...allDates].sort((a, b) => new Date(a) - new Date(b));
    
    // Créer une carte date -> données pour chaque pays
    const countryDataByDate = {};
    props.selectedCountries.forEach(country => {
        if (!countriesData.value[country]) return;
        
        countryDataByDate[country] = {};
        countriesData.value[country].forEach(item => {
            countryDataByDate[country][item.date] = item;
        });
    });
    
    // Créer les lignes du CSV
    const rows = [headers];
    sortedDates.forEach(date => {
        const row = [new Date(date).toLocaleDateString()];
        
        props.selectedCountries.forEach(country => {
            const data = countryDataByDate[country][date];
            if (data) {
                row.push(data.confirmed);
                row.push(data.deaths);
                row.push(data.recovered);
                row.push(data.active);
            } else {
                // Pas de données pour ce pays à cette date
                row.push('');
                row.push('');
                row.push('');
                row.push('');
            }
        });
        
        rows.push(row);
    });
    
    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `covid-multi-country-data.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
}

// Fonction pour obtenir une version plus foncée d'une couleur pour les décès
function getDarkerColor(hexColor) {
    // Enlever le #
    const hex = hexColor.replace('#', '');
    
    // Convertir en RGB
    let r = parseInt(hex.substring(0, 2), 16);
    let g = parseInt(hex.substring(2, 4), 16);
    let b = parseInt(hex.substring(4, 6), 16);
    
    // Assombrir (multiplier par 0.7)
    r = Math.round(r * 0.7);
    g = Math.round(g * 0.7);
    b = Math.round(b * 0.7);
    
    // Reconvertir en hex
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

// Nettoyage des ressources lors du démontage du composant
onBeforeUnmount(() => {
    isDestroyed.value = true;
    
    if (multiCountryChart.value) {
        multiCountryChart.value.destroy();
        multiCountryChart.value = null;
    }
});

// Initialisation du graphique au montage du composant
onMounted(() => {
    updateMultiCountryChart();
});
</script>

<style scoped>
.countries-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 15px;
    justify-content: center;
}

.country-legend-item {
    padding: 5px 15px;
    border-left: 4px solid;
    background-color: #f8f9fa;
    border-radius: 4px;
    font-weight: bold;
    font-size: 14px;
}
</style>
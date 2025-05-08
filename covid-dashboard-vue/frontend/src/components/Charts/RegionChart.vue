<template>
    <div>
        <div class="options-panel">
            <div class="options-grid">
                <div class="option-group">
                    <label>Type de visualisation</label>
                    <select id="regionChartType" v-model="chartType" @change="updateRegionChart"
                        data-tooltip="Choisissez le type de graphique">
                        <option value="line">Ligne</option>
                        <option value="bar">Barres</option>
                        <option value="area">Aire</option>
                    </select>
                </div>

                <div class="option-group">
                    <label>Format des données</label>
                    <select id="regionDataFormat" v-model="dataFormat" @change="updateRegionChart"
                        data-tooltip="Choisissez le format d'affichage des données">
                        <option value="raw">Valeurs brutes</option>
                        <option value="daily">Variation quotidienne</option>
                    </select>
                </div>

                <div class="option-group">
                    <label>Échelle Y</label>
                    <select id="regionScaleType" v-model="scaleType" @change="updateRegionChart"
                        data-tooltip="Choisissez l'échelle de l'axe Y">
                        <option value="linear">Linéaire</option>
                        <option value="logarithmic">Logarithmique</option>
                    </select>
                </div>
            </div>

            <div class="options-grid" style="margin-top: 20px;">
                <div class="option-group">
                    <label>Régions visibles</label>
                    <div class="dataset-toggles" id="regionToggles">
                        <label v-for="region in regions" :key="region" class="toggle-item">
                            <input type="checkbox" :id="`toggle${region.replace(/\s+/g, '')}`"
                                :checked="regionConfig.datasets[region]" @change="toggleRegion(region)">
                            <span>{{ region }}</span>
                        </label>
                    </div>
                </div>

                <div class="option-group">
                    <label>Personnalisation des couleurs</label>
                    <div class="color-options" id="regionColors">
                        <div v-for="region in regions" :key="region" class="color-option">
                            <input type="color" :id="`color${region.replace(/\s+/g, '')}`"
                                :value="regionConfig.colors[region] || defaultRegionColors[region] || '#000000'"
                                @change="updateRegionColor(region, $event.target.value)">
                            <span>{{ region }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <chart-controls @zoom-in="zoomInRegion" @zoom-out="zoomOutRegion" @reset-zoom="resetRegionZoom"
            @download="downloadRegionChart" @export="exportRegionData" />

        <div class="chart-container">
            <canvas id="regionChart" ref="regionChart"></canvas>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartControls from './ChartControls.vue';
import axios from 'axios';

Chart.register(zoomPlugin);

// Émetteurs d'événements
const emit = defineEmits(['toggle-loading', 'show-error']);

// Références
const regionChart = ref(null);
const regionChartRef = ref(null);
const regionData = ref(null);
const regions = ref([]);

// Variables réactives
const regionConfig = ref({
    datasets: {},
    colors: {}
});

const defaultRegionColors = {
    'Europe': '#1a73e8',
    'Americas': '#dc3545',
    'Western Pacific': '#28a745',
    'Eastern Mediterranean': '#ffc107',
    'Africa': '#6f42c1',
    'South-East Asia': '#fd7e14'
};

const chartType = ref('line');
const dataFormat = ref('raw');
const scaleType = ref('linear');

// Méthodes
const updateRegionChart = async () => {
    try {
        emit('toggle-loading', true);

        // Récupération des données
        const response = await axios.get('/api/region-stats');
        let data = response.data;

        // Stocker les données pour l'export CSV
        regionData.value = data;

        // Obtenir la liste unique des régions
        regions.value = [...new Set(data.map(item => item.region_name))];

        // Initialiser regionConfig pour les nouvelles régions
        regions.value.forEach(region => {
            if (regionConfig.value.datasets[region] === undefined) {
                regionConfig.value.datasets[region] = true;
            }
            if (regionConfig.value.colors[region] === undefined) {
                regionConfig.value.colors[region] = defaultRegionColors[region] ||
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
        if (dataFormat.value === 'daily') {
            for (let i = 1; i < sortedDates.length; i++) {
                const currentDate = sortedDates[i];
                const previousDate = sortedDates[i - 1];

                regions.value.forEach(region => {
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
        regions.value.forEach(region => {
            if (regionConfig.value.datasets[region]) {
                const regionData = sortedDates.map(date => {
                    return groupedByDate[date][region] ? groupedByDate[date][region].confirmed : 0;
                });

                datasets.push({
                    label: region,
                    data: regionData,
                    borderColor: regionConfig.value.colors[region],
                    backgroundColor: chartType.value === 'bar' ?
                        `${regionConfig.value.colors[region]}88` :
                        `${regionConfig.value.colors[region]}22`,
                    fill: chartType.value === 'area',
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                });
            }
        });

        const ctx = regionChartRef.value.getContext('2d');

        if (regionChart.value) {
            regionChart.value.destroy();
        }

        // Créer le graphique
        regionChart.value = new Chart(ctx, {
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
                        },
                        onClick: (e, legendItem, legend) => {
                            const index = legendItem.datasetIndex;
                            const ci = legend.chart;
                            const meta = ci.getDatasetMeta(index);

                            meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                            ci.update();

                            // Mise à jour des toggles
                            const region = ci.data.datasets[index].label;
                            regionConfig.value.datasets[region] = !meta.hidden;

                            if (document.getElementById(`toggle${region.replace(/\s+/g, '')}`)) {
                                document.getElementById(`toggle${region.replace(/\s+/g, '')}`).checked = !meta.hidden;
                            }
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

    } catch (error) {
        console.error('Erreur:', error);
        emit('show-error', 'Erreur lors de la mise à jour du graphique régional');
    } finally {
        emit('toggle-loading', false);
    }
};

const getChartType = (selectedType) => {
    switch (selectedType) {
        case 'area':
            return 'line';
        case 'mixed':
            return 'bar';
        default:
            return selectedType;
    }
};

const toggleRegion = (region) => {
    regionConfig.value.datasets[region] = !regionConfig.value.datasets[region];
    document.getElementById(`toggle${region.replace(/\s+/g, '')}`).checked = regionConfig.value.datasets[region];
    updateRegionChart();
};

const updateRegionColor = (region, color) => {
    regionConfig.value.colors[region] = color;
    updateRegionChart();
};

const zoomInRegion = () => {
    if (regionChart.value) {
        regionChart.value.zoom(1.2); // Zoom de 20%
    }
};

const zoomOutRegion = () => {
    if (regionChart.value) {
        regionChart.value.zoom(0.8); // Zoom out de 20%
    }
};

const resetRegionZoom = () => {
    if (regionChart.value) {
        regionChart.value.resetZoom();
    }
};

const downloadRegionChart = () => {
    if (regionChart.value) {
        const link = document.createElement('a');
        link.download = 'region-chart.png';
        link.href = regionChart.value.toBase64Image();
        link.click();
    }
};

const exportRegionData = () => {
    if (!regionData.value) return;

    const rows = [['Date', 'Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']];
    regionData.value.forEach(item => {
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
};

const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
};

// Cycle de vie du composant
onMounted(() => {
    updateRegionChart();
});
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
<template>
    <div>
        <div class="options-panel">
            <div class="options-grid">
                <div class="option-group">
                    <label>Type de visualisation</label>
                    <select id="regionChartType" v-model="chartType" @change="updateRegionChart"
                        data-tooltip="Choisissez le type de graphique" :disabled="isCoolingDown">
                        <option value="line">Ligne</option>
                        <option value="bar">Barres</option>
                        <option value="area">Aire</option>
                    </select>
                </div>

                <div class="option-group">
                    <label>Format des données</label>
                    <select id="regionDataFormat" v-model="dataFormat" @change="updateRegionChart"
                        data-tooltip="Choisissez le format d'affichage des données" :disabled="isCoolingDown">
                        <option value="raw">Valeurs brutes</option>
                        <option value="daily">Variation quotidienne</option>
                    </select>
                </div>

                <div class="option-group">
                    <label>Échelle Y</label>
                    <select id="regionScaleType" v-model="scaleType" @change="updateRegionChart"
                        data-tooltip="L'échelle logarithmique montre mieux les taux de croissance" :disabled="isCoolingDown">
                        <option value="linear">Linéaire (standard)</option>
                        <option value="logarithmic">Logarithmique (puissances de 10)</option>
                    </select>
                </div>
            </div>

            <div class="options-grid" style="margin-top: 20px;">
                <div class="option-group">
                    <label>Régions visibles</label>
                    <div class="dataset-toggles" id="regionToggles">
                        <label v-for="region in regions" :key="region" class="toggle-item"
                            :class="{ 'disabled': isCoolingDown }">
                            <input type="checkbox" :id="`toggle${region.replace(/\s+/g, '')}`"
                                :checked="regionConfig.datasets[region]" @change="toggleRegion(region)"
                                :disabled="isCoolingDown">
                            <span>{{ region }}</span>
                        </label>
                    </div>
                </div>

                <div class="option-group">
                    <label>Personnalisation des couleurs</label>
                    <div class="color-options" id="regionColors">
                        <div v-for="region in regions" :key="region" class="color-option"
                            :class="{ 'disabled': isCoolingDown }">
                            <input type="color" :id="`color${region.replace(/\s+/g, '')}`"
                                :value="regionConfig.colors[region] || defaultRegionColors[region] || '#000000'"
                                @change="updateRegionColor(region, $event.target.value)" :disabled="isCoolingDown">
                            <span>{{ region }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <chart-controls @zoom-in="zoomInRegion" @zoom-out="zoomOutRegion" @reset-zoom="resetRegionZoom"
            @download="downloadRegionChart" @export="exportRegionData" />

        <div class="chart-container">
            <canvas id="regionChart" ref="regionChartRef"></canvas>
        </div>

        <!-- Mini Loader -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartControls from './ChartControlsResponsive.vue';
import axios from 'axios';
import MiniLoader from '../MiniLoader.vue';

Chart.register(zoomPlugin);

// Émetteurs d'événements
const emit = defineEmits(['toggle-loading', 'show-error']);

// Références
const regionChart = ref(null);
const regionChartRef = ref(null);
const regionData = ref(null);
const regions = ref([]);
const isCoolingDown = ref(false);
const isDestroyed = ref(false);

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

// Fonction pour activer le cooldown
function activateCooldown(duration = 800) {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, duration);
}

// Méthodes avec cooldown
const updateRegionChart = async () => {
    if (isDestroyed.value) return;

    try {
        emit('toggle-loading', true);

        // Récupération des données
        const response = await axios.get('/api/region-stats');
        let data = response.data;

        if (isDestroyed.value) return;

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
                    '#' + Math.floor(Math.random() * 16777215).toString(16);
            }
        });

        if (isDestroyed.value) return;

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
            // Créer une copie des données pour les variations quotidiennes
            // afin de ne pas modifier les données originales
            const dailyData = JSON.parse(JSON.stringify(data));

            // Créer un nouveau regroupement pour les données quotidiennes
            let dailyGroupedByDate = {};

            // Préparer la structure avec les mêmes dates que les données d'origine
            sortedDates.forEach(date => {
                dailyGroupedByDate[date] = {};
                regions.value.forEach(region => {
                    // Initialiser chaque région pour cette date
                    dailyGroupedByDate[date][region] = null;
                });
            });

            // Calculer les variations quotidiennes
            regions.value.forEach(region => {
                for (let i = 1; i < sortedDates.length; i++) {
                    const currentDate = sortedDates[i];
                    const previousDate = sortedDates[i - 1];

                    const current = groupedByDate[currentDate] ? groupedByDate[currentDate][region] : null;
                    const previous = groupedByDate[previousDate] ? groupedByDate[previousDate][region] : null;

                    if (current && previous) {
                        // Créer un nouvel objet pour cette région et cette date
                        dailyGroupedByDate[currentDate][region] = {
                            confirmed: Math.max(0, current.confirmed - previous.confirmed), // Éviter les valeurs négatives
                            deaths: Math.max(0, current.deaths - previous.deaths),
                            recovered: Math.max(0, current.recovered - previous.recovered),
                            active: current.active - previous.active // Peut être négatif légitimement
                        };
                    } else if (current && !previous) {
                        // Si on a des données aujourd'hui mais pas hier, on prend les valeurs d'aujourd'hui
                        dailyGroupedByDate[currentDate][region] = {
                            confirmed: current.confirmed,
                            deaths: current.deaths,
                            recovered: current.recovered,
                            active: current.active
                        };
                    } else {
                        // Pas de données pour cette région à cette date
                        dailyGroupedByDate[currentDate][region] = {
                            confirmed: 0,
                            deaths: 0,
                            recovered: 0,
                            active: 0
                        };
                    }
                }

                // Traiter spécialement la première date
                const firstDate = sortedDates[0];
                if (groupedByDate[firstDate] && groupedByDate[firstDate][region]) {
                    dailyGroupedByDate[firstDate][region] = {
                        confirmed: groupedByDate[firstDate][region].confirmed,
                        deaths: groupedByDate[firstDate][region].deaths,
                        recovered: groupedByDate[firstDate][region].recovered,
                        active: groupedByDate[firstDate][region].active
                    };
                } else {
                    dailyGroupedByDate[firstDate][region] = {
                        confirmed: 0,
                        deaths: 0,
                        recovered: 0,
                        active: 0
                    };
                }
            });

            // Utiliser le nouveau regroupement pour le graphique
            groupedByDate = dailyGroupedByDate;
        }

        if (isDestroyed.value) return;

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

        // Vérifier si la référence existe
        if (!regionChartRef.value) {
            console.error("La référence regionChartRef est null");
            emit('show-error', 'Erreur lors de la mise à jour du graphique régional - référence manquante');
            return;
        }

        if (isDestroyed.value) return;

        const ctx = regionChartRef.value.getContext('2d');

        if (regionChart.value) {
            regionChart.value.destroy();
        }

        if (isDestroyed.value) return;

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
                        onClick: function (e, legendItem, legend) {
                            // Récupérer la région/légende
                            const region = legendItem.text;

                            // Mettre à jour la configuration
                            regionConfig.value.datasets[region] = !regionConfig.value.datasets[region];

                            // Détruire et recréer le graphique
                            regionChart.value.destroy();
                            regionChart.value = null;

                            // Recréer le graphique avec la nouvelle configuration
                            setTimeout(() => {
                                updateRegionChart();
                            }, 50);
                        }
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
                        type: scaleType.value,
                        beginAtZero: scaleType.value === 'linear',
                        min: scaleType.value === 'logarithmic' ? 1 : 0, // Valeur minimale pour l'échelle log
                        grid: {
                            color: function (context) {
                                // Lignes de grille plus foncées pour les puissances de 10 en échelle log
                                if (scaleType.value === 'logarithmic') {
                                    const value = context.tick.value;
                                    if (value === 1 || value === 10 || value === 100 ||
                                        value === 1000 || value === 10000 || value === 100000 ||
                                        value === 1000000 || value === 10000000) {
                                        return 'rgba(0, 0, 0, 0.2)';
                                    }
                                    return 'rgba(0, 0, 0, 0.05)';
                                }
                                return 'rgba(0, 0, 0, 0.1)';
                            }
                        },
                        ticks: {
                            callback: function (value) {
                                // Formatage spécial pour l'échelle logarithmique
                                if (scaleType.value === 'logarithmic') {
                                    if (value === 1 || value === 10 || value === 100 ||
                                        value === 1000 || value === 10000 || value === 100000 ||
                                        value === 1000000 || value === 10000000) {
                                        return formatNumber(value);
                                    }
                                    return '';
                                } else {
                                    // Formatage standard pour l'échelle linéaire
                                    if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                    if (value >= 1000) return (value / 1000).toFixed(0) + 'k';
                                    return value;
                                }
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
            }
        });

    } catch (error) {
        console.error('Erreur:', error);
        if (!isDestroyed.value) {
            emit('show-error', 'Erreur lors de la mise à jour du graphique régional');
        }
    } finally {
        if (!isDestroyed.value) {
            emit('toggle-loading', false);
        }
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

// SOLUTION FONCTIONNELLE : RECRÉER COMPLÈTEMENT LE GRAPHIQUE À CHAQUE CHANGEMENT
const toggleRegion = (region) => {
    if (isCoolingDown.value) return;
    activateCooldown(500);

    // Inverser la valeur
    regionConfig.value.datasets[region] = !regionConfig.value.datasets[region];

    // Détruire le graphique existant
    if (regionChart.value) {
        regionChart.value.destroy();
        regionChart.value = null;
    }

    // Attendre un peu puis recréer complètement le graphique
    setTimeout(() => {
        updateRegionChart();
    }, 50);
};

// SOLUTION FONCTIONNELLE : RECRÉER COMPLÈTEMENT LE GRAPHIQUE À CHAQUE CHANGEMENT
const updateRegionColor = (region, color) => {
    if (isCoolingDown.value) return;
    activateCooldown(500);

    // Mettre à jour la couleur
    regionConfig.value.colors[region] = color;

    // Détruire le graphique existant
    if (regionChart.value) {
        regionChart.value.destroy();
        regionChart.value = null;
    }

    // Attendre un peu puis recréer complètement le graphique
    setTimeout(() => {
        updateRegionChart();
    }, 50);
};

const zoomInRegion = () => {
    if (isCoolingDown.value || !regionChart.value) return;
    activateCooldown(300);

    const centerX = regionChart.value.chartArea.width / 2;
    const centerY = regionChart.value.chartArea.height / 2;

    regionChart.value.zoom(1.2, 'xy', { x: centerX, y: centerY });
};

const zoomOutRegion = () => {
    if (isCoolingDown.value || !regionChart.value) return;
    activateCooldown(300);

    const centerX = regionChart.value.chartArea.width / 2;
    const centerY = regionChart.value.chartArea.height / 2;

    regionChart.value.zoom(0.8, 'xy', { x: centerX, y: centerY });
};

const resetRegionZoom = () => {
    if (isCoolingDown.value || !regionChart.value) return;
    activateCooldown(300);

    regionChart.value.resetZoom();
};

const downloadRegionChart = () => {
    if (isCoolingDown.value || !regionChart.value) return;
    activateCooldown(1000);

    try {
        const link = document.createElement('a');
        link.download = 'region-chart.png';
        link.href = regionChart.value.toBase64Image();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        console.error('Erreur lors du téléchargement:', error);
        emit('show-error', 'Erreur lors du téléchargement du graphique');
    }
};

const exportRegionData = () => {
    if (isCoolingDown.value || !regionData.value) return;
    activateCooldown(1000);

    try {
        const rows = [['Date', 'Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate']];

        regionData.value.forEach(item => {
            rows.push([
                new Date(item.date).toLocaleDateString(),
                item.region_name,
                item.confirmed,
                item.deaths,
                item.recovered,
                item.active,
                item.mortality_rate ? item.mortality_rate.toFixed(2) + '%' : '0%'
            ]);
        });

        const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "region_data.csv");
        document.body.appendChild(link);
        link.click();

        // Ajouter un délai avant de supprimer le lien
        setTimeout(() => {
            document.body.removeChild(link);
        }, 100);
    } catch (error) {
        console.error('Erreur lors de l\'exportation:', error);
        emit('show-error', 'Erreur lors de l\'exportation des données');
    }
};

const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
};

// Nettoyage des ressources lors du démontage du composant
onBeforeUnmount(() => {
    isDestroyed.value = true;

    if (regionChart.value) {
        regionChart.value.destroy();
        regionChart.value = null;
    }
});

// Cycle de vie du composant
onMounted(() => {
    updateRegionChart();
});
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
}
</style>
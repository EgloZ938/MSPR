<template>
    <div>
        <ChartOptions :chartType="chartType" :dataFormat="dataFormat" :datasets="chartConfig.datasets"
            :colors="chartConfig.colors" :is-country-view="true" :show-pie-options="true"
            @toggle-dataset="toggleDataset" @update-color="updateColor" @chart-type-change="updateChartType"
            @data-format-change="updateDataFormat" />

        <ChartControls @zoom-in="zoomInCountry" @zoom-out="zoomOutCountry" @reset-zoom="resetCountryZoom"
            @download="downloadCountryChart" @export="exportCountryData" />

        <div class="visualization-row">
            <div class="chart-container">
                <canvas id="countryChart" ref="countryChartRef"></canvas>
            </div>

            <!-- Section pour le classement -->
            <div class="ranking-container">
                <h3>Classement des pays</h3>
                <div class="ranking-tabs">
                    <button class="ranking-tab"
                        :class="{ active: activeRankingTab === 'confirmed', disabled: isCoolingDown }"
                        @click="handleUpdateRanking('confirmed')" :disabled="isCoolingDown">Cas confirmés</button>
                    <button class="ranking-tab"
                        :class="{ active: activeRankingTab === 'deaths', disabled: isCoolingDown }"
                        @click="handleUpdateRanking('deaths')" :disabled="isCoolingDown">Décès</button>
                    <button class="ranking-tab"
                        :class="{ active: activeRankingTab === 'mortality', disabled: isCoolingDown }"
                        @click="handleUpdateRanking('mortality')" :disabled="isCoolingDown">Taux de mortalité</button>
                </div>
                <div class="ranking-list" id="countryRanking">
                    <div v-for="(country, index) in rankings" :key="country.country_region" class="ranking-item"
                        :style="country.country_region === selectedCountry ? 'background-color: rgba(26, 115, 232, 0.1); font-weight: bold;' : ''"
                        :class="{ 'disabled': isCoolingDown }"
                        @click="handleSelectCountryFromRanking(country.country_region)">
                        <span class="rank">{{ index + 1 }}</span>
                        <span class="country">{{ country.country_region }}</span>
                        <span class="value">{{ formatRankingValue(country) }}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mini Loader pour indiquer visuellement le cooldown -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref, watch, onMounted, onBeforeUnmount, defineProps, defineEmits } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartOptions from './ChartOptions.vue';
import ChartControls from './ChartControlsResponsive.vue';
import MiniLoader from '../MiniLoader.vue';
import axios from 'axios';

Chart.register(zoomPlugin);

const props = defineProps({
    selectedCountry: {
        type: String,
        required: true
    }
});

const emit = defineEmits(['toggle-loading', 'show-error', 'update-stats', 'country-changed']);

const countryChart = ref(null);
const countryChartRef = ref(null);
const countryData = ref(null);
const latestStats = ref(null);
const isCoolingDown = ref(false);
const isDestroyed = ref(false);

const chartConfig = ref({
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
});
const chartType = ref('line');
const dataFormat = ref('raw');
const rankings = ref([]);
const activeRankingTab = ref('confirmed');

function activateCooldown(duration = 800) {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, duration);
}

watch(() => props.selectedCountry, () => {
    if (!isDestroyed.value) {
        updateCountryChart();
    }
}, { immediate: true });

async function updateCountryChart() {
    if (isDestroyed.value) return;

    try {
        emit('toggle-loading', true);

        await loadCountryStats();

        if (!countryData.value || countryData.value.length === 0) {
            emit('show-error', `Aucune donnée disponible pour ${props.selectedCountry}`);
            return;
        }

        if (isDestroyed.value) return;

        let processedData = countryData.value;
        if (dataFormat.value === 'daily') {
            // Créer un nouveau tableau pour les variations quotidiennes
            processedData = [];

            // Traiter chaque jour
            countryData.value.forEach((item, index) => {
                if (index === 0) {
                    // Premier jour : garder les valeurs d'origine
                    processedData.push({ ...item });
                } else {
                    // Jours suivants : calculer la différence avec le jour précédent
                    const previous = countryData.value[index - 1];
                    processedData.push({
                        ...item,
                        confirmed: Math.max(0, item.confirmed - previous.confirmed), // Éviter les valeurs négatives
                        deaths: Math.max(0, item.deaths - previous.deaths),
                        recovered: Math.max(0, item.recovered - previous.recovered),
                        active: item.active - previous.active, // Peut être négatif légitimement
                        // Recalculer le taux de mortalité pour la variation quotidienne
                        mortality_rate: item.deaths && item.confirmed ?
                            ((item.deaths - previous.deaths) / (item.confirmed - previous.confirmed) * 100) : 0
                    });
                }
            });
        }

        // Vérifier si la référence existe
        if (!countryChartRef.value) {
            console.error("La référence countryChartRef est null");
            emit('show-error', 'Erreur lors de la mise à jour du graphique - référence manquante');
            return;
        }

        if (isDestroyed.value) return;

        const ctx = countryChartRef.value.getContext('2d');

        if (countryChart.value) {
            countryChart.value.destroy();
        }

        if (isDestroyed.value) return;

        if (chartType.value === 'pie' || chartType.value === 'doughnut') {
            const latestData = processedData[processedData.length - 1];
            const labels = [];
            const data = [];
            const backgroundColors = [];

            if (chartConfig.value.datasets.confirmed) {
                labels.push('Cas confirmés');
                data.push(latestData.confirmed);
                backgroundColors.push(chartConfig.value.colors.confirmed);
            }
            if (chartConfig.value.datasets.deaths) {
                labels.push('Décès');
                data.push(latestData.deaths);
                backgroundColors.push(chartConfig.value.colors.deaths);
            }
            if (chartConfig.value.datasets.recovered) {
                labels.push('Guéris');
                data.push(latestData.recovered);
                backgroundColors.push(chartConfig.value.colors.recovered);
            }
            if (chartConfig.value.datasets.active) {
                labels.push('Cas actifs');
                data.push(latestData.active);
                backgroundColors.push(chartConfig.value.colors.active);
            }

            countryChart.value = new Chart(ctx, {
                type: chartType.value,
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
                                label: (context) => {
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
            const datasets = [];

            if (chartConfig.value.datasets.confirmed) {
                datasets.push(createDataset('Cas confirmés', processedData.map(item => item.confirmed), chartConfig.value.colors.confirmed));
            }
            if (chartConfig.value.datasets.deaths) {
                datasets.push(createDataset('Décès', processedData.map(item => item.deaths), chartConfig.value.colors.deaths));
            }
            if (chartConfig.value.datasets.recovered) {
                datasets.push(createDataset('Guéris', processedData.map(item => item.recovered), chartConfig.value.colors.recovered));
            }
            if (chartConfig.value.datasets.active) {
                datasets.push(createDataset('Cas actifs', processedData.map(item => item.active), chartConfig.value.colors.active));
            }

            countryChart.value = new Chart(ctx, {
                type: getChartType(chartType.value),
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
                            onClick: (e, legendItem, legend) => {
                                if (isCoolingDown.value) return;

                                const index = legendItem.datasetIndex;
                                const ci = legend.chart;
                                const meta = ci.getDatasetMeta(index);

                                meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                                ci.update();

                                const datasetId = ci.data.datasets[index].id || legendItem.text.toLowerCase().replace(/\s+/g, '');
                                const key = Object.keys(chartConfig.value.datasets).find(
                                    k => datasetId.includes(k.toLowerCase())
                                );
                                if (key) {
                                    chartConfig.value.datasets[key] = !meta.hidden;
                                }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: (context) => {
                                    return `${context.dataset.label}: ${formatNumber(context.raw)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            type: scaleType ? scaleType.value : 'linear', // Assurez-vous que scaleType existe
                            beginAtZero: !scaleType || scaleType.value === 'linear',
                            min: scaleType && scaleType.value === 'logarithmic' ? 1 : 0, // Valeur minimale pour l'échelle log
                            grid: {
                                color: function (context) {
                                    // Lignes de grille plus foncées pour les puissances de 10 en échelle log
                                    if (scaleType && scaleType.value === 'logarithmic') {
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
                                    if (scaleType && scaleType.value === 'logarithmic') {
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
        }

        if (isDestroyed.value) return;

        await loadCountryRanking(activeRankingTab.value);

    } catch (error) {
        console.error('Erreur lors de la mise à jour du graphique du pays:', error);
        if (!isDestroyed.value) {
            emit('show-error', `Erreur lors de la mise à jour du graphique pour ${props.selectedCountry}`);
        }
    } finally {
        if (!isDestroyed.value) {
            emit('toggle-loading', false);
        }
    }
}

async function loadCountryStats() {
    try {
        const response = await axios.get(`/api/country-timeline/${props.selectedCountry}`);
        countryData.value = response.data;

        if (countryData.value.length > 0 && !isDestroyed.value) {
            latestStats.value = countryData.value[countryData.value.length - 1];
            emit('update-stats', {
                confirmed: latestStats.value.confirmed,
                deaths: latestStats.value.deaths,
                recovered: latestStats.value.recovered,
                active: latestStats.value.active,
                mortalityRate: latestStats.value.mortality_rate
            });
        }
    } catch (error) {
        console.error('Erreur lors du chargement des statistiques du pays:', error);
        if (!isDestroyed.value) {
            emit('show-error', `Erreur lors du chargement des statistiques pour ${props.selectedCountry}`);
        }
    }
}

async function loadCountryRanking(metric) {
    try {
        activeRankingTab.value = metric;

        const response = await axios.get('/api/top-countries');
        if (isDestroyed.value) return;

        let countries = response.data;

        if (metric === 'confirmed') {
            countries = countries.sort((a, b) => b.confirmed - a.confirmed);
        } else if (metric === 'deaths') {
            countries = countries.sort((a, b) => b.deaths - a.deaths);
        } else if (metric === 'mortality') {
            countries = countries.sort((a, b) => b.mortality_rate - a.mortality_rate);
        }

        rankings.value = countries.slice(0, 20);
    } catch (error) {
        console.error('Erreur lors du chargement du classement:', error);
        if (!isDestroyed.value) {
            emit('show-error', 'Erreur lors du chargement du classement des pays');
        }
    }
}

function createDataset(label, data, color) {
    return {
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: chartType.value === 'bar' ? `${color}88` : `${color}22`,
        fill: chartType.value === 'area',
        tension: 0.4,
        id: label.toLowerCase().replace(/\s+/g, ''),
        pointRadius: chartType.value === 'scatter' ? 4 : 0,
        pointHoverRadius: 6
    };
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
    updateCountryChart();
}

function updateColor({ datasetName, color }) {
    if (isCoolingDown.value) return;

    activateCooldown(500);
    chartConfig.value.colors[datasetName] = color;
    updateCountryChart();
}

function updateChartType(type) {
    if (isCoolingDown.value) return;

    activateCooldown(800);
    chartType.value = type;
    updateCountryChart();
}

function updateDataFormat(format) {
    if (isCoolingDown.value) return;

    activateCooldown(800);
    dataFormat.value = format;
    updateCountryChart();
}

function handleUpdateRanking(metric) {
    if (isCoolingDown.value) return;

    activateCooldown(600);
    updateRanking(metric);
}

function handleSelectCountryFromRanking(countryName) {
    if (isCoolingDown.value) return;

    activateCooldown(1200); // Plus long car le chargement d'un nouveau pays est une opération lourde
    emit('country-changed', countryName);
}

function updateRanking(metric) {
    loadCountryRanking(metric);
}

function zoomInCountry() {
    if (isCoolingDown.value || !countryChart.value) return;

    activateCooldown(300);
    const zoomOptions = countryChart.value.options.plugins.zoom.zoom;
    zoomOptions.wheel.enabled = false;

    const centerX = countryChart.value.chartArea.width / 2;
    const centerY = countryChart.value.chartArea.height / 2;
    countryChart.value.pan({ x: 0, y: 0 }, 'none', 'default');
    countryChart.value.zoom(1.2, 'xy', { x: centerX, y: centerY });
}

function zoomOutCountry() {
    if (isCoolingDown.value || !countryChart.value) return;

    activateCooldown(300);
    const zoomOptions = countryChart.value.options.plugins.zoom.zoom;
    zoomOptions.wheel.enabled = false;

    const centerX = countryChart.value.chartArea.width / 2;
    const centerY = countryChart.value.chartArea.height / 2;
    countryChart.value.pan({ x: 0, y: 0 }, 'none', 'default');
    countryChart.value.zoom(0.8, 'xy', { x: centerX, y: centerY });
}

function resetCountryZoom() {
    if (isCoolingDown.value || !countryChart.value) return;

    activateCooldown(300);
    countryChart.value.resetZoom();
}

function downloadCountryChart() {
    if (isCoolingDown.value || !countryChart.value) return;

    activateCooldown(1000);
    const link = document.createElement('a');
    link.download = `covid-${props.selectedCountry.toLowerCase()}-chart.png`;
    link.href = countryChart.value.toBase64Image();
    link.click();
}

function exportCountryData() {
    if (isCoolingDown.value || !countryData.value) {
        console.error("Export impossible:", isCoolingDown.value ? "Cooldown actif" : "Données non disponibles");
        return;
    }

    try {
        activateCooldown(1000);
        console.log("Exportation des données pour", props.selectedCountry);

        const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate']];
        countryData.value.forEach(item => {
            rows.push([
                new Date(item.date).toLocaleDateString(),
                item.confirmed,
                item.deaths,
                item.recovered,
                item.active,
                item.mortality_rate ? item.mortality_rate.toFixed(2) : '0.00'
            ]);
        });

        const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `covid-${props.selectedCountry.toLowerCase()}-data.csv`);
        document.body.appendChild(link);
        link.click();
        setTimeout(() => {
            document.body.removeChild(link);
        }, 100);
    } catch (error) {
        console.error("Erreur lors de l'exportation des données:", error);
        emit('show-error', "Erreur lors de l'exportation des données");
    }
}

function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
}

function formatRankingValue(country) {
    if (activeRankingTab.value === 'confirmed') {
        return formatNumber(country.confirmed);
    } else if (activeRankingTab.value === 'deaths') {
        return formatNumber(country.deaths);
    } else if (activeRankingTab.value === 'mortality') {
        return `${country.mortality_rate.toFixed(2)}%`;
    }
    return '';
}

// Nettoyage des ressources lors du démontage du composant
onBeforeUnmount(() => {
    isDestroyed.value = true;

    if (countryChart.value) {
        countryChart.value.destroy();
        countryChart.value = null;
    }
});
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
    opacity: 0.6;
    cursor: not-allowed;
    pointer-events: none;
}
</style>
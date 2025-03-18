<template>
    <div>
        <ChartOptions :chartType="chartType" :dataFormat="dataFormat" :datasets="chartConfig.datasets" :colors="chartConfig.colors" :is-country-view="true" :show-pie-options="true" @toggle-dataset="toggleDataset" @update-color="updateColor" @chart-type-change="updateChartType" @data-format-change="updateDataFormat" />

        <ChartControls @zoom-in="zoomInCountry" @zoom-out="zoomOutCountry" @reset-zoom="resetCountryZoom" @download="downloadCountryChart" @export="exportCountryData" />

        <div class="visualization-row">
            <div class="chart-container">
                <canvas id="countryChart" ref="countryChart"></canvas>
            </div>

            <!-- Section pour le classement -->
            <div class="ranking-container">
                <h3>Classement des pays</h3>
                <div class="ranking-tabs">
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'confirmed' }" @click="updateRanking('confirmed')">Cas confirmés</button>
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'deaths' }" @click="updateRanking('deaths')">Décès</button>
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'mortality' }" @click="updateRanking('mortality')">Taux de mortalité</button>
                </div>
                <div class="ranking-list" id="countryRanking">
                    <div v-for="(country, index) in rankings" :key="country.country_region" class="ranking-item" :style="country.country_region === selectedCountry ? 'background-color: rgba(26, 115, 232, 0.1); font-weight: bold;' : ''" @click="selectCountryFromRanking(country.country_region)">
                        <span class="rank">{{ index + 1 }}</span>
                        <span class="country">{{ country.country_region }}</span>
                        <span class="value">{{ formatRankingValue(country) }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, onMounted, defineProps, defineEmits } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartOptions from './ChartOptions.vue';
import ChartControls from './ChartControls.vue';
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
const countryData = ref(null);
const latestStats = ref(null);
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

watch(() => props.selectedCountry, () => {
    updateCountryChart();
}, { immediate: true });

async function updateCountryChart() {
    try {
        emit('toggle-loading', true);

        await loadCountryStats();

        if (!countryData.value || countryData.value.length === 0) {
            emit('show-error', `Aucune donnée disponible pour ${props.selectedCountry}`);
            return;
        }

        let processedData = countryData.value;
        if (dataFormat.value === 'daily') {
            processedData = countryData.value.map((item, index) => {
                if (index === 0) return item;
                return {
                    ...item,
                    confirmed: item.confirmed - countryData.value[index - 1].confirmed,
                    deaths: item.deaths - countryData.value[index - 1].deaths,
                    recovered: item.recovered - countryData.value[index - 1].recovered,
                    active: item.active - countryData.value[index - 1].active
                };
            });
        }

        const ctx = document.getElementById('countryChart').getContext('2d');

        if (countryChart.value) {
            countryChart.value.destroy();
        }

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
                            beginAtZero: true,
                            ticks: {
                                callback: (value) => {
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

        await loadCountryRanking(activeRankingTab.value);

    } catch (error) {
        console.error('Erreur lors de la mise à jour du graphique du pays:', error);
        emit('show-error', `Erreur lors de la mise à jour du graphique pour ${props.selectedCountry}`);
    } finally {
        emit('toggle-loading', false);
    }
}

async function loadCountryStats() {
    try {
        const response = await axios.get(`/api/country-timeline/${props.selectedCountry}`);
        countryData.value = response.data;

        if (countryData.value.length > 0) {
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
        emit('show-error', `Erreur lors du chargement des statistiques pour ${props.selectedCountry}`);
    }
}

async function loadCountryRanking(metric) {
    try {
        activeRankingTab.value = metric;

        const response = await axios.get('/api/top-countries');
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
        emit('show-error', 'Erreur lors du chargement du classement des pays');
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
    chartConfig.value.datasets[datasetName] = !chartConfig.value.datasets[datasetName];
    updateCountryChart();
}

function updateColor({ datasetName, color }) {
    chartConfig.value.colors[datasetName] = color;
    updateCountryChart();
}

function updateChartType(type) {
    chartType.value = type;
    updateCountryChart();
}

function updateDataFormat(format) {
    dataFormat.value = format;
    updateCountryChart();
}

function updateRanking(metric) {
    loadCountryRanking(metric);
}

function selectCountryFromRanking(countryName) {
    emit('country-changed', countryName);
}

function zoomInCountry() {
    if (countryChart.value) {
        const zoomOptions = countryChart.value.options.plugins.zoom.zoom;
        zoomOptions.wheel.enabled = false;

        const centerX = countryChart.value.chartArea.width / 2;
        const centerY = countryChart.value.chartArea.height / 2;
        countryChart.value.pan({ x: 0, y: 0 }, 'none', 'default');
        countryChart.value.zoom(1.2, 'xy', { x: centerX, y: centerY });
    }
}

function zoomOutCountry() {
    if (countryChart.value) {
        const zoomOptions = countryChart.value.options.plugins.zoom.zoom;
        zoomOptions.wheel.enabled = false;

        const centerX = countryChart.value.chartArea.width / 2;
        const centerY = countryChart.value.chartArea.height / 2;
        countryChart.value.pan({ x: 0, y: 0 }, 'none', 'default');
        countryChart.value.zoom(0.8, 'xy', { x: centerX, y: centerY });
    }
}

function resetCountryZoom() {
    if (countryChart.value) {
        countryChart.value.resetZoom();
    }
}

function downloadCountryChart() {
    if (countryChart.value) {
        const link = document.createElement('a');
        link.download = `covid-${props.selectedCountry.toLowerCase()}-chart.png`;
        link.href = countryChart.value.toBase64Image();
        link.click();
    }
}

function exportCountryData() {
    if (!countryData.value) return;

    const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate']];
    countryData.value.forEach(item => {
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
    link.setAttribute("download", `covid-${props.selectedCountry.toLowerCase()}-data.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>

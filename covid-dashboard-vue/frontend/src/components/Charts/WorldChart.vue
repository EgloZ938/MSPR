<template>
    <div>
        <chart-options :chartType="chartType" :dataFormat="dataFormat" :scaleType="scaleType"
            :datasets="chartConfig.datasets" :colors="chartConfig.colors" @toggle-dataset="toggleDataset"
            @update-color="updateColor" @chart-type-change="updateChartType" @data-format-change="updateDataFormat"
            @scale-type-change="updateScaleType" />

        <chart-controls @zoom-in="zoomIn" @zoom-out="zoomOut" @reset-zoom="resetZoom" @download="downloadChart"
            @export="exportData" />

        <div class="chart-container">
            <canvas id="worldChart" ref="worldChartRef"></canvas>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartOptions from './ChartOptions.vue';
import ChartControls from './ChartControls.vue';
import axios from 'axios';

Chart.register(zoomPlugin);

// Émetteurs d'événements
const emit = defineEmits(['toggle-loading', 'show-error']);

// Références
const worldChart = ref(null);
const worldChartRef = ref(null);
const chartData = ref(null);

// Variables réactives
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
const scaleType = ref('linear');

// Méthodes
const updateChart = async () => {
    try {
        emit('toggle-loading', true);
        const response = await axios.get('/api/global-timeline');
        let data = response.data;

        // Stocker les données pour l'export CSV
        chartData.value = data;

        // Traitement des données selon le format
        data = processData(data, dataFormat.value);

        // Vérifier si la référence existe
        if (!worldChartRef.value) {
            console.error("La référence worldChartRef est null");
            emit('show-error', 'Erreur lors de la mise à jour du graphique - référence manquante');
            return;
        }

        const ctx = worldChartRef.value.getContext('2d');

        if (worldChart.value) {
            worldChart.value.destroy();
        }

        const datasets = [];

        // Création des datasets en fonction des sélections
        if (chartConfig.value.datasets.confirmed) {
            datasets.push(createDataset('Cas confirmés', data.map(item => item.confirmed), chartConfig.value.colors.confirmed));
        }
        if (chartConfig.value.datasets.deaths) {
            datasets.push(createDataset('Décès', data.map(item => item.deaths), chartConfig.value.colors.deaths));
        }
        if (chartConfig.value.datasets.recovered) {
            datasets.push(createDataset('Guéris', data.map(item => item.recovered), chartConfig.value.colors.recovered));
        }
        if (chartConfig.value.datasets.active) {
            datasets.push(createDataset('Cas actifs', data.map(item => item.active), chartConfig.value.colors.active));
        }

        worldChart.value = new Chart(ctx, {
            type: getChartType(chartType.value),
            data: {
                labels: data.map(item => new Date(item.date).toLocaleDateString()),
                datasets: datasets
            },
            options: getChartOptions()
        });
    } catch (error) {
        console.error('Erreur:', error);
        emit('show-error', 'Erreur lors de la mise à jour du graphique');
    } finally {
        emit('toggle-loading', false);
    }
};

const createDataset = (label, data, color) => {
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
};

const getChartOptions = () => {
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

                    // Mise à jour des checkboxes
                    const datasetId = ci.data.datasets[index].id;
                    const key = Object.keys(chartConfig.value.datasets).find(
                        k => k.toLowerCase() === datasetId
                    );
                    if (key) {
                        chartConfig.value.datasets[key] = !meta.hidden;
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
    };
};

const processData = (data, format) => {
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
};

const calculateMovingAverage = (data, window) => {
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

const toggleDataset = (datasetName) => {
    chartConfig.value.datasets[datasetName] = !chartConfig.value.datasets[datasetName];
    updateChart();
};

const updateColor = ({ datasetName, color }) => {
    chartConfig.value.colors[datasetName] = color;
    updateChart();
};

const updateChartType = (type) => {
    chartType.value = type;
    updateChart();
};

const updateDataFormat = (format) => {
    dataFormat.value = format;
    updateChart();
};

const updateScaleType = (type) => {
    scaleType.value = type;
    updateChart();
};

const zoomIn = () => {
    if (worldChart.value) {
        worldChart.value.zoom(1.2); // Zoom de 20%
    }
};

const zoomOut = () => {
    if (worldChart.value) {
        worldChart.value.zoom(0.8); // Zoom out de 20%
    }
};

const resetZoom = () => {
    if (worldChart.value) {
        worldChart.value.resetZoom();
    }
};

const downloadChart = () => {
    if (worldChart.value) {
        const link = document.createElement('a');
        link.download = 'covid-chart.png';
        link.href = worldChart.value.toBase64Image();
        link.click();
    }
};

const exportData = () => {
    if (!chartData.value) return;

    const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']];
    chartData.value.forEach(item => {
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
};

const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
};

// Cycle de vie du composant
onMounted(() => {
    updateChart();
});
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
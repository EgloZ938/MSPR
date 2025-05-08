<template>
    <div>
        <chart-options :chartType="chartType" :dataFormat="dataFormat" :scaleType="scaleType"
            :datasets="chartConfig.datasets" :colors="chartConfig.colors" @toggle-dataset="toggleDataset"
            @update-color="updateColor" @chart-type-change="updateChartType" @data-format-change="updateDataFormat"
            @scale-type-change="updateScaleType" />

        <chart-controls @zoom-in="zoomIn" @zoom-out="zoomOut" @reset-zoom="resetZoom" @download="downloadChart"
            @export="exportData" />

        <div class="chart-container">
            <canvas id="worldChart" ref="worldChart"></canvas>
        </div>
    </div>
</template>

<script>
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import ChartOptions from './ChartOptions.vue';
import ChartControls from './ChartControls.vue';
import axios from 'axios';

Chart.register(zoomPlugin);

export default {
    name: 'WorldChart',
    components: {
        ChartOptions,
        ChartControls
    },
    data() {
        return {
            worldChart: null,
            chartData: null,
            chartConfig: {
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
            },
            chartType: 'line',
            dataFormat: 'raw',
            scaleType: 'linear'
        }
    },
    mounted() {
        this.updateChart();
    },
    methods: {
        async updateChart() {
            try {
                this.$emit('toggle-loading', true);
                const response = await axios.get('/api/global-timeline');
                let data = response.data;

                // Stocker les données pour l'export CSV
                this.chartData = data;

                // Traitement des données selon le format
                data = this.processData(data, this.dataFormat);

                const ctx = this.$refs.worldChart.getContext('2d');

                if (this.worldChart) {
                    this.worldChart.destroy();
                }

                const datasets = [];

                // Création des datasets en fonction des sélections
                if (this.chartConfig.datasets.confirmed) {
                    datasets.push(this.createDataset('Cas confirmés', data.map(item => item.confirmed), this.chartConfig.colors.confirmed));
                }
                if (this.chartConfig.datasets.deaths) {
                    datasets.push(this.createDataset('Décès', data.map(item => item.deaths), this.chartConfig.colors.deaths));
                }
                if (this.chartConfig.datasets.recovered) {
                    datasets.push(this.createDataset('Guéris', data.map(item => item.recovered), this.chartConfig.colors.recovered));
                }
                if (this.chartConfig.datasets.active) {
                    datasets.push(this.createDataset('Cas actifs', data.map(item => item.active), this.chartConfig.colors.active));
                }

                this.worldChart = new Chart(ctx, {
                    type: this.getChartType(this.chartType),
                    data: {
                        labels: data.map(item => new Date(item.date).toLocaleDateString()),
                        datasets: datasets
                    },
                    options: this.getChartOptions()
                });
            } catch (error) {
                console.error('Erreur:', error);
                this.$emit('show-error', 'Erreur lors de la mise à jour du graphique');
            } finally {
                this.$emit('toggle-loading', false);
            }
        },
        createDataset(label, data, color) {
            return {
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: this.chartType === 'bar' ? `${color}88` : `${color}22`,
                fill: this.chartType === 'area',
                tension: 0.4,
                id: label.toLowerCase().replace(/\s+/g, ''),
                pointRadius: this.chartType === 'scatter' ? 4 : 0,
                pointHoverRadius: 6
            };
        },
        getChartOptions() {
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
                            const key = Object.keys(this.chartConfig.datasets).find(
                                k => k.toLowerCase() === datasetId
                            );
                            if (key) {
                                this.chartConfig.datasets[key] = !meta.hidden;
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
                                return `${context.dataset.label}: ${this.formatNumber(context.raw)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        type: this.scaleType,
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
        },
        processData(data, format) {
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
                    return this.calculateMovingAverage(data, 7);
                case 'monthly':
                    return this.calculateMovingAverage(data, 30);
                default:
                    return data;
            }
        },
        calculateMovingAverage(data, window) {
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
        },
        getChartType(selectedType) {
            switch (selectedType) {
                case 'area':
                    return 'line';
                case 'mixed':
                    return 'bar';
                default:
                    return selectedType;
            }
        },
        toggleDataset(datasetName) {
            this.chartConfig.datasets[datasetName] = !this.chartConfig.datasets[datasetName];
            this.updateChart();
        },
        updateColor({ datasetName, color }) {
            this.chartConfig.colors[datasetName] = color;
            this.updateChart();
        },
        updateChartType(type) {
            this.chartType = type;
            this.updateChart();
        },
        updateDataFormat(format) {
            this.dataFormat = format;
            this.updateChart();
        },
        updateScaleType(type) {
            this.scaleType = type;
            this.updateChart();
        },
        zoomIn() {
            if (this.worldChart) {
                this.worldChart.zoom(1.2); // Zoom de 20%
            }
        },
        zoomOut() {
            if (this.worldChart) {
                this.worldChart.zoom(0.8); // Zoom out de 20%
            }
        },
        resetZoom() {
            if (this.worldChart) {
                this.worldChart.resetZoom();
            }
        },
        downloadChart() {
            if (this.worldChart) {
                const link = document.createElement('a');
                link.download = 'covid-chart.png';
                link.href = this.worldChart.toBase64Image();
                link.click();
            }
        },
        exportData() {
            if (!this.chartData) return;

            const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']];
            this.chartData.forEach(item => {
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
        },
        formatNumber(num) {
            if (num === null || num === undefined) return '0';
            return new Intl.NumberFormat().format(num);
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
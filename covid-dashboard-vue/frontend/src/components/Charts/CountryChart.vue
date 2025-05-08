<template>
    <div>
        <chart-options :chartType="chartType" :dataFormat="dataFormat" :datasets="chartConfig.datasets"
            :colors="chartConfig.colors" :is-country-view="true" :show-pie-options="true"
            @toggle-dataset="toggleDataset" @update-color="updateColor" @chart-type-change="updateChartType"
            @data-format-change="updateDataFormat" />

        <chart-controls @zoom-in="zoomInCountry" @zoom-out="zoomOutCountry" @reset-zoom="resetCountryZoom"
            @download="downloadCountryChart" @export="exportCountryData" />

        <div class="visualization-row">
            <div class="chart-container">
                <canvas id="countryChart" ref="countryChart"></canvas>
            </div>

            <!-- Section pour le classement -->
            <div class="ranking-container">
                <h3>Classement des pays</h3>
                <div class="ranking-tabs">
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'confirmed' }"
                        @click="updateRanking('confirmed')">Cas confirmés</button>
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'deaths' }"
                        @click="updateRanking('deaths')">Décès</button>
                    <button class="ranking-tab" :class="{ active: activeRankingTab === 'mortality' }"
                        @click="updateRanking('mortality')">Taux de mortalité</button>
                </div>
                <div class="ranking-list" id="countryRanking">
                    <div v-for="(country, index) in rankings" :key="country.country_region" class="ranking-item"
                        :style="country.country_region === selectedCountry ? 'background-color: rgba(26, 115, 232, 0.1); font-weight: bold;' : ''"
                        @click="selectCountryFromRanking(country.country_region)">
                        <span class="rank">{{ index + 1 }}</span>
                        <span class="country">{{ country.country_region }}</span>
                        <span class="value">{{ formatRankingValue(country) }}</span>
                    </div>
                </div>
            </div>
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
    name: 'CountryChart',
    components: {
        ChartOptions,
        ChartControls
    },
    props: {
        selectedCountry: {
            type: String,
            required: true
        }
    },
    data() {
        return {
            countryChart: null,
            countryData: null,
            latestStats: null,
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
            rankings: [],
            activeRankingTab: 'confirmed'
        }
    },
    watch: {
        selectedCountry: {
            handler() {
                this.updateCountryChart();
            },
            immediate: true
        }
    },
    methods: {
        async updateCountryChart() {
            try {
                this.$emit('toggle-loading', true);

                // Charger les statistiques pour ce pays
                await this.loadCountryStats();

                if (!this.countryData || this.countryData.length === 0) {
                    this.$emit('show-error', `Aucune donnée disponible pour ${this.selectedCountry}`);
                    return;
                }

                // Traitement des données selon le format
                let processedData = this.countryData;
                if (this.dataFormat === 'daily') {
                    processedData = this.countryData.map((item, index) => {
                        if (index === 0) return item;
                        return {
                            ...item,
                            confirmed: item.confirmed - this.countryData[index - 1].confirmed,
                            deaths: item.deaths - this.countryData[index - 1].deaths,
                            recovered: item.recovered - this.countryData[index - 1].recovered,
                            active: item.active - this.countryData[index - 1].active
                        };
                    });
                }

                const ctx = this.$refs.countryChart.getContext('2d');

                if (this.countryChart) {
                    this.countryChart.destroy();
                }

                // Configuration spécifique pour les graphiques de type camembert/anneau
                if (this.chartType === 'pie' || this.chartType === 'doughnut') {
                    // Pour les graphiques circulaires, on utilise seulement la dernière date
                    const latestData = processedData[processedData.length - 1];

                    const labels = [];
                    const data = [];
                    const backgroundColors = [];

                    if (this.chartConfig.datasets.confirmed) {
                        labels.push('Cas confirmés');
                        data.push(latestData.confirmed);
                        backgroundColors.push(this.chartConfig.colors.confirmed);
                    }
                    if (this.chartConfig.datasets.deaths) {
                        labels.push('Décès');
                        data.push(latestData.deaths);
                        backgroundColors.push(this.chartConfig.colors.deaths);
                    }
                    if (this.chartConfig.datasets.recovered) {
                        labels.push('Guéris');
                        data.push(latestData.recovered);
                        backgroundColors.push(this.chartConfig.colors.recovered);
                    }
                    if (this.chartConfig.datasets.active) {
                        labels.push('Cas actifs');
                        data.push(latestData.active);
                        backgroundColors.push(this.chartConfig.colors.active);
                    }

                    this.countryChart = new Chart(ctx, {
                        type: this.chartType,
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
                                            return `${context.label}: ${this.formatNumber(value)} (${percentage}%)`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                } else {
                    // Configuration pour les graphiques chronologiques (ligne, barres)
                    const datasets = [];

                    if (this.chartConfig.datasets.confirmed) {
                        datasets.push(this.createDataset('Cas confirmés', processedData.map(item => item.confirmed), this.chartConfig.colors.confirmed));
                    }
                    if (this.chartConfig.datasets.deaths) {
                        datasets.push(this.createDataset('Décès', processedData.map(item => item.deaths), this.chartConfig.colors.deaths));
                    }
                    if (this.chartConfig.datasets.recovered) {
                        datasets.push(this.createDataset('Guéris', processedData.map(item => item.recovered), this.chartConfig.colors.recovered));
                    }
                    if (this.chartConfig.datasets.active) {
                        datasets.push(this.createDataset('Cas actifs', processedData.map(item => item.active), this.chartConfig.colors.active));
                    }

                    this.countryChart = new Chart(ctx, {
                        type: this.getChartType(this.chartType),
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

                                        // Mise à jour des checkboxes
                                        const datasetId = ci.data.datasets[index].id || legendItem.text.toLowerCase().replace(/\s+/g, '');
                                        const key = Object.keys(this.chartConfig.datasets).find(
                                            k => datasetId.includes(k.toLowerCase())
                                        );
                                        if (key) {
                                            this.chartConfig.datasets[key] = !meta.hidden;
                                        }
                                    }
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false,
                                    callbacks: {
                                        label: (context) => {
                                            return `${context.dataset.label}: ${this.formatNumber(context.raw)}`;
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

                // Charger le classement des pays
                await this.loadCountryRanking(this.activeRankingTab);

            } catch (error) {
                console.error('Erreur lors de la mise à jour du graphique du pays:', error);
                this.$emit('show-error', `Erreur lors de la mise à jour du graphique pour ${this.selectedCountry}`);
            } finally {
                this.$emit('toggle-loading', false);
            }
        },
        async loadCountryStats() {
            try {
                const response = await axios.get(`/api/country-timeline/${this.selectedCountry}`);
                this.countryData = response.data;

                // Afficher les dernières statistiques
                if (this.countryData.length > 0) {
                    this.latestStats = this.countryData[this.countryData.length - 1];
                    // Émettre un événement pour mettre à jour les stats du pays
                    this.$emit('update-stats', {
                        confirmed: this.latestStats.confirmed,
                        deaths: this.latestStats.deaths,
                        recovered: this.latestStats.recovered,
                        active: this.latestStats.active,
                        mortalityRate: this.latestStats.mortality_rate
                    });
                }
            } catch (error) {
                console.error('Erreur lors du chargement des statistiques du pays:', error);
                this.$emit('show-error', `Erreur lors du chargement des statistiques pour ${this.selectedCountry}`);
            }
        },
        async loadCountryRanking(metric) {
            try {
                this.activeRankingTab = metric;

                // Charger les données
                const response = await axios.get('/api/top-countries');
                let countries = response.data;

                // Trier par métrique choisie
                if (metric === 'confirmed') {
                    countries = countries.sort((a, b) => b.confirmed - a.confirmed);
                } else if (metric === 'deaths') {
                    countries = countries.sort((a, b) => b.deaths - a.deaths);
                } else if (metric === 'mortality') {
                    countries = countries.sort((a, b) => b.mortality_rate - a.mortality_rate);
                }

                // Afficher le top 20
                this.rankings = countries.slice(0, 20);
            } catch (error) {
                console.error('Erreur lors du chargement du classement:', error);
                this.$emit('show-error', 'Erreur lors du chargement du classement des pays');
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
            this.updateCountryChart();
        },
        updateColor({ datasetName, color }) {
            this.chartConfig.colors[datasetName] = color;
            this.updateCountryChart();
        },
        updateChartType(type) {
            this.chartType = type;
            this.updateCountryChart();
        },
        updateDataFormat(format) {
            this.dataFormat = format;
            this.updateCountryChart();
        },
        updateRanking(metric) {
            this.loadCountryRanking(metric);
        },
        selectCountryFromRanking(countryName) {
            this.$emit('country-changed', countryName);
        },
        zoomInCountry() {
            if (this.countryChart) {
                const zoomOptions = this.countryChart.options.plugins.zoom.zoom;
                zoomOptions.wheel.enabled = false;

                const centerX = this.countryChart.chartArea.width / 2;
                const centerY = this.countryChart.chartArea.height / 2;
                this.countryChart.pan({ x: 0, y: 0 }, 'none', 'default');
                this.countryChart.zoom(1.2, 'xy', { x: centerX, y: centerY });
            }
        },
        zoomOutCountry() {
            if (this.countryChart) {
                const zoomOptions = this.countryChart.options.plugins.zoom.zoom;
                zoomOptions.wheel.enabled = false;

                const centerX = this.countryChart.chartArea.width / 2;
                const centerY = this.countryChart.chartArea.height / 2;
                this.countryChart.pan({ x: 0, y: 0 }, 'none', 'default');
                this.countryChart.zoom(0.8, 'xy', { x: centerX, y: centerY });
            }
        },
        resetCountryZoom() {
            if (this.countryChart) {
                this.countryChart.resetZoom();
            }
        },
        downloadCountryChart() {
            if (this.countryChart) {
                const link = document.createElement('a');
                link.download = `covid-${this.selectedCountry.toLowerCase()}-chart.png`;
                link.href = this.countryChart.toBase64Image();
                link.click();
            }
        },
        exportCountryData() {
            if (!this.countryData) return;

            const rows = [['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate']];
            this.countryData.forEach(item => {
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
            link.setAttribute("download", `covid-${this.selectedCountry.toLowerCase()}-data.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        },
        formatNumber(num) {
            if (num === null || num === undefined) return '0';
            return new Intl.NumberFormat().format(num);
        },
        formatRankingValue(country) {
            if (this.activeRankingTab === 'confirmed') {
                return this.formatNumber(country.confirmed);
            } else if (this.activeRankingTab === 'deaths') {
                return this.formatNumber(country.deaths);
            } else if (this.activeRankingTab === 'mortality') {
                return `${country.mortality_rate.toFixed(2)}%`;
            }
            return '';
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
<template>
    <div id="predictive" class="visualization-section">
        <div class="predictive-header">
            <h2>🤖 Prédictions IA Révolutionnaire</h2>
            <div class="model-status">
                <span class="status-indicator" :class="modelStatus"></span>
                <span class="status-text">{{ modelStatusText }}</span>
            </div>
        </div>

        <!-- Sélection Pays et Configuration -->
        <div class="prediction-controls">
            <div class="controls-grid">
                <!-- Sélecteur de Pays (réutilise ton composant existant) -->
                <div class="control-group">
                    <label class="control-label">Pays à analyser :</label>
                    <country-selector-responsive v-model="selectedCountry" @country-changed="handleCountryChange"
                        @show-error="showError" />
                </div>

                <!-- Horizons de Prédiction -->
                <div class="control-group">
                    <label class="control-label">Horizons de prédiction :</label>
                    <div class="horizons-selector">
                        <button v-for="horizon in availableHorizons" :key="horizon" @click="toggleHorizon(horizon)"
                            :class="{
                                'horizon-btn': true,
                                'active': selectedHorizons.includes(horizon),
                                'disabled': isCoolingDown
                            }" :disabled="isCoolingDown">
                            {{ horizon }}j
                        </button>
                    </div>
                </div>

                <!-- Options Avancées -->
                <div class="control-group">
                    <label class="control-label">Options :</label>
                    <div class="advanced-options">
                        <label class="option-checkbox">
                            <input type="checkbox" v-model="includeUncertainty" :disabled="isCoolingDown" />
                            <span class="checkmark"></span>
                            Intervalles de confiance
                        </label>
                        <label class="option-checkbox">
                            <input type="checkbox" v-model="includeVaccination" :disabled="isCoolingDown" />
                            <span class="checkmark"></span>
                            Impact vaccination
                        </label>
                    </div>
                </div>
            </div>

            <!-- Bouton Principal de Prédiction -->
            <div class="prediction-action">
                <button @click="runPrediction" :disabled="!canPredict" class="predict-button"
                    :class="{ 'loading': isPredicting }">
                    <div v-if="isPredicting" class="loading-content">
                        <div class="loading-spinner"></div>
                        <span>Calcul IA en cours...</span>
                    </div>
                    <div v-else class="predict-content">
                        <span class="predict-icon">🚀</span>
                        <span>Lancer Prédictions IA</span>
                    </div>
                </button>
            </div>
        </div>

        <!-- Résultats des Prédictions -->
        <div v-if="predictions.length > 0" class="predictions-results">

            <!-- Stats Cards Style Dashboard Existant -->
            <div class="predictions-stats-grid">
                <div v-for="prediction in predictions" :key="prediction.horizon_days" class="prediction-card">
                    <div class="card-header">
                        <h4>{{ prediction.horizon_days }} jour{{ prediction.horizon_days > 1 ? 's' : '' }}</h4>
                        <span class="horizon-badge">{{ prediction.date }}</span>
                    </div>
                    <div class="card-stats">
                        <div class="stat-item confirmed">
                            <span class="stat-label">Cas Confirmés</span>
                            <span class="stat-value">{{ formatNumber(prediction.confirmed) }}</span>
                        </div>
                        <div class="stat-item active">
                            <span class="stat-label">Cas Actifs</span>
                            <span class="stat-value">{{ formatNumber(prediction.active) }}</span>
                        </div>
                        <div class="stat-item deaths">
                            <span class="stat-label">Décès</span>
                            <span class="stat-value">{{ formatNumber(prediction.deaths) }}</span>
                        </div>
                    </div>

                    <!-- Intervalle de Confiance -->
                    <div v-if="includeUncertainty && prediction.confidence_intervals" class="confidence-section">
                        <div class="confidence-bar">
                            <div class="confidence-fill" :style="{ width: getConfidencePercent(prediction) + '%' }">
                            </div>
                        </div>
                        <span class="confidence-text">{{ getConfidencePercent(prediction) }}% confiance</span>
                    </div>
                </div>
            </div>

            <!-- Graphique Prédictions (Style Chart.js Existant) -->
            <div class="prediction-chart-container">
                <div class="chart-header">
                    <h3>📈 Évolution Prédite - {{ selectedCountry }}</h3>
                    <!-- Réutilise tes contrôles de chart existants -->
                    <chart-controls-responsive @zoom-in="zoomInChart" @zoom-out="zoomOutChart"
                        @reset-zoom="resetZoomChart" @download="downloadChart" @export="exportPredictions" />
                </div>
                <div class="chart-container">
                    <canvas ref="predictionChartRef" id="predictionChart"></canvas>
                </div>
            </div>

            <!-- Analyse IA -->
            <div class="ai-analysis">
                <h3>🧠 Analyse Intelligente</h3>

                <!-- 🔍 DEBUG: Affichage brut des données reçues -->
                <div class="debug-section"
                    style="background: #f0f0f0; padding: 10px; margin-bottom: 16px; border-radius: 4px; font-family: monospace; font-size: 12px;">
                    <strong>🔍 DEBUG API Response:</strong><br>
                    Vaccination Impact: {{ JSON.stringify(vaccinationImpact, null, 2) }}<br>
                    Demographic Factors: {{ JSON.stringify(demographicFactors, null, 2) }}<br>
                    Model Confidence: {{ JSON.stringify(modelConfidence, null, 2) }}
                </div>

                <div class="analysis-grid">
                    <div class="analysis-card trend">
                        <h4>Tendance Détectée</h4>
                        <div class="trend-indicator" :class="getTrendClass()">
                            {{ analyzeTrend() }}
                        </div>
                    </div>

                    <div class="analysis-card vaccination">
                        <h4>Impact Vaccination</h4>
                        <div class="vaccination-info">
                            <div v-if="vaccinationImpact && vaccinationImpact.effectiveness_score !== undefined">
                                Score: {{ Math.round(vaccinationImpact.effectiveness_score * 100) }}%
                            </div>
                            <div v-else-if="vaccinationImpact && vaccinationImpact.current_coverage !== undefined">
                                Couverture: {{ Math.round(vaccinationImpact.current_coverage * 100) }}%
                            </div>
                            <div v-else class="no-data">
                                ⚠️ Données non disponibles
                                <small>{{ vaccinationImpact ? 'Structure inattendue' : 'Réponse vide' }}</small>
                            </div>
                        </div>
                    </div>

                    <div class="analysis-card demographic">
                        <h4>Facteurs Démographiques</h4>
                        <div class="demographic-info">
                            <div v-if="demographicFactors && demographicFactors.vulnerability_score !== undefined">
                                Vulnérabilité: {{ Math.round(demographicFactors.vulnerability_score * 100) }}%
                            </div>
                            <div v-else-if="demographicFactors && demographicFactors.population_millions !== undefined">
                                Population: {{ demographicFactors.population_millions }}M hab.
                            </div>
                            <div v-else class="no-data">
                                ⚠️ Données non disponibles
                                <small>{{ demographicFactors ? 'Structure inattendue' : 'Réponse vide' }}</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- État Vide (Style de ton Dashboard) -->
        <div v-else-if="!isPredicting" class="empty-state">
            <div class="empty-icon">🎯</div>
            <h3>Intelligence Artificielle Prête</h3>
            <p>Sélectionnez un pays et configurez vos horizons pour générer des prédictions COVID-19 révolutionnaires.
            </p>
        </div>

        <!-- Mini Loader (Réutilise ton système) -->
        <mini-loader :show="isCoolingDown" />
        <loading-indicator :is-loading="isPredicting" />
    </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue';
import Chart from 'chart.js/auto';
import axios from 'axios';

// Réutilise tes composants existants
import CountrySelectorResponsive from './CountrySelectorResponsive.vue';
import ChartControlsResponsive from '../Charts/ChartControlsResponsive.vue';
import MiniLoader from '../MiniLoader.vue';
import LoadingIndicator from './LoadingIndicator.vue';

// Émissions (compatible avec ton système existant)
const emit = defineEmits(['show-error', 'toggle-loading']);

// État réactif
const selectedCountry = ref('France');
const predictions = ref([]);
const isPredicting = ref(false);
const isCoolingDown = ref(false);
const modelStatus = ref('loading');
const modelStatusText = ref('Vérification...');

// Configuration (utilise ta logique existante)
const availableHorizons = ref([1, 7, 14, 30]);
const selectedHorizons = ref([1, 7, 14, 30]);
const includeUncertainty = ref(true);
const includeVaccination = ref(true);

// Données additionnelles
const vaccinationImpact = ref(null);
const demographicFactors = ref(null);
const modelConfidence = ref(null);

// Chart
const predictionChartRef = ref(null);
const predictionChart = ref(null);

// Computed (Style de ton code existant)
const canPredict = computed(() => {
    return selectedCountry.value &&
        selectedHorizons.value.length > 0 &&
        !isPredicting.value;
});

// Lifecycle (Compatible avec ton onMounted existant)
onMounted(async () => {
    await checkModelStatus();
});

// Méthodes (Style de tes fonctions existantes)
async function checkModelStatus() {
    try {
        const response = await axios.get('http://localhost:8000/');

        if (response.data.model_loaded) {
            modelStatus.value = 'healthy';
            modelStatusText.value = 'Modèle Opérationnel';
        } else {
            modelStatus.value = 'warning';
            modelStatusText.value = 'Modèle Partiel';
        }
    } catch (error) {
        modelStatus.value = 'error';
        modelStatusText.value = 'Modèle Indisponible';
        console.error('Erreur vérification modèle:', error);
    }
}

// Fonction principale (Style de tes updateChart existantes)
async function runPrediction() {
    if (!canPredict.value) return;

    isPredicting.value = true;
    emit('toggle-loading', true);

    try {
        const requestData = {
            country: selectedCountry.value,
            prediction_horizons: selectedHorizons.value,
            include_uncertainty: includeUncertainty.value,
            include_attention: false
        };

        console.log('🚀 Envoi requête prédiction IA:', requestData);

        const response = await axios.post('http://localhost:8000/predict', requestData, {
            timeout: 30000
        });

        console.log('✅ Réponse API Prédictive COMPLÈTE:', response.data);

        // 🔍 DEBUG SPÉCIFIQUE pour vaccination et démographie
        console.log('💉 Vaccination Impact:', response.data.vaccination_impact);
        console.log('👥 Demographic Factors:', response.data.demographic_factors);
        console.log('🎯 Model Confidence:', response.data.model_confidence);

        // Traitement des données
        predictions.value = response.data.predictions || [];

        // 🔧 VÉRIFICATION et traitement des données additionnelles
        vaccinationImpact.value = response.data.vaccination_impact;
        demographicFactors.value = response.data.demographic_factors;
        modelConfidence.value = response.data.model_confidence;

        // 🚨 DEBUG: Afficher les valeurs dans la console
        if (vaccinationImpact.value) {
            console.log('💉 Impact Vaccination détaillé:', {
                effectiveness_score: vaccinationImpact.value.effectiveness_score,
                current_coverage: vaccinationImpact.value.current_coverage,
                vaccination_momentum: vaccinationImpact.value.vaccination_momentum,
                data_available: vaccinationImpact.value.data_available
            });
        } else {
            console.warn('⚠️ vaccinationImpact est null/undefined');
        }

        if (demographicFactors.value) {
            console.log('👥 Facteurs Démographiques détaillés:', {
                vulnerability_score: demographicFactors.value.vulnerability_score,
                population_millions: demographicFactors.value.population_millions,
                elderly_ratio: demographicFactors.value.elderly_ratio,
                life_expectancy: demographicFactors.value.life_expectancy
            });
        } else {
            console.warn('⚠️ demographicFactors est null/undefined');
        }

        // Créer le graphique
        await nextTick();
        createPredictionChart();

    } catch (error) {
        console.error('❌ Erreur prédiction:', error);

        // 🔍 DEBUG: Plus de détails sur l'erreur
        if (error.response) {
            console.error('📋 Détails de l\'erreur:', {
                status: error.response.status,
                statusText: error.response.statusText,
                data: error.response.data
            });
        }

        let errorMessage = 'Erreur lors de la prédiction';
        if (error.response?.status === 404) {
            errorMessage = `Pays "${selectedCountry.value}" non trouvé`;
        } else if (error.response?.status === 503) {
            errorMessage = 'Modèle IA temporairement indisponible';
        } else if (error.code === 'ECONNABORTED') {
            errorMessage = 'Timeout: Le modèle prend trop de temps à répondre';
        }

        emit('show-error', errorMessage);

    } finally {
        isPredicting.value = false;
        emit('toggle-loading', false);
    }
}

// Gestion des événements (Style de tes handlers existants)
function handleCountryChange(country) {
    selectedCountry.value = country;
    predictions.value = []; // Reset des prédictions
}

function toggleHorizon(horizon) {
    if (isCoolingDown.value) return;

    const index = selectedHorizons.value.indexOf(horizon);
    if (index > -1) {
        if (selectedHorizons.value.length > 1) {
            selectedHorizons.value.splice(index, 1);
        }
    } else {
        selectedHorizons.value.push(horizon);
        selectedHorizons.value.sort((a, b) => a - b);
    }
}

// Création graphique (Style de tes createChart existantes)
function createPredictionChart() {
    if (!predictionChartRef.value || predictions.value.length === 0) return;

    if (predictionChart.value) {
        predictionChart.value.destroy();
    }

    const ctx = predictionChartRef.value.getContext('2d');

    // 🚀 DONNÉES COMPLÈTES - Toutes les métriques
    const labels = predictions.value.map(p => `${p.horizon_days}j`);

    // ✅ Dataset pour TOUTES les métriques COVID
    const datasets = [
        {
            label: 'Cas Confirmés (Prédiction)',
            data: predictions.value.map(p => p.confirmed),
            borderColor: '#1a73e8',
            backgroundColor: 'rgba(26, 115, 232, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointStyle: 'circle',
            pointRadius: 4
        },
        {
            label: 'Cas Actifs (Prédiction)',
            data: predictions.value.map(p => p.active),
            borderColor: '#ffc107',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointStyle: 'triangle',
            pointRadius: 4
        },
        {
            label: 'Décès (Prédiction)',
            data: predictions.value.map(p => p.deaths),
            borderColor: '#dc3545',
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointStyle: 'rect',
            pointRadius: 3
        },
        {
            label: 'Guérisons (Prédiction)',
            data: predictions.value.map(p => p.recovered),
            borderColor: '#28a745',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointStyle: 'rectRot',
            pointRadius: 3
        }
    ];

    // 🎯 Configuration améliorée du graphique
    predictionChart.value = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        color: '#666',
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#1a73e8',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        title: (tooltipItems) => {
                            return `Prédiction à ${tooltipItems[0].label}`;
                        },
                        label: (context) => {
                            const value = formatNumber(context.raw);
                            const percentage = getChangePercentage(context.datasetIndex, context.dataIndex);
                            return `${context.dataset.label}: ${value} ${percentage}`;
                        },
                        afterBody: (tooltipItems) => {
                            if (includeUncertainty.value) {
                                const prediction = predictions.value[tooltipItems[0].dataIndex];
                                if (prediction.confidence_intervals) {
                                    return [`Confiance: ${getConfidencePercent(prediction)}%`];
                                }
                            }
                            return [];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0,0,0,0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#666',
                        font: { size: 12 },
                        callback: (value) => formatNumber(value)
                    },
                    title: {
                        display: true,
                        text: 'Nombre de cas',
                        color: '#666',
                        font: { size: 13, weight: 'bold' }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#666',
                        font: { size: 12 }
                    },
                    title: {
                        display: true,
                        text: 'Horizon de prédiction',
                        color: '#666',
                        font: { size: 13, weight: 'bold' }
                    }
                }
            },
            elements: {
                point: {
                    hoverRadius: 8,
                    hoverBorderWidth: 3
                },
                line: {
                    borderCapStyle: 'round',
                    borderJoinStyle: 'round'
                }
            }
        }
    });
}

// Fonction pour calculer les variations
function getChangePercentage(datasetIndex, pointIndex) {
    if (pointIndex === 0) return '';

    const current = predictions.value[pointIndex];
    const previous = predictions.value[pointIndex - 1];

    let currentVal, previousVal;

    switch (datasetIndex) {
        case 0: // Confirmed
            currentVal = current.confirmed;
            previousVal = previous.confirmed;
            break;
        case 1: // Active
            currentVal = current.active;
            previousVal = previous.active;
            break;
        case 2: // Deaths
            currentVal = current.deaths;
            previousVal = previous.deaths;
            break;
        case 3: // Recovered
            currentVal = current.recovered;
            previousVal = previous.recovered;
            break;
        default:
            return '';
    }

    if (previousVal === 0) return '';

    const change = ((currentVal - previousVal) / previousVal) * 100;
    const sign = change > 0 ? '+' : '';
    return `(${sign}${change.toFixed(1)}%)`;
}

// Fonctions utilitaires (Style de tes helpers existants)
function formatNumber(num) {
    if (!num && num !== 0) return '0';
    return new Intl.NumberFormat('fr-FR').format(Math.round(num));
}

function getConfidencePercent(prediction) {
    // Calcul simple basé sur l'horizon (plus c'est loin, moins c'est fiable)
    const baseConfidence = 95;
    const horizonPenalty = prediction.horizon_days * 2;
    return Math.max(baseConfidence - horizonPenalty, 70);
}

function analyzeTrend() {
    if (predictions.value.length < 2) return 'Données insuffisantes';

    const first = predictions.value[0];
    const last = predictions.value[predictions.value.length - 1];
    const growth = ((last.confirmed - first.confirmed) / first.confirmed) * 100;

    if (growth > 5) return '📈 Tendance à la hausse';
    if (growth > 1) return '↗️ Légère hausse';
    if (growth > -1) return '➡️ Stabilisation';
    if (growth > -5) return '↘️ Légère baisse';
    return '📉 Tendance à la baisse';
}

function getTrendClass() {
    const trend = analyzeTrend();
    if (trend.includes('hausse')) return 'trend-up';
    if (trend.includes('baisse')) return 'trend-down';
    return 'trend-stable';
}

// Contrôles Chart (Réutilise ta logique existante)
function zoomInChart() {
    if (predictionChart.value) predictionChart.value.zoom(1.2);
}

function zoomOutChart() {
    if (predictionChart.value) predictionChart.value.zoom(0.8);
}

function resetZoomChart() {
    if (predictionChart.value) predictionChart.value.resetZoom();
}

function downloadChart() {
    if (!predictionChart.value) return;

    const link = document.createElement('a');
    link.download = `predictions-covid-${selectedCountry.value}-${new Date().toISOString().split('T')[0]}.png`;
    link.href = predictionChart.value.toBase64Image();
    link.click();
}

function exportPredictions() {
    if (predictions.value.length === 0) return;

    // CSV Export (Style de tes export functions)
    const headers = ['Horizon (jours)', 'Date', 'Cas Confirmés', 'Décès', 'Guéris', 'Cas Actifs'];
    const rows = [headers];

    predictions.value.forEach(pred => {
        rows.push([
            pred.horizon_days,
            pred.date,
            pred.confirmed,
            pred.deaths,
            pred.recovered,
            pred.active
        ]);
    });

    const csvContent = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `predictions-covid-${selectedCountry.value}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function showError(message) {
    emit('show-error', message);
}
</script>

<style scoped>
/* Réutilise le style de ton dashboard existant */
.visualization-section {
    /* Hérite de tes styles existants */
}

.predictive-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.predictive-header h2 {
    color: var(--primary-color);
    font-weight: 700;
    margin: 0;
}

.model-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
}

.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.status-indicator.healthy {
    background-color: #10b981;
}

.status-indicator.warning {
    background-color: #f59e0b;
}

.status-indicator.error {
    background-color: #ef4444;
}

.status-indicator.loading {
    background-color: #6b7280;
}

.prediction-controls {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.controls-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.control-label {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.875rem;
}

.horizons-selector {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.horizon-btn {
    padding: 8px 16px;
    border: 1px solid var(--border-color);
    background: white;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 0.875rem;
}

.horizon-btn:hover:not(.disabled) {
    border-color: var(--primary-color);
}

.horizon-btn.active {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.horizon-btn.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.advanced-options {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.option-checkbox {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
}

.option-checkbox input[type="checkbox"] {
    display: none;
}

.checkmark {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border-color);
    border-radius: 3px;
    position: relative;
    transition: all 0.2s ease;
}

.option-checkbox input[type="checkbox"]:checked+.checkmark {
    background: var(--primary-color);
    border-color: var(--primary-color);
}

.option-checkbox input[type="checkbox"]:checked+.checkmark::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-size: 12px;
    font-weight: bold;
}

.prediction-action {
    text-align: center;
}

.predict-button {
    background: linear-gradient(135deg, var(--primary-color), var(--hover-color));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.predict-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

.predict-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.predict-button.loading {
    background: #6b7280;
}

.loading-content,
.predict-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.loading-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.predict-icon {
    font-size: 1.1rem;
}

.predictions-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.prediction-card {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid var(--border-color);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.card-header h4 {
    margin: 0;
    color: var(--primary-color);
    font-weight: 600;
}

.horizon-badge {
    background: var(--primary-color);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
}

.card-stats {
    display: grid;
    gap: 8px;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
}

.stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
}

.stat-value {
    font-weight: 600;
    font-size: 1rem;
}

.stat-item.confirmed .stat-value {
    color: #1a73e8;
}

.stat-item.deaths .stat-value {
    color: #dc3545;
}

.stat-item.active .stat-value {
    color: #ffc107;
}

.confidence-section {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
}

.confidence-bar {
    width: 100%;
    height: 6px;
    background: #e0e0e0;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 4px;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    transition: width 0.3s ease;
}

.confidence-text {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.prediction-chart-container {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.chart-header h3 {
    margin: 0;
    color: var(--text-primary);
    font-weight: 600;
}

.chart-container {
    height: 400px;
    position: relative;
}

.ai-analysis {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.ai-analysis h3 {
    margin: 0 0 16px 0;
    color: var(--primary-color);
    font-weight: 600;
}

.analysis-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.analysis-card {
    background: rgba(0, 0, 0, 0.02);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border-color);
}

.analysis-card h4 {
    margin: 0 0 8px 0;
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.trend-indicator {
    font-weight: 600;
    font-size: 1rem;
}

.trend-indicator.trend-up {
    color: #ef4444;
}

.trend-indicator.trend-down {
    color: #10b981;
}

.trend-indicator.trend-stable {
    color: #6b7280;
}

.vaccination-score,
.demographic-risk {
    font-weight: 600;
    font-size: 1rem;
    color: var(--primary-color);
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-secondary);
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 16px;
}

.empty-state h3 {
    color: var(--text-primary);
    margin-bottom: 8px;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

@keyframes pulse {

    0%,
    100% {
        opacity: 1;
    }

    50% {
        opacity: 0.5;
    }
}

/* Responsive */
@media (max-width: 768px) {
    .controls-grid {
        grid-template-columns: 1fr;
    }

    .predictive-header {
        flex-direction: column;
        gap: 12px;
        align-items: flex-start;
    }

    .chart-header {
        flex-direction: column;
        gap: 12px;
        align-items: flex-start;
    }

    .analysis-grid {
        grid-template-columns: 1fr;
    }
}
</style>
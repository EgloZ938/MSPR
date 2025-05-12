<template>
  <div id="pays" class="visualization-section">
    <div class="options-panel">
      <div class="options-grid">
        <div class="option-group">
          <label>Sélectionner les pays (maximum 3)</label>
          <multi-country-selector :value="selectedCountries" @countries-changed="updateSelectedCountries"
            @show-error="showError" />
        </div>

        <div class="option-group view-toggle">
          <label>Mode d'affichage</label>
          <div class="view-toggle-buttons">
            <button :class="{ 'active': viewMode === 'single', 'disabled': isCoolingDown }"
              @click="changeViewMode('single')" :disabled="isCoolingDown">
              Vue par pays
            </button>
            <button :class="{ 'active': viewMode === 'compare', 'disabled': isCoolingDown }"
              @click="changeViewMode('compare')" :disabled="isCoolingDown">
              Comparaison
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Mode Vue par pays : affiche les stats pour le pays actif -->
    <template v-if="viewMode === 'single'">
      <!-- Sélecteur du pays actif -->
      <div v-if="selectedCountries.length > 1" class="active-country-selector">
        <label>Pays affiché :</label>
        <div class="active-country-buttons">
          <button v-for="country in selectedCountries" :key="country"
            :class="{ 'active': activeCountry === country, 'disabled': isCoolingDown }"
            @click="changeActiveCountry(country)">
            {{ country }}
          </button>
        </div>
      </div>

      <country-stats :stats="countryStats" />
      <country-chart ref="countryChartRef" :selectedCountry="activeCountry" @toggle-loading="setLoading"
        @show-error="showError" @update-stats="updateCountryStats" @country-changed="updatePrimaryCountry"
        @chart-type-change="updateChartType" @data-format-change="updateDataFormat" @scale-type-change="updateScaleType"
        @toggle-dataset="updateDatasetVisibility" @update-color="updateDatasetColor" />
    </template>

    <!-- Mode Comparaison : affiche le graphique de comparaison des pays -->
    <template v-if="viewMode === 'compare'">
      <div class="comparison-instructions" v-if="selectedCountries.length > 1">
        <p>Comparaison des données entre <strong>{{ selectedCountries.join(', ') }}</strong></p>
      </div>
      <div class="comparison-instructions" v-else>
        <p>Veuillez sélectionner au moins deux pays pour la comparaison</p>
      </div>
      <multi-country-chart ref="multiCountryChartRef" :selectedCountries="selectedCountries"
        :chartType="chartOptions.chartType" :dataFormat="chartOptions.dataFormat" :scaleType="chartOptions.scaleType"
        :datasetConfig="chartOptions.datasets" :colorConfig="chartOptions.colors" @toggle-loading="setLoading"
        @show-error="showError" @chart-type-change="updateChartType" @data-format-change="updateDataFormat"
        @scale-type-change="updateScaleType" @toggle-dataset="updateDatasetVisibility"
        @update-color="updateDatasetColor" />
    </template>

    <loading-indicator :is-loading="isLoading" />

    <!-- Mini Loader pour le cooldown -->
    <mini-loader :show="isCoolingDown" />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import CountryStats from './CountryStats.vue';
import CountryChart from '../Charts/CountryChart.vue';
import MultiCountryChart from '../Charts/MultiCountryChart.vue';
import MultiCountrySelector from './MultiCountrySelector.vue';
import LoadingIndicator from './LoadingIndicator.vue';
import MiniLoader from '../MiniLoader.vue';

// Définir l'émetteur d'événements
const emit = defineEmits(['show-error']);

const selectedCountries = ref(['France']);
const countryStats = ref({
  confirmed: null,
  deaths: null,
  recovered: null,
  active: null,
  mortalityRate: null
});
const isLoading = ref(false);
const isCoolingDown = ref(false);
const viewMode = ref('single'); // 'single' ou 'compare'

// Références aux composants de graphiques
const countryChartRef = ref(null);
const multiCountryChartRef = ref(null);

// Pays actif pour la vue détaillée (par défaut, le premier pays sélectionné)
const activeCountry = ref('France');

// Options partagées entre les deux graphiques
const chartOptions = ref({
  chartType: 'line',
  dataFormat: 'raw',
  scaleType: 'linear',
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

// Observer les changements dans selectedCountries pour mettre à jour activeCountry
watch(selectedCountries, (newCountries) => {
  // Si le pays actif n'est plus dans la liste, prendre le premier pays de la liste
  if (newCountries.length > 0 && !newCountries.includes(activeCountry.value)) {
    activeCountry.value = newCountries[0];
  }
}, { deep: true });

// Synchroniser les options entre les composants lors du changement de mode
watch(viewMode, (newMode) => {
  // Laisser un peu de temps pour que les composants soient montés
  setTimeout(() => {
    synchronizeChartOptions();
  }, 100);
});

// Fonction pour activer le cooldown
function activateCooldown(duration = 800) {
  isCoolingDown.value = true;
  setTimeout(() => {
    isCoolingDown.value = false;
  }, duration);
}

// Fonction pour synchroniser les options entre les deux graphiques
function synchronizeChartOptions() {
  if (viewMode.value === 'single' && countryChartRef.value) {
    // Récupérer les options du composant CountryChart s'il est actif
    const countryChart = countryChartRef.value;
    if (countryChart.chartType) chartOptions.value.chartType = countryChart.chartType;
    if (countryChart.dataFormat) chartOptions.value.dataFormat = countryChart.dataFormat;
    if (countryChart.scaleType) chartOptions.value.scaleType = countryChart.scaleType;
    if (countryChart.chartConfig?.datasets) chartOptions.value.datasets = { ...countryChart.chartConfig.datasets };
    if (countryChart.chartConfig?.colors) chartOptions.value.colors = { ...countryChart.chartConfig.colors };
  } else if (viewMode.value === 'compare' && multiCountryChartRef.value) {
    // Récupérer les options du composant MultiCountryChart s'il est actif
    const multiChart = multiCountryChartRef.value;
    if (multiChart.chartType) chartOptions.value.chartType = multiChart.chartType;
    if (multiChart.dataFormat) chartOptions.value.dataFormat = multiChart.dataFormat;
    if (multiChart.scaleType) chartOptions.value.scaleType = multiChart.scaleType;
    if (multiChart.chartConfig?.datasets) chartOptions.value.datasets = { ...multiChart.chartConfig.datasets };
    if (multiChart.chartConfig?.colors) chartOptions.value.colors = { ...multiChart.chartConfig.colors };
  }
}

function updateSelectedCountries(countries) {
  selectedCountries.value = countries;

  // Si la liste est vide, ajouter "France" par défaut
  if (countries.length === 0) {
    selectedCountries.value = ['France'];
  }

  // Mettre à jour le pays actif si nécessaire
  if (!countries.includes(activeCountry.value) && countries.length > 0) {
    activeCountry.value = countries[0];
  }

  // Réinitialiser les stats pour qu'elles soient rechargées
  if (viewMode.value === 'single') {
    updateCountryStats({
      confirmed: null,
      deaths: null,
      recovered: null,
      active: null,
      mortalityRate: null
    });
  }
}

function updatePrimaryCountry(country) {
  if (isCoolingDown.value) return;
  activateCooldown(1000);

  // Mettre à jour le pays actif
  activeCountry.value = country;

  // Mettre le pays sélectionné en première position
  const countries = [...selectedCountries.value];
  const index = countries.indexOf(country);

  if (index > 0) {
    // Si le pays existe déjà mais n'est pas en première position
    countries.splice(index, 1);
    countries.unshift(country);
  } else if (index === -1) {
    // Si le pays n'existe pas dans la liste
    countries.unshift(country);
    if (countries.length > 3) {
      countries.pop(); // Garder maximum 3 pays
    }
  }

  selectedCountries.value = countries;
}

function changeActiveCountry(country) {
  if (isCoolingDown.value) return;
  if (activeCountry.value === country) return;

  activateCooldown(800);
  activeCountry.value = country;
}

function updateCountryStats(stats) {
  countryStats.value = stats;
}

function showError(message) {
  emit('show-error', message);
}

function setLoading(loading) {
  isLoading.value = loading;
}

function changeViewMode(mode) {
  if (isCoolingDown.value) return;
  if (viewMode.value === mode) return;

  activateCooldown(800);
  viewMode.value = mode;
}

// Fonctions pour mettre à jour les options partagées
function updateChartType(type) {
  chartOptions.value.chartType = type;
}

function updateDataFormat(format) {
  chartOptions.value.dataFormat = format;
}

function updateScaleType(type) {
  chartOptions.value.scaleType = type;
}

function updateDatasetVisibility(datasetName) {
  chartOptions.value.datasets[datasetName] = !chartOptions.value.datasets[datasetName];
}

function updateDatasetColor({ datasetName, color }) {
  chartOptions.value.colors[datasetName] = color;
}

// Synchroniser les options au montage initial
onMounted(() => {
  // Attendre que les composants soient montés
  setTimeout(() => {
    synchronizeChartOptions();
  }, 200);
});
</script>

<style scoped>
.view-toggle-buttons {
  display: flex;
  gap: 10px;
}

.view-toggle-buttons button {
  flex: 1;
  padding: 8px 16px;
  border: 1px solid var(--border-color);
  background-color: white;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.view-toggle-buttons button.active {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.view-toggle-buttons button:hover:not(.active):not(.disabled) {
  border-color: var(--primary-color);
}

.comparison-instructions {
  margin: 20px 0;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid var(--primary-color);
  text-align: center;
}

.disabled {
  opacity: 0.6;
  cursor: not-allowed;
  pointer-events: none;
}

/* Styles pour le sélecteur de pays actif */
.active-country-selector {
  display: flex;
  align-items: center;
  margin: 20px 0;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.active-country-selector label {
  margin-right: 15px;
  font-weight: bold;
  color: var(--text-secondary);
}

.active-country-buttons {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.active-country-buttons button {
  padding: 8px 16px;
  border: 1px solid var(--border-color);
  background-color: white;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.active-country-buttons button.active {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.active-country-buttons button:hover:not(.active):not(.disabled) {
  border-color: var(--primary-color);
  background-color: rgba(26, 115, 232, 0.1);
}
</style>
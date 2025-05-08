<template>
    <div id="pays" class="visualization-section">
      <div class="options-panel">
        <div class="options-grid">
          <div class="option-group">
            <label>Sélectionner un pays</label>
            <country-selector :value="selectedCountry" @country-changed="updateSelectedCountry" @show-error="showError" />
          </div>
        </div>
      </div>
  
      <country-stats :stats="countryStats" />
      <country-chart :selectedCountry="selectedCountry" @toggle-loading="setLoading" @show-error="showError" @update-stats="updateCountryStats" @country-changed="updateSelectedCountry" />
      <loading-indicator :is-loading="isLoading" />
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  import CountrySelector from './CountrySelector.vue';
  import CountryStats from './CountryStats.vue';
  import CountryChart from '../Charts/CountryChart.vue';
  import LoadingIndicator from './LoadingIndicator.vue';
  
  const selectedCountry = ref('France');
  const countryStats = ref({
    confirmed: null,
    deaths: null,
    recovered: null,
    active: null,
    mortalityRate: null
  });
  const isLoading = ref(false);
  
  function updateSelectedCountry(country) {
    selectedCountry.value = country;
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
  </script>
  
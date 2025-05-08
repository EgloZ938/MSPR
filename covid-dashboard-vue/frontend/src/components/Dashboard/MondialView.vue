<template>
    <div id="mondial" class="visualization-section">
      <global-stats @toggle-dataset="toggleDataset" @show-error="showError" />
      <world-chart ref="worldChart" @toggle-loading="setLoading" @show-error="showError" />
      <loading-indicator :is-loading="isLoading" />
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  import GlobalStats from './GlobalStats.vue';
  import WorldChart from '../Charts/WorldChart.vue';
  import LoadingIndicator from './LoadingIndicator.vue';
  
  const isLoading = ref(false);
  const worldChart = ref(null);
  
  function toggleDataset(datasetName) {
    if (worldChart.value) {
      worldChart.value.toggleDataset(datasetName);
    }
  }
  
  function showError(message) {
    emit('show-error', message);
  }
  
  function setLoading(loading) {
    isLoading.value = loading;
  }
  </script>
  
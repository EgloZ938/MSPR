<template>
  <div id="regions" class="visualization-section">
    <region-chart @toggle-loading="setLoading" @show-error="showError" />
    <loading-indicator :is-loading="isLoading" />
  </div>
</template>

<script setup>
import { ref } from 'vue';
import RegionChart from '../Charts/RegionChart.vue';
import LoadingIndicator from './LoadingIndicator.vue';

// Définir l'émetteur d'événements
const emit = defineEmits(['show-error']);

const isLoading = ref(false);

function showError(message) {
  emit('show-error', message);
}

function setLoading(loading) {
  isLoading.value = loading;
}
</script>
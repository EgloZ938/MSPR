<template>
  <div class="country-stats-grid">
    <div class="stat-card">
      <h3>Cas Confirmés</h3>
      <div class="number" id="countryConfirmed">{{ formatNumber(stats.confirmed) }}</div>
    </div>
    <div class="stat-card">
      <h3>Décès</h3>
      <div class="number" id="countryDeaths">{{ formatNumber(stats.deaths) }}</div>
    </div>
    <div class="stat-card">
      <h3>Guéris</h3>
      <div class="number" id="countryRecovered">{{ formatNumber(stats.recovered) }}</div>
    </div>
    <div class="stat-card">
      <h3>Cas Actifs</h3>
      <div class="number" id="countryActive">{{ formatNumber(stats.active) }}</div>
    </div>
    <div class="stat-card">
      <h3>Taux de mortalité</h3>
      <div class="number" id="countryMortalityRate">{{ stats.mortalityRate ? `${stats.mortalityRate.toFixed(2)}%` :
        '...' }}</div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue';

const props = defineProps({
  stats: {
    type: Object,
    default: () => ({
      confirmed: null,
      deaths: null,
      recovered: null,
      active: null,
      mortalityRate: null
    })
  }
});

function formatNumber(num) {
  if (num === null || num === undefined) return '...';
  return new Intl.NumberFormat().format(num);
}
</script>

<style scoped></style>
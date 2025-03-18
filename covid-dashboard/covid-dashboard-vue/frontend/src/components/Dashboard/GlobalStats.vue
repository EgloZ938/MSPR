<template>
    <div class="stats-grid">
      <stats-card v-for="(stat, idx) in stats" :key="idx" :title="stat.title" :value="formatNumber(stat.value)" :id="stat.id" :dataset-name="stat.datasetName" @toggle="toggleDataset" />
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import StatsCard from './StatsCard.vue';
  import axios from 'axios';
  
  const stats = ref([
    { title: 'Cas Confirmés', value: null, id: 'confirmed', datasetName: 'confirmed' },
    { title: 'Décès', value: null, id: 'deaths', datasetName: 'deaths' },
    { title: 'Guéris', value: null, id: 'recovered', datasetName: 'recovered' },
    { title: 'Cas Actifs', value: null, id: 'active', datasetName: 'active' }
  ]);
  
  onMounted(() => {
    loadGlobalStats();
  });
  
  async function loadGlobalStats() {
    try {
      const response = await axios.get('/api/global-stats');
      const data = response.data;
  
      // Mettre à jour les valeurs
      stats.value.find(s => s.id === 'confirmed').value = data.total_confirmed;
      stats.value.find(s => s.id === 'deaths').value = data.total_deaths;
      stats.value.find(s => s.id === 'recovered').value = data.total_recovered;
      stats.value.find(s => s.id === 'active').value = data.total_active;
    } catch (error) {
      console.error('Erreur:', error);
      emit('show-error', 'Erreur lors du chargement des statistiques globales');
    }
  }
  
  function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return new Intl.NumberFormat().format(num);
  }
  
  function toggleDataset(datasetName) {
    emit('toggle-dataset', datasetName);
  }
  </script>
  
  <style scoped>
  /* Les styles seront lus du fichier assets/style.css global */
  </style>
  
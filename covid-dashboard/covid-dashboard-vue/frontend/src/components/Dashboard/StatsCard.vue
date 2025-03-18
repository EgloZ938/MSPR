<template>
    <div class="stat-card" @click="emit('toggle', datasetName)" :data-tooltip="tooltip">
      <h3>{{ title }}</h3>
      <div class="number" :id="id">{{ value || '...' }}</div>
    </div>
  </template>
  
  <script setup>
  import { defineProps } from 'vue';
  
  defineProps({
    title: {
      type: String,
      required: true
    },
    value: {
      type: [String, Number],
      default: null
    },
    id: {
      type: String,
      required: true
    },
    datasetName: {
      type: String,
      required: true
    },
    tooltip: {
      type: String,
      default: "Cliquez pour afficher/masquer dans le graphique"
    }
  });
  </script>
  
  <style scoped>
  /* Les styles seront lus du fichier assets/style.css global */
  </style>
  
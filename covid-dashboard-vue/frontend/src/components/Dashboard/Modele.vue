<template>
  <div id="modele" class="visualization-section">
    <div class="coming-soon">
      <h2>Modèle prédictif</h2>
      <p>Cette fonctionnalité sera disponible prochainement.</p>
    </div>
  </div>
</template>

<script setup>
import { defineEmits } from 'vue';

// Définir l'émetteur d'événements
const emit = defineEmits(['show-error']);
</script>

<style scoped>
.coming-soon {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  background-color: var(--card-bg);
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.coming-soon h2 {
  color: var(--primary-color);
  margin-bottom: 20px;
}

.coming-soon p {
  color: var(--text-secondary);
}
</style>
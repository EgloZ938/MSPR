<template>
  <nav class="navbar">
    <div class="nav-container">
      <h1>COVID-19 Dashboard</h1>
      <div class="nav-links">
        <a href="#" class="nav-link" :class="{ active: activeView === 'mondial', disabled: isCoolingDown }"
          @click.prevent="handleChangeView('mondial')" data-tooltip="Statistiques mondiales">Vue Mondiale</a>
        <a href="#" class="nav-link" :class="{ active: activeView === 'regions', disabled: isCoolingDown }"
          @click.prevent="handleChangeView('regions')" data-tooltip="Analyse par région">Par Région</a>
        <a href="#" class="nav-link" :class="{ active: activeView === 'pays', disabled: isCoolingDown }"
          @click.prevent="handleChangeView('pays')" data-tooltip="Données par pays">Par Pays</a>
        <a href="#" class="nav-link" :class="{ active: activeView === 'correlation', disabled: isCoolingDown }"
          @click.prevent="handleChangeView('correlation')" data-tooltip="Analyse des corrélations">Corrélations</a>
        <a href="#" class="nav-link" :class="{ active: activeView === 'tendances', disabled: isCoolingDown }"
          @click.prevent="handleChangeView('tendances')" data-tooltip="Analyse des tendances">Tendances</a>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { defineProps, defineEmits, ref } from 'vue';

const props = defineProps({
  activeView: {
    type: String,
    required: true
  }
});

const emit = defineEmits(['change-view']);
const isCoolingDown = ref(false);

function activateCooldown(duration = 800) {
  isCoolingDown.value = true;
  setTimeout(() => {
    isCoolingDown.value = false;
  }, duration);
}

function handleChangeView(view) {
  // Ne rien faire si la vue est déjà active ou si un cooldown est en cours
  if (view === props.activeView || isCoolingDown.value) return;

  // Activer le cooldown pour éviter les clics multiples rapides
  activateCooldown();

  // Changer la vue
  emit('change-view', view);
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
  opacity: 0.7;
  cursor: not-allowed;
  pointer-events: none;
}
</style>
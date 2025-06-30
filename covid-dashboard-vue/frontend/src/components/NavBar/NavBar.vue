<template>
  <nav class="navbar">
    <div class="nav-container">
      <h1 class="navbar-title">COVID-19 Dashboard</h1>

      <!-- Navigation desktop -->
      <div class="nav-links desktop-nav">
        <a v-for="item in menuItems" :key="item.view" href="#" class="nav-link"
          :class="{ active: activeView === item.view, disabled: isCoolingDown }"
          @click.prevent="handleChangeView(item.view)" :data-tooltip="item.tooltip">
          {{ item.label }}
        </a>
      </div>

      <!-- Menu burger pour mobile -->
      <BurgerMenu :active-view="activeView" @change-view="handleChangeView" />
    </div>
  </nav>
</template>

<script setup>
import { defineProps, defineEmits, ref } from 'vue';
import BurgerMenu from './BurgerMenu.vue';

const props = defineProps({
  activeView: {
    type: String,
    required: true
  }
});

const emit = defineEmits(['change-view']);
const isCoolingDown = ref(false);

// 🚀 MODIFIE cette liste pour ajouter l'onglet prédictif
const menuItems = [
  { view: 'mondial', label: 'Vue Mondiale', tooltip: 'Statistiques mondiales' },
  { view: 'regions', label: 'Par Région', tooltip: 'Analyse par région' },
  { view: 'pays', label: 'Par Pays', tooltip: 'Données par pays' },

  { view: 'predictive', label: '🤖 Prédictions IA', tooltip: 'Intelligence Artificielle Révolutionnaire' },

  { view: 'correlation', label: 'Corrélations', tooltip: 'Analyse des corrélations' },
  { view: 'modele', label: 'Modèle', tooltip: 'Ancien modèle prédictif' },
];

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
/* Base styles from global CSS */

/* Responsive adjustments */
.navbar-title {
  font-size: 1.2rem;
  white-space: nowrap;
  margin: 0;
}

.desktop-nav {
  display: flex;
}

/* Media Queries */
@media (max-width: 768px) {
  .desktop-nav {
    display: none;
    /* Cacher la navigation desktop sur mobile */
  }

  .navbar-title {
    font-size: 1.1rem;
    /* Titre plus petit sur mobile */
  }

  .nav-container {
    padding: 0 15px;
    /* Padding réduit sur mobile */
  }
}

/* Très petits écrans */
@media (max-width: 350px) {
  .navbar-title {
    font-size: 1rem;
  }
}

.disabled {
  opacity: 0.7;
  cursor: not-allowed;
  pointer-events: none;
}
</style>
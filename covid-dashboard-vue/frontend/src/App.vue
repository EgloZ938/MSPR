<template>
  <div>
    <nav-bar :active-view="activeView" @change-view="changeView" />

    <div class="container">
      <error-message :message="errorMessage" />

      <mondial-view v-if="activeView === 'mondial'" @show-error="showError" />
      <regions-view v-if="activeView === 'regions'" @show-error="showError" />
      <pays-view v-if="activeView === 'pays'" @show-error="showError" />
      <correlation-view v-if="activeView === 'correlation'" @show-error="showError" />

      <!-- 🚀 NOUVEAU: Vue Prédictive -->
      <predictive-view v-if="activeView === 'predictive'" @show-error="showError" />

      <!-- Garde ton modele existant aussi -->
      <tendances-view v-if="activeView === 'modele'" @show-error="showError" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import NavBar from './components/NavBar/NavBar.vue';
import ErrorMessage from './components/Dashboard/ErrorMessage.vue';
import MondialView from './components/Dashboard/MondialView.vue';
import RegionsView from './components/Dashboard/RegionsView.vue';
import PaysView from './components/Dashboard/PaysView.vue';
import CorrelationView from './components/Dashboard/CorrelationView.vue';
import TendancesView from './components/Dashboard/Modele.vue';

// 🚀 NOUVEAU: Import du composant prédictif
import PredictiveView from './components/Dashboard/PredictiveView.vue';

const activeView = ref('mondial');
const errorMessage = ref('');

function changeView(view) {
  activeView.value = view;
  errorMessage.value = '';
}

function showError(message) {
  errorMessage.value = message;
  setTimeout(() => {
    if (errorMessage.value === message) {
      errorMessage.value = '';
    }
  }, 5000);
}
</script>

<style>
@import './assets/style.css';
</style>
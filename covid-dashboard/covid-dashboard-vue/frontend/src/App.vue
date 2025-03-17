<template>
  <div>
    <nav-bar :active-view="activeView" @change-view="changeView" />

    <div class="container">
      <error-message :message="errorMessage" />

      <mondial-view v-if="activeView === 'mondial'" @show-error="showError" />

      <regions-view v-if="activeView === 'regions'" @show-error="showError" />

      <pays-view v-if="activeView === 'pays'" @show-error="showError" />

      <correlation-view v-if="activeView === 'correlation'" @show-error="showError" />

      <tendances-view v-if="activeView === 'tendances'" @show-error="showError" />
    </div>
  </div>
</template>

<script>
import NavBar from './components/NavBar/NavBar.vue';
import ErrorMessage from './components/Dashboard/ErrorMessage.vue';
import MondialView from './components/Dashboard/MondialView.vue';
import RegionsView from './components/Dashboard/RegionsView.vue';
import PaysView from './components/Dashboard/PaysView.vue';
import CorrelationView from './components/Dashboard/CorrelationView.vue';
import TendancesView from './components/Dashboard/TendancesView.vue';

export default {
  name: 'App',
  components: {
    NavBar,
    ErrorMessage,
    MondialView,
    RegionsView,
    PaysView,
    CorrelationView,
    TendancesView
  },
  data() {
    return {
      activeView: 'mondial',
      errorMessage: ''
    }
  },
  methods: {
    changeView(view) {
      this.activeView = view;
      // Reset error message when changing views
      this.errorMessage = '';
    },
    showError(message) {
      this.errorMessage = message;
      // Auto-dismiss error after 5 seconds
      setTimeout(() => {
        if (this.errorMessage === message) {
          this.errorMessage = '';
        }
      }, 5000);
    }
  }
}
</script>

<style>
@import './assets/style.css';
</style>
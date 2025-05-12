<template>
  <div class="stat-card" @click="handleToggle" :data-tooltip="tooltip" :class="{ 'disabled': isCoolingDown }">
    <h3>{{ title }}</h3>
    <div class="number" :id="id">{{ value || '...' }}</div>

    <!-- Miniloader pour indiquer visuellement le cooldown -->
    <div v-if="isCoolingDown" class="stats-loading-indicator"></div>
  </div>
</template>

<script setup>
import { defineProps, defineEmits, ref } from 'vue';

const props = defineProps({
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

const emit = defineEmits(['toggle']);
const isCoolingDown = ref(false);

function activateCooldown(duration = 500) {
  isCoolingDown.value = true;
  setTimeout(() => {
    isCoolingDown.value = false;
  }, duration);
}

function handleToggle() {
  if (isCoolingDown.value) return;

  activateCooldown();
  emit('toggle', props.datasetName);
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
  opacity: 0.8;
  cursor: wait;
}

.stats-loading-indicator {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 2px solid var(--primary-color);
  border-top-color: transparent;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style>
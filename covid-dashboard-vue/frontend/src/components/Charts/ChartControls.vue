<template>
    <div class="chart-controls">
        <button class="control-btn" @click="handleZoomIn" :disabled="isCoolingDown" data-tooltip="Zoom +"
            :class="{ 'disabled': isCoolingDown }">
            <i class="fas fa-search-plus"></i> +
        </button>
        <button class="control-btn" @click="handleZoomOut" :disabled="isCoolingDown" data-tooltip="Zoom -"
            :class="{ 'disabled': isCoolingDown }">
            <i class="fas fa-search-minus"></i> -
        </button>
        <button class="control-btn" @click="handleResetZoom" :disabled="isCoolingDown"
            data-tooltip="Réinitialiser le zoom" :class="{ 'disabled': isCoolingDown }">Reset</button>
        <button class="control-btn" @click="handleDownload" :disabled="isCoolingDown"
            data-tooltip="Télécharger le graphique" :class="{ 'disabled': isCoolingDown }">Télécharger</button>
        <button class="control-btn" @click="handleExport" :disabled="isCoolingDown"
            data-tooltip="Exporter les données en CSV" :class="{ 'disabled': isCoolingDown }">Exporter CSV</button>

        <!-- Mini Loader -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref } from 'vue';
import MiniLoader from '../MiniLoader.vue';

const emit = defineEmits(['zoom-in', 'zoom-out', 'reset-zoom', 'download', 'export']);
const isCoolingDown = ref(false);

// Fonction pour activer le cooldown
function activateCooldown(duration = 800) {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, duration);
}

// Fonctions avec cooldown
function handleZoomIn() {
    if (isCoolingDown.value) return;
    activateCooldown(500);
    emit('zoom-in');
}

function handleZoomOut() {
    if (isCoolingDown.value) return;
    activateCooldown(500);
    emit('zoom-out');
}

function handleResetZoom() {
    if (isCoolingDown.value) return;
    activateCooldown(500);
    emit('reset-zoom');
}

function handleDownload() {
    if (isCoolingDown.value) return;
    activateCooldown(1000); // Plus long pour le téléchargement
    emit('download');
}

function handleExport() {
    if (isCoolingDown.value) return;
    activateCooldown(1000); // Plus long pour l'export
    emit('export');
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
</style>
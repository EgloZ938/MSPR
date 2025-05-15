<template>
    <div class="chart-controls-responsive">
        <!-- En desktop, tout s'affichera sur une ligne grâce au CSS -->
        <div class="desktop-row">
            <!-- Première rangée de boutons (zoom et reset) -->
            <div class="controls-row controls-row-1">
                <button class="control-btn zoom-btn" @click="handleZoomIn" :disabled="isCoolingDown"
                    data-tooltip="Zoom +" :class="{ 'disabled': isCoolingDown }">
                    <i class="fas fa-search-plus"></i>
                </button>
                <button class="control-btn zoom-btn" @click="handleZoomOut" :disabled="isCoolingDown"
                    data-tooltip="Zoom -" :class="{ 'disabled': isCoolingDown }">
                    <i class="fas fa-search-minus"></i>
                </button>
                <button class="control-btn" @click="handleResetZoom" :disabled="isCoolingDown"
                    data-tooltip="Réinitialiser" :class="{ 'disabled': isCoolingDown }">
                    <span class="btn-text">Reset</span>
                    <i class="fas fa-undo-alt btn-icon"></i>
                </button>
            </div>

            <!-- Deuxième rangée de boutons (télécharger et exporter) -->
            <div class="controls-row controls-row-2">
                <button class="control-btn" @click="handleDownload" :disabled="isCoolingDown" data-tooltip="Télécharger"
                    :class="{ 'disabled': isCoolingDown }">
                    <span class="btn-text">Télécharger</span>
                    <i class="fas fa-download btn-icon"></i>
                </button>
                <button class="control-btn" @click="handleExport" :disabled="isCoolingDown" data-tooltip="Exporter CSV"
                    :class="{ 'disabled': isCoolingDown }">
                    <span class="btn-text">Exporter CSV</span>
                    <i class="fas fa-file-csv btn-icon"></i>
                </button>
            </div>
        </div>

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
.chart-controls-responsive {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 12px;
    position: relative;
}

.desktop-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
}

.controls-row {
    display: flex;
    gap: 8px;
    width: 100%;
}

.control-btn {
    padding: 8px 12px;
    border: none;
    border-radius: 4px;
    background: var(--primary-color);
    color: white;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.1s ease;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    flex: 1;
}

.zoom-btn {
    flex: 0.5;
}

.control-btn:hover {
    background: var(--hover-color);
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-btn:active {
    transform: translateY(0);
}

.btn-icon {
    font-size: 14px;
}

/* Desktop styles */
@media (min-width: 769px) {
    .desktop-row {
        flex-direction: row;
        align-items: center;
        gap: 10px;
    }

    .controls-row {
        flex: none;
        width: auto;
        display: flex;
        align-items: center;
        /* Assure l'alignement vertical */
    }

    .control-btn {
        flex: 0 0 auto;
        min-width: auto;
        white-space: nowrap;
        height: 40px !important;
        /* Hauteur fixe pour tous les boutons en desktop */
        padding: 0 16px;
        /* Padding horizontal plus généreux */
        font-size: 14px;
        /* Taille de police uniforme */
    }

    .zoom-btn {
        min-width: 40px;
        width: 40px;
        padding: 0;
        /* Reset padding pour ces boutons */
    }

    /* Assurer que les boutons télécharger et exporter ont une taille visuelle suffisante */
    .controls-row-2 .control-btn {
        padding: 0 20px;
        /* Padding horizontal plus grand pour ces boutons */
        font-weight: 500;
        /* Un peu moins gras pour équilibrer visuellement */
    }
}

/* Tablet styles */
@media (max-width: 768px) {
    .control-btn {
        padding: 8px 10px;
        font-size: 0.9em;
    }
}

/* Mobile styles */
@media (max-width: 576px) {
    .btn-text {
        font-size: 0.9em;
    }

    .control-btn {
        padding: 7px 8px;
    }
}

/* Very small screens */
@media (max-width: 350px) {
    .controls-row {
        gap: 5px;
    }

    .btn-text {
        display: none;
    }

    .btn-icon {
        font-size: 16px;
        margin: 0;
    }

    .control-btn {
        padding: 8px;
    }
}

.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
</style>
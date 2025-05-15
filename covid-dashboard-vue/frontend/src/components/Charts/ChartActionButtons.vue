<!-- Composant optimisé pour les boutons d'action des graphiques -->
<template>
    <div class="chart-action-buttons">
        <div class="action-buttons-row">
            <button class="action-btn zoom-btn" @click="handleZoomIn" :disabled="isCoolingDown">
                <i class="fas fa-search-plus"></i>
                <span class="btn-text">Zoom +</span>
            </button>
            <button class="action-btn zoom-btn" @click="handleZoomOut" :disabled="isCoolingDown">
                <i class="fas fa-search-minus"></i>
                <span class="btn-text">Zoom -</span>
            </button>
            <button class="action-btn reset-btn" @click="handleResetZoom" :disabled="isCoolingDown">
                <i class="fas fa-undo-alt"></i>
                <span class="btn-text">Reset</span>
            </button>
        </div>

        <div class="action-buttons-row">
            <button class="action-btn download-btn" @click="handleDownload" :disabled="isCoolingDown">
                <i class="fas fa-download"></i>
                <span class="btn-text">Télécharger</span>
            </button>
            <button class="action-btn export-btn" @click="handleExport" :disabled="isCoolingDown">
                <i class="fas fa-file-csv"></i>
                <span class="btn-text">Exporter CSV</span>
            </button>
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
.chart-action-buttons {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 20px;
    position: relative;
}

.action-buttons-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    background-color: var(--primary-color);
    color: white;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.1s ease;
    gap: 6px;
    flex: 1;
    min-width: 120px;
    height: auto;
    box-sizing: border-box;
}

.action-btn:hover:not(:disabled) {
    background-color: var(--hover-color);
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.action-btn:active:not(:disabled) {
    transform: translateY(0);
}

.zoom-btn {
    min-width: 100px;
}

.action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.action-btn i {
    font-size: 14px;
}

/* Média queries pour l'affichage responsive */
@media (min-width: 768px) {
    .chart-action-buttons {
        flex-direction: row;
        align-items: center;
    }

    .action-buttons-row {
        flex: 1;
    }

    .action-btn {
        min-width: auto;
    }
}

@media (max-width: 576px) {
    .action-btn {
        padding: 8px 12px;
        min-width: 0;
        flex: 1;
    }

    .btn-text {
        font-size: 13px;
    }
}

@media (max-width: 400px) {
    .action-buttons-row {
        gap: 5px;
    }

    .action-btn {
        padding: 6px 10px;
    }

    .btn-text {
        font-size: 12px;
    }
}

/* Pour les très petits écrans, on peut cacher les textes et ne montrer que les icônes */
@media (max-width: 350px) {
    .btn-text {
        display: none;
    }

    .action-btn {
        padding: 8px;
        min-width: 0;
    }

    .action-btn i {
        font-size: 16px;
        margin: 0;
    }
}
</style>
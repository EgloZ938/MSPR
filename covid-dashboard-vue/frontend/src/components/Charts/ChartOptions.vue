<template>
    <div class="options-panel">
        <div class="options-grid">
            <div class="option-group">
                <label>Type de visualisation</label>
                <select :id="chartTypeId" v-model="localChartType" @change="emitChartTypeChange"
                    data-tooltip="Choisissez le type de graphique" :disabled="isCoolingDown">
                    <option value="line">Ligne</option>
                    <option value="bar">Barres</option>
                    <option value="area">Aire</option>
                    <option v-if="showPieOptions" value="pie">Camembert</option>
                    <option v-if="showPieOptions" value="doughnut">Anneau</option>
                </select>
            </div>

            <div class="option-group">
                <label>Format des données</label>
                <select :id="dataFormatId" v-model="localDataFormat" @change="emitDataFormatChange"
                    data-tooltip="Choisissez le format d'affichage des données" :disabled="isCoolingDown">
                    <option value="raw">Valeurs brutes</option>
                    <option value="daily">Variation quotidienne</option>
                    <option v-if="!isCountryView" value="weekly">Moyenne mobile (7j)</option>
                    <option v-if="!isCountryView" value="monthly">Moyenne mobile (30j)</option>
                </select>
            </div>

            <div class="option-group" v-if="!isCountryView || isLineView">
                <label>Échelle Y</label>
                <select :id="scaleTypeId" v-model="localScaleType" @change="emitScaleTypeChange"
                    data-tooltip="L'échelle logarithmique montre mieux les taux de croissance" :disabled="isCoolingDown">
                    <option value="linear">Linéaire (standard)</option>
                    <option value="logarithmic">Logarithmique (puissances de 10)</option>
                </select>
            </div>
        </div>

        <div class="options-grid" style="margin-top: 20px;">
            <div class="option-group">
                <label>Datasets visibles</label>
                <div class="dataset-toggles">
                    <label v-for="(dataset, key) in datasets" :key="key" class="toggle-item"
                        :class="{ 'disabled': isCoolingDown }">
                        <input type="checkbox" :id="`toggle${key.charAt(0).toUpperCase() + key.slice(1)}`"
                            :checked="dataset" @change="toggleDataset(key)" :disabled="isCoolingDown">
                        <span>{{ getDatasetLabel(key) }}</span>
                    </label>
                </div>
            </div>

            <div class="option-group" v-if="showColors">
                <label>Personnalisation des couleurs</label>
                <div class="color-options">
                    <div v-for="(color, key) in colors" :key="key" class="color-option"
                        :class="{ 'disabled': isCoolingDown }">
                        <input type="color" :id="`${key}Color`" :value="color"
                            @change="updateColor(key, $event.target.value)" :disabled="isCoolingDown">
                        <span>{{ getDatasetLabel(key) }}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mini Loader Overlay -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref, watch, defineProps, defineEmits, computed } from 'vue';
import { withCooldown } from '../../utils/cooldown';
import MiniLoader from '../MiniLoader.vue';

const props = defineProps({
    chartType: {
        type: String,
        default: 'line'
    },
    dataFormat: {
        type: String,
        default: 'raw'
    },
    scaleType: {
        type: String,
        default: 'linear'
    },
    chartTypeId: {
        type: String,
        default: 'chartType'
    },
    dataFormatId: {
        type: String,
        default: 'dataFormat'
    },
    scaleTypeId: {
        type: String,
        default: 'scaleType'
    },
    datasets: {
        type: Object,
        required: true
    },
    colors: {
        type: Object,
        required: true
    },
    isCountryView: {
        type: Boolean,
        default: false
    },
    showPieOptions: {
        type: Boolean,
        default: false
    },
    showColors: {
        type: Boolean,
        default: true
    }
});

const emit = defineEmits(['chart-type-change', 'data-format-change', 'scale-type-change', 'toggle-dataset', 'update-color']);

const localChartType = ref(props.chartType);
const localDataFormat = ref(props.dataFormat);
const localScaleType = ref(props.scaleType);
const isCoolingDown = ref(false);

const isLineView = computed(() => localChartType.value === 'line');

watch(() => props.chartType, (newValue) => {
    localChartType.value = newValue;
});

watch(() => props.dataFormat, (newValue) => {
    localDataFormat.value = newValue;
});

watch(() => props.scaleType, (newValue) => {
    localScaleType.value = newValue;
});

// Fonction pour activer le cooldown
function activateCooldown() {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, 800); // Cooldown de 800ms
}

// Fonctions avec cooldown
const emitChartTypeChange = function () {
    if (isCoolingDown.value) return;
    activateCooldown();
    emit('chart-type-change', localChartType.value);
};

const emitDataFormatChange = function () {
    if (isCoolingDown.value) return;
    activateCooldown();
    emit('data-format-change', localDataFormat.value);
};

const emitScaleTypeChange = function () {
    if (isCoolingDown.value) return;
    activateCooldown();
    emit('scale-type-change', localScaleType.value);
};

const toggleDataset = function (datasetName) {
    if (isCoolingDown.value) return;
    activateCooldown();
    emit('toggle-dataset', datasetName);
};

const updateColor = function (datasetName, color) {
    if (isCoolingDown.value) return;
    activateCooldown();
    emit('update-color', { datasetName, color });
};

function getDatasetLabel(key) {
    const labels = {
        confirmed: 'Cas confirmés',
        deaths: 'Décès',
        recovered: 'Guéris',
        active: 'Cas actifs'
    };
    return labels[key] || key;
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.toggle-item.disabled {
    pointer-events: none;
}

.color-option.disabled {
    pointer-events: none;
}
</style>
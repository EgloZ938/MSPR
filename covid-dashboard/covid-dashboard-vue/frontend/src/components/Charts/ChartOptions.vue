<template>
    <div class="options-panel">
        <div class="options-grid">
            <div class="option-group">
                <label>Type de visualisation</label>
                <select :id="chartTypeId" v-model="localChartType" @change="emitChartTypeChange"
                    data-tooltip="Choisissez le type de graphique">
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
                    data-tooltip="Choisissez le format d'affichage des données">
                    <option value="raw">Valeurs brutes</option>
                    <option value="daily">Variation quotidienne</option>
                    <option v-if="!isCountryView" value="weekly">Moyenne mobile (7j)</option>
                    <option v-if="!isCountryView" value="monthly">Moyenne mobile (30j)</option>
                </select>
            </div>

            <div class="option-group" v-if="!isCountryView || isLineView">
                <label>Échelle Y</label>
                <select :id="scaleTypeId" v-model="localScaleType" @change="emitScaleTypeChange"
                    data-tooltip="Choisissez l'échelle de l'axe Y">
                    <option value="linear">Linéaire</option>
                    <option value="logarithmic">Logarithmique</option>
                </select>
            </div>
        </div>

        <div class="options-grid" style="margin-top: 20px;">
            <div class="option-group">
                <label>Datasets visibles</label>
                <div class="dataset-toggles">
                    <label v-for="(dataset, key) in datasets" :key="key" class="toggle-item">
                        <input type="checkbox" :id="`toggle${key.charAt(0).toUpperCase() + key.slice(1)}`"
                            :checked="dataset" @change="toggleDataset(key)">
                        <span>{{ getDatasetLabel(key) }}</span>
                    </label>
                </div>
            </div>

            <div class="option-group" v-if="showColors">
                <label>Personnalisation des couleurs</label>
                <div class="color-options">
                    <div v-for="(color, key) in colors" :key="key" class="color-option">
                        <input type="color" :id="`${key}Color`" :value="color"
                            @change="updateColor(key, $event.target.value)">
                        <span>{{ getDatasetLabel(key) }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'ChartOptions',
    props: {
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
    },
    data() {
        return {
            localChartType: this.chartType,
            localDataFormat: this.dataFormat,
            localScaleType: this.scaleType
        }
    },
    computed: {
        isLineView() {
            return this.localChartType === 'line';
        }
    },
    watch: {
        chartType(newValue) {
            this.localChartType = newValue;
        },
        dataFormat(newValue) {
            this.localDataFormat = newValue;
        },
        scaleType(newValue) {
            this.localScaleType = newValue;
        }
    },
    methods: {
        emitChartTypeChange() {
            this.$emit('chart-type-change', this.localChartType);
        },
        emitDataFormatChange() {
            this.$emit('data-format-change', this.localDataFormat);
        },
        emitScaleTypeChange() {
            this.$emit('scale-type-change', this.localScaleType);
        },
        toggleDataset(datasetName) {
            this.$emit('toggle-dataset', datasetName);
        },
        updateColor(datasetName, color) {
            this.$emit('update-color', { datasetName, color });
        },
        getDatasetLabel(key) {
            const labels = {
                confirmed: 'Cas confirmés',
                deaths: 'Décès',
                recovered: 'Guéris',
                active: 'Cas actifs'
            };
            return labels[key] || key;
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
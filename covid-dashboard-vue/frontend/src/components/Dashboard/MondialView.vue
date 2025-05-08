<template>
    <div id="mondial" class="visualization-section">
        <global-stats @toggle-dataset="toggleDataset" @show-error="showError" />

        <world-chart ref="worldChart" @toggle-loading="setLoading" @show-error="showError" />

        <loading-indicator :is-loading="isLoading" />
    </div>
</template>

<script>
import GlobalStats from './GlobalStats.vue';
import WorldChart from '../Charts/WorldChart.vue';
import LoadingIndicator from './LoadingIndicator.vue';

export default {
    name: 'MondialView',
    components: {
        GlobalStats,
        WorldChart,
        LoadingIndicator
    },
    data() {
        return {
            isLoading: false
        }
    },
    methods: {
        toggleDataset(datasetName) {
            // Propager l'événement au composant WorldChart
            if (this.$refs.worldChart) {
                this.$refs.worldChart.toggleDataset(datasetName);
            }
        },
        showError(message) {
            this.$emit('show-error', message);
        },
        setLoading(loading) {
            this.isLoading = loading;
        }
    }
}
</script>
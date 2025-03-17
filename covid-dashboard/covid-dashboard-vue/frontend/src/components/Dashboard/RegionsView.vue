<template>
    <div id="regions" class="visualization-section">
        <region-chart @toggle-loading="setLoading" @show-error="showError" />

        <loading-indicator :is-loading="isLoading" />
    </div>
</template>

<script>
import RegionChart from '../Charts/RegionChart.vue';
import LoadingIndicator from './LoadingIndicator.vue';

export default {
    name: 'RegionsView',
    components: {
        RegionChart,
        LoadingIndicator
    },
    data() {
        return {
            isLoading: false
        }
    },
    methods: {
        showError(message) {
            this.$emit('show-error', message);
        },
        setLoading(loading) {
            this.isLoading = loading;
        }
    }
}
</script>
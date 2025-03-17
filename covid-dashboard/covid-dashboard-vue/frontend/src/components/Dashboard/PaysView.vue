<template>
    <div id="pays" class="visualization-section">
        <div class="options-panel">
            <div class="options-grid">
                <div class="option-group">
                    <label>Sélectionner un pays</label>
                    <country-selector :value="selectedCountry" @country-changed="updateSelectedCountry"
                        @show-error="showError" />
                </div>
            </div>
        </div>

        <country-stats :stats="countryStats" />

        <country-chart :selectedCountry="selectedCountry" @toggle-loading="setLoading" @show-error="showError"
            @update-stats="updateCountryStats" @country-changed="updateSelectedCountry" />

        <loading-indicator :is-loading="isLoading" />
    </div>
</template>

<script>
import CountrySelector from './CountrySelector.vue';
import CountryStats from './CountryStats.vue';
import CountryChart from '../Charts/CountryChart.vue';
import LoadingIndicator from './LoadingIndicator.vue';

export default {
    name: 'PaysView',
    components: {
        CountrySelector,
        CountryStats,
        CountryChart,
        LoadingIndicator
    },
    data() {
        return {
            selectedCountry: 'France',
            countryStats: {
                confirmed: null,
                deaths: null,
                recovered: null,
                active: null,
                mortalityRate: null
            },
            isLoading: false
        }
    },
    methods: {
        updateSelectedCountry(country) {
            this.selectedCountry = country;
        },
        updateCountryStats(stats) {
            this.countryStats = stats;
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
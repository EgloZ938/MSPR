<template>
    <div class="country-selector-custom">
        <div class="selected-country" id="selectedCountryDisplay" @click="toggleDropdown">
            <span id="selectedCountryText">{{ selectedCountry }}</span>
            <i class="dropdown-icon">▼</i>
        </div>
        <div class="dropdown-container" id="countriesDropdown" :class="{ 'active': isDropdownActive }">
            <div class="search-container">
                <input type="text" id="countrySearch" placeholder="Rechercher un pays..." class="search-input"
                    v-model="searchTerm" @click.stop>
            </div>
            <div class="countries-list" id="countriesList">
                <div v-for="country in filteredCountries" :key="country.country_name" class="country-item"
                    :class="{ 'selected': country.country_name === selectedCountry }"
                    @click="selectCountry(country.country_name)">
                    {{ country.country_name }}
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: 'CountrySelector',
    props: {
        value: {
            type: String,
            default: 'France'
        }
    },
    data() {
        return {
            selectedCountry: this.value,
            countries: [],
            isDropdownActive: false,
            searchTerm: ''
        }
    },
    computed: {
        filteredCountries() {
            if (!this.searchTerm) return this.countries;

            const term = this.searchTerm.toLowerCase();
            return this.countries.filter(country =>
                country.country_name.toLowerCase().includes(term)
            );
        }
    },
    mounted() {
        this.loadCountries();

        // Fermer le dropdown si on clique ailleurs sur la page
        document.addEventListener('click', this.closeDropdown);
    },
    beforeUnmount() {
        document.removeEventListener('click', this.closeDropdown);
    },
    methods: {
        async loadCountries() {
            try {
                const response = await axios.get('/api/countries');
                this.countries = response.data;
            } catch (error) {
                console.error('Erreur lors du chargement des pays:', error);
                this.$emit('show-error', 'Erreur lors du chargement de la liste des pays');
            }
        },
        toggleDropdown(event) {
            event.stopPropagation();
            this.isDropdownActive = !this.isDropdownActive;

            if (this.isDropdownActive) {
                this.$nextTick(() => {
                    const searchInput = document.getElementById('countrySearch');
                    if (searchInput) {
                        searchInput.focus();
                    }
                });
            }
        },
        closeDropdown(event) {
            if (!event.target.closest('.country-selector-custom')) {
                this.isDropdownActive = false;
                this.searchTerm = '';
            }
        },
        selectCountry(countryName) {
            this.selectedCountry = countryName;
            this.isDropdownActive = false;
            this.searchTerm = '';
            this.$emit('input', countryName);
            this.$emit('country-changed', countryName);
        }
    },
    watch: {
        value(newValue) {
            this.selectedCountry = newValue;
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
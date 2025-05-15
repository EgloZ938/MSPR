<!-- Composant de sélection de pays optimisé pour mobile -->
<template>
    <div class="responsive-country-selector">
        <div class="selected-country" @click="toggleDropdown" :class="{ 'disabled': isCoolingDown }">
            <div class="country-display">
                <span class="label-mobile">Pays:</span>
                <span class="selected-value">{{ selectedCountry }}</span>
            </div>
            <i class="dropdown-icon" :class="{ 'open': isDropdownActive }">▼</i>
        </div>

        <div class="dropdown-container" :class="{ 'active': isDropdownActive }">
            <div class="search-container">
                <div class="search-icon">
                    <i class="fas fa-search"></i>
                </div>
                <input type="text" placeholder="Rechercher un pays..." class="search-input" v-model="searchTerm"
                    @click.stop @input="onSearchInput">
                <button class="close-btn" @click="closeDropdown">
                    <i class="fas fa-times"></i>
                </button>
            </div>

            <div class="countries-list">
                <div v-for="country in filteredCountries" :key="country.country_name" class="country-item"
                    :class="{ 'selected': country.country_name === selectedCountry, 'disabled': isCoolingDown }"
                    @click="selectCountry(country.country_name)">
                    {{ country.country_name }}
                </div>
            </div>
        </div>

        <!-- Overlay qui apparaît en mode mobile -->
        <div class="dropdown-overlay" v-if="isDropdownActive" @click="closeDropdown"></div>

        <!-- Mini Loader -->
        <mini-loader :show="isCoolingDown" />
    </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import axios from 'axios';
import MiniLoader from '../MiniLoader.vue';

const props = defineProps({
    value: {
        type: String,
        default: 'France'
    }
});

// Définir l'émetteur d'événements
const emit = defineEmits(['input', 'country-changed', 'show-error']);

const selectedCountry = ref(props.value);
const countries = ref([]);
const isDropdownActive = ref(false);
const searchTerm = ref('');
const isCoolingDown = ref(false);
const isMobile = ref(window.innerWidth < 768);

// Détecter les changements de taille d'écran
function handleResize() {
    isMobile.value = window.innerWidth < 768;
}

// Filtrer les pays selon le terme de recherche
const filteredCountries = computed(() => {
    if (!searchTerm.value) return countries.value;

    const term = searchTerm.value.toLowerCase();
    return countries.value.filter(country =>
        country.country_name.toLowerCase().includes(term)
    );
});

onMounted(() => {
    loadCountries();
    document.addEventListener('click', closeDropdown);
    window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
    document.removeEventListener('click', closeDropdown);
    window.removeEventListener('resize', handleResize);
});

async function loadCountries() {
    try {
        const response = await axios.get('/api/countries');
        countries.value = response.data;
    } catch (error) {
        console.error('Erreur lors du chargement des pays:', error);
        emit('show-error', 'Erreur lors du chargement de la liste des pays');
    }
}

// Fonction pour activer le cooldown
function activateCooldown(duration = 800) {
    isCoolingDown.value = true;
    setTimeout(() => {
        isCoolingDown.value = false;
    }, duration);
}

function toggleDropdown(event) {
    event.stopPropagation();

    if (isCoolingDown.value) return;

    isDropdownActive.value = !isDropdownActive.value;

    if (isDropdownActive.value) {
        // En mode mobile, empêcher le défilement du body
        if (isMobile.value) {
            document.body.style.overflow = 'hidden';
        }

        nextTick(() => {
            const searchInput = document.querySelector('.responsive-country-selector .search-input');
            if (searchInput) {
                searchInput.focus();
            }
        });
    } else {
        document.body.style.overflow = '';
    }
}

function closeDropdown() {
    if (isDropdownActive.value) {
        isDropdownActive.value = false;
        searchTerm.value = '';
        document.body.style.overflow = '';
    }
}

function selectCountry(countryName) {
    if (isCoolingDown.value) return;

    activateCooldown(1000); // Plus long pour le changement de pays

    selectedCountry.value = countryName;
    closeDropdown();
    emit('input', countryName);
    emit('country-changed', countryName);
}

// Version optimisée du filtrage pour la recherche
function onSearchInput() {
    // Logique simple, pas besoin de debounce pour un cas d'utilisation standard
    console.log('Recherche pour:', searchTerm.value);
}

watch(() => props.value, (newValue) => {
    selectedCountry.value = newValue;
});
</script>

<style scoped>
.responsive-country-selector {
    position: relative;
    width: 100%;
}

.selected-country {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: white;
    cursor: pointer;
    transition: all 0.2s ease;
}

.selected-country:hover {
    border-color: var(--primary-color);
}

.country-display {
    display: flex;
    align-items: center;
    gap: 8px;
}

.label-mobile {
    display: none;
    font-weight: 500;
    color: var(--text-secondary);
}

.selected-value {
    font-weight: 500;
}

.dropdown-icon {
    font-size: 10px;
    color: var(--text-secondary);
    transition: transform 0.2s ease;
}

.dropdown-icon.open {
    transform: rotate(180deg);
}

.dropdown-container {
    position: absolute;
    top: calc(100% + 5px);
    left: 0;
    width: 100%;
    background: white;
    border-radius: 6px;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
    z-index: 100;
    max-height: 0;
    overflow: hidden;
    visibility: hidden;
    opacity: 0;
    transition: all 0.3s ease;
}

.dropdown-container.active {
    max-height: 300px;
    visibility: visible;
    opacity: 1;
}

.search-container {
    padding: 10px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    display: flex;
    align-items: center;
    gap: 8px;
}

.search-icon {
    color: var(--text-secondary);
}

.search-input {
    flex: 1;
    padding: 8px 10px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 14px;
}

.close-btn {
    display: none;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 5px;
}

.close-btn:hover {
    color: var(--primary-color);
}

.countries-list {
    max-height: 250px;
    overflow-y: auto;
}

.country-item {
    padding: 10px 12px;
    cursor: pointer;
    transition: background 0.2s ease;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.country-item:hover {
    background-color: rgba(26, 115, 232, 0.1);
}

.country-item.selected {
    background-color: rgba(26, 115, 232, 0.1);
    font-weight: 500;
}

.dropdown-overlay {
    display: none;
}

/* Styles responsives */
@media (max-width: 768px) {
    .dropdown-container.active {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        max-height: 100%;
        border-radius: 0;
        z-index: 1001;
        display: flex;
        flex-direction: column;
    }

    .label-mobile {
        display: inline;
    }

    .search-container {
        padding: 15px;
        position: sticky;
        top: 0;
        background: white;
        z-index: 2;
        border-bottom: 1px solid var(--border-color);
    }

    .close-btn {
        display: block;
    }

    .search-input {
        padding: 10px;
    }

    .countries-list {
        max-height: none;
        flex: 1;
        overflow-y: auto;
    }

    .country-item {
        padding: 12px 15px;
    }

    .dropdown-overlay {
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
    }
}

.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
}
</style>
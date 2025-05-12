<template>
    <div class="multi-country-selector">
      <div class="selected-countries">
        <div v-for="(country, index) in selectedCountries" :key="index" class="selected-country-tag">
          {{ country }}
          <span class="remove-country" @click="removeCountry(index)">×</span>
        </div>
        <div v-if="selectedCountries.length < maxCountries" 
             class="add-country-btn" 
             @click="toggleDropdown" 
             :class="{ 'disabled': isCoolingDown }"
             data-tooltip="Ajouter un pays (max 3)">
          + Ajouter
        </div>
      </div>
      
      <div class="dropdown-container" id="countriesDropdown" :class="{ 'active': isDropdownActive }">
        <div class="search-container">
          <input type="text" id="countrySearch" placeholder="Rechercher un pays..." class="search-input"
            v-model="searchTerm" @click.stop @input="onSearchInput">
        </div>
        <div class="countries-list" id="countriesList">
          <div v-for="country in filteredAvailableCountries" :key="country.country_name" class="country-item"
            :class="{ 'disabled': isCoolingDown }"
            @click="selectCountry(country.country_name)">
            {{ country.country_name }}
          </div>
        </div>
      </div>
      
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
      type: Array,
      default: () => ['France']
    },
    maxCountries: {
      type: Number,
      default: 3
    }
  });
  
  // Définir l'émetteur d'événements
  const emit = defineEmits(['input', 'countries-changed', 'show-error']);
  
  const selectedCountries = ref(props.value || []);
  const countries = ref([]);
  const isDropdownActive = ref(false);
  const searchTerm = ref('');
  const isCoolingDown = ref(false);
  
  // Filtrer les pays disponibles (retirer ceux déjà sélectionnés)
  const availableCountries = computed(() => {
    return countries.value.filter(country => 
      !selectedCountries.value.includes(country.country_name)
    );
  });
  
  const filteredAvailableCountries = computed(() => {
    if (!searchTerm.value) return availableCountries.value;
  
    const term = searchTerm.value.toLowerCase();
    return availableCountries.value.filter(country =>
      country.country_name.toLowerCase().includes(term)
    );
  });
  
  onMounted(() => {
    loadCountries();
    document.addEventListener('click', closeDropdown);
  });
  
  onBeforeUnmount(() => {
    document.removeEventListener('click', closeDropdown);
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
  
  // Version debounce du filtrage pour éviter des calculs inutiles lors de la saisie rapide
  function onSearchInput() {
    // Ce callback sera exécuté après un court délai après la dernière frappe
    console.log('Recherche pour:', searchTerm.value);
  }
  
  function toggleDropdown(event) {
    event.stopPropagation();
    
    if (isCoolingDown.value) return;
    
    isDropdownActive.value = !isDropdownActive.value;
  
    if (isDropdownActive.value) {
      nextTick(() => {
        const searchInput = document.getElementById('countrySearch');
        if (searchInput) {
          searchInput.focus();
        }
      });
    }
  }
  
  function closeDropdown(event) {
    if (!event.target.closest('.multi-country-selector')) {
      isDropdownActive.value = false;
      searchTerm.value = '';
    }
  }
  
  function selectCountry(countryName) {
    if (isCoolingDown.value) return;
    if (selectedCountries.value.length >= props.maxCountries) return;
    
    activateCooldown(800);
    
    // Ajouter le pays s'il n'est pas déjà présent
    if (!selectedCountries.value.includes(countryName)) {
      const newSelectedCountries = [...selectedCountries.value, countryName];
      selectedCountries.value = newSelectedCountries;
      emit('input', newSelectedCountries);
      emit('countries-changed', newSelectedCountries);
    }
    
    isDropdownActive.value = false;
    searchTerm.value = '';
  }
  
  function removeCountry(index) {
    if (isCoolingDown.value) return;
    if (selectedCountries.value.length <= 1) return; // Garder au moins un pays
    
    activateCooldown(800);
    
    const newSelectedCountries = [...selectedCountries.value];
    newSelectedCountries.splice(index, 1);
    selectedCountries.value = newSelectedCountries;
    emit('input', newSelectedCountries);
    emit('countries-changed', newSelectedCountries);
  }
  
  watch(() => props.value, (newValue) => {
    if (newValue && Array.isArray(newValue)) {
      selectedCountries.value = newValue;
    }
  });
  </script>
  
  <style scoped>
  .multi-country-selector {
    position: relative;
    width: 100%;
  }
  
  .selected-countries {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: white;
    min-height: 42px;
  }
  
  .selected-country-tag {
    display: flex;
    align-items: center;
    background-color: rgba(26, 115, 232, 0.1);
    color: var(--primary-color);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 14px;
  }
  
  .remove-country {
    margin-left: 6px;
    cursor: pointer;
    font-weight: bold;
    font-size: 16px;
  }
  
  .add-country-btn {
    display: flex;
    align-items: center;
    color: var(--primary-color);
    padding: 4px 12px;
    border-radius: 16px;
    border: 1px dashed var(--primary-color);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .add-country-btn:hover {
    background-color: rgba(26, 115, 232, 0.05);
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
    padding: 8px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  }
  
  .search-input {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    font-size: 14px;
  }
  
  .search-input:focus {
    outline: none;
    border-color: var(--primary-color);
  }
  
  .countries-list {
    max-height: 250px;
    overflow-y: auto;
  }
  
  .country-item {
    padding: 8px 12px;
    cursor: pointer;
    transition: background 0.2s ease;
  }
  
  .country-item:hover {
    background-color: rgba(26, 115, 232, 0.1);
  }
  
  .disabled {
    opacity: 0.6;
    cursor: not-allowed;
    pointer-events: none;
  }
</style>
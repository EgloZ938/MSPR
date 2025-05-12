<template>
  <div class="country-selector-custom">
    <div class="selected-country" id="selectedCountryDisplay" @click="toggleDropdown"
      :class="{ 'disabled': isCoolingDown }">
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
          :class="{ 'selected': country.country_name === selectedCountry, 'disabled': isCoolingDown }"
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
  if (!event.target.closest('.country-selector-custom')) {
    isDropdownActive.value = false;
    searchTerm.value = '';
  }
}

function selectCountry(countryName) {
  if (isCoolingDown.value) return;

  activateCooldown(1000); // Plus long pour le changement de pays car cela recharge beaucoup de données

  selectedCountry.value = countryName;
  isDropdownActive.value = false;
  searchTerm.value = '';
  emit('input', countryName);
  emit('country-changed', countryName);
}

watch(() => props.value, (newValue) => {
  selectedCountry.value = newValue;
});
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}
</style>
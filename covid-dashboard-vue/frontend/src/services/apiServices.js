import axios from 'axios';

// Créer une instance axios avec des paramètres par défaut
const apiClient = axios.create({
    baseURL: '/api',
    timeout: 15000,
    headers: {
        'Content-Type': 'application/json'
    }
});

// Garder en cache les résultats des requêtes pour éviter des appels réseau inutiles
const cache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes en millisecondes

// Gestionnaire d'erreurs global
const handleError = (error) => {
    console.error('API Error:', error);

    if (error.response) {
        // Le serveur a répondu avec un code d'erreur
        console.error('Status:', error.response.status);
        console.error('Data:', error.response.data);
        return Promise.reject(error.response.data.error || 'Une erreur est survenue lors de la communication avec le serveur');
    } else if (error.request) {
        // La requête a été faite mais pas de réponse reçue
        console.error('Request:', error.request);
        return Promise.reject('Aucune réponse reçue du serveur. Veuillez vérifier votre connexion internet.');
    } else {
        // Erreur lors de la configuration de la requête
        return Promise.reject('Erreur lors de la configuration de la requête: ' + error.message);
    }
};

// Gestion des requêtes en cours pour éviter les requêtes en double
const pendingRequests = new Map();

// Fonction pour effectuer une requête avec cache
const cachedRequest = async (url, options = {}) => {
    const cacheKey = `${url}-${JSON.stringify(options)}`;

    // Si une requête identique est en cours, retourner la promesse existante
    if (pendingRequests.has(cacheKey)) {
        return pendingRequests.get(cacheKey);
    }

    // Vérifier si nous avons une version en cache valide
    const cachedResponse = cache.get(cacheKey);
    if (cachedResponse && cachedResponse.timestamp > Date.now() - CACHE_DURATION) {
        return Promise.resolve(cachedResponse.data);
    }

    // Créer une nouvelle promesse pour cette requête
    const request = apiClient(url, options)
        .then(response => {
            // Mettre en cache la réponse
            cache.set(cacheKey, {
                data: response.data,
                timestamp: Date.now()
            });

            // Supprimer cette requête de la liste des requêtes en cours
            pendingRequests.delete(cacheKey);

            return response.data;
        })
        .catch(error => {
            // Supprimer cette requête de la liste des requêtes en cours
            pendingRequests.delete(cacheKey);

            // Gérer l'erreur
            throw handleError(error);
        });

    // Ajouter cette requête à la liste des requêtes en cours
    pendingRequests.set(cacheKey, request);

    return request;
};

// Exporter les méthodes API optimisées
export default {
    // Statistiques globales
    getGlobalStats() {
        return cachedRequest('/global-stats');
    },

    getGlobalTimeline() {
        return cachedRequest('/global-timeline');
    },

    // Statistiques par pays
    getTopCountries() {
        return cachedRequest('/top-countries');
    },

    getAllCountries() {
        return cachedRequest('/countries');
    },

    getCountryTimeline(country) {
        return cachedRequest(`/country-timeline/${country}`);
    },

    getCountryDetails(country) {
        return cachedRequest(`/country-details/${country}`);
    },

    // Statistiques par région
    getRegionStats() {
        return cachedRequest('/region-stats');
    },

    // Comparaisons et filtrage
    compareCountries(countriesList) {
        return cachedRequest(`/country-comparison?countries=${countriesList.join(',')}`);
    },

    getFilteredData(filters = {}) {
        const params = new URLSearchParams();

        if (filters.region) params.append('region', filters.region);
        if (filters.dateStart) params.append('dateStart', filters.dateStart);
        if (filters.dateEnd) params.append('dateEnd', filters.dateEnd);
        if (filters.minCases) params.append('minCases', filters.minCases);
        if (filters.maxCases) params.append('maxCases', filters.maxCases);

        return cachedRequest(`/filtered-data?${params.toString()}`);
    },

    // Utilitaires
    clearCache() {
        cache.clear();
    },

    clearCacheFor(url) {
        for (const key of cache.keys()) {
            if (key.startsWith(url)) {
                cache.delete(key);
            }
        }
    }
};
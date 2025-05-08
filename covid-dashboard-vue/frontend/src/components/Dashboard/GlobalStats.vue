<template>
    <div class="stats-grid">
        <stats-card v-for="(stat, idx) in stats" :key="idx" :title="stat.title" :value="formatNumber(stat.value)"
            :id="stat.id" :dataset-name="stat.datasetName" @toggle="toggleDataset" />
    </div>
</template>

<script>
import StatsCard from './StatsCard.vue';
import axios from 'axios';

export default {
    name: 'GlobalStats',
    components: {
        StatsCard
    },
    data() {
        return {
            stats: [
                { title: 'Cas Confirmés', value: null, id: 'confirmed', datasetName: 'confirmed' },
                { title: 'Décès', value: null, id: 'deaths', datasetName: 'deaths' },
                { title: 'Guéris', value: null, id: 'recovered', datasetName: 'recovered' },
                { title: 'Cas Actifs', value: null, id: 'active', datasetName: 'active' }
            ]
        }
    },
    mounted() {
        this.loadGlobalStats();
    },
    methods: {
        async loadGlobalStats() {
            try {
                const response = await axios.get('/api/global-stats');
                const data = response.data;

                // Mettre à jour les valeurs
                this.stats.find(s => s.id === 'confirmed').value = data.total_confirmed;
                this.stats.find(s => s.id === 'deaths').value = data.total_deaths;
                this.stats.find(s => s.id === 'recovered').value = data.total_recovered;
                this.stats.find(s => s.id === 'active').value = data.total_active;
            } catch (error) {
                console.error('Erreur:', error);
                this.$emit('show-error', 'Erreur lors du chargement des statistiques globales');
            }
        },
        formatNumber(num) {
            if (num === null || num === undefined) return '0';
            return new Intl.NumberFormat().format(num);
        },
        toggleDataset(datasetName) {
            this.$emit('toggle-dataset', datasetName);
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
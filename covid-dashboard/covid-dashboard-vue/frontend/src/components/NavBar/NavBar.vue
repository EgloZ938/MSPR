<template>
    <nav class="navbar">
        <div class="nav-container">
            <h1>COVID-19 Dashboard</h1>
            <div class="nav-links">
                <a href="#" class="nav-link" :class="{ active: activeView === 'mondial' }"
                    @click.prevent="changeView('mondial')" data-tooltip="Statistiques mondiales">Vue Mondiale</a>
                <a href="#" class="nav-link" :class="{ active: activeView === 'regions' }"
                    @click.prevent="changeView('regions')" data-tooltip="Analyse par région">Par Région</a>
                <a href="#" class="nav-link" :class="{ active: activeView === 'pays' }"
                    @click.prevent="changeView('pays')" data-tooltip="Données par pays">Par Pays</a>
                <a href="#" class="nav-link" :class="{ active: activeView === 'correlation' }"
                    @click.prevent="changeView('correlation')" data-tooltip="Analyse des corrélations">Corrélations</a>
                <a href="#" class="nav-link" :class="{ active: activeView === 'tendances' }"
                    @click.prevent="changeView('tendances')" data-tooltip="Analyse des tendances">Tendances</a>
            </div>
        </div>
    </nav>
</template>

<script>
export default {
    name: 'NavBar',
    props: {
        activeView: {
            type: String,
            required: true
        }
    },
    methods: {
        changeView(view) {
            this.$emit('change-view', view);
        }
    }
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
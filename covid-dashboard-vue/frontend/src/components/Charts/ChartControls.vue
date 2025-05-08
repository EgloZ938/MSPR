<template>
    <div class="chart-controls">
        <button class="control-btn" @click="$emit('zoom-in')" data-tooltip="Zoom +">
            <i class="fas fa-search-plus"></i> +
        </button>
        <button class="control-btn" @click="$emit('zoom-out')" data-tooltip="Zoom -">
            <i class="fas fa-search-minus"></i> -
        </button>
        <button class="control-btn" @click="$emit('reset-zoom')" data-tooltip="Réinitialiser le zoom">Reset</button>
        <button class="control-btn" @click="$emit('download')"
            data-tooltip="Télécharger le graphique">Télécharger</button>
        <button class="control-btn" @click="$emit('export')" data-tooltip="Exporter les données en CSV">Exporter
            CSV</button>
    </div>
</template>

<script>
export default {
    name: 'ChartControls'
}
</script>

<style scoped>
/* Les styles seront lus du fichier assets/style.css global */
</style>
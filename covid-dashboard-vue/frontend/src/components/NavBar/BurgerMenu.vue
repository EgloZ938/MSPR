<template>
    <div class="burger-menu-container">
        <button class="burger-button" :class="{ 'active': isOpen }" @click="toggleMenu" aria-label="Menu">
            <span></span>
            <span></span>
            <span></span>
        </button>

        <div class="mobile-menu" :class="{ 'active': isOpen }">
            <div class="mobile-nav-links">
                <a v-for="item in menuItems" :key="item.view" href="#" class="mobile-nav-link"
                    :class="{ 'active': activeView === item.view }" @click.prevent="handleChangeView(item.view)">
                    {{ item.label }}
                </a>
            </div>
        </div>

        <!-- Overlay de fond sombre quand le menu est ouvert -->
        <div v-if="isOpen" class="menu-overlay" @click="closeMenu"></div>
    </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';

const props = defineProps({
    activeView: {
        type: String,
        required: true
    }
});

const emit = defineEmits(['change-view']);
const isOpen = ref(false);

const menuItems = [
    { view: 'mondial', label: 'Vue Mondiale' },
    { view: 'regions', label: 'Par Région' },
    { view: 'pays', label: 'Par Pays' },

    { view: 'predictive', label: '🤖 Prédictions IA' },

    { view: 'correlation', label: 'Corrélations' },
    { view: 'modele', label: 'Modèle' },
];

function toggleMenu() {
    isOpen.value = !isOpen.value;

    // Empêcher le défilement du body quand le menu est ouvert
    if (isOpen.value) {
        document.body.style.overflow = 'hidden';
    } else {
        document.body.style.overflow = '';
    }
}

function closeMenu() {
    isOpen.value = false;
    document.body.style.overflow = '';
}

function handleChangeView(view) {
    // Ne rien faire si la vue est déjà active
    if (view === props.activeView) {
        closeMenu();
        return;
    }

    // Changer la vue et fermer le menu
    emit('change-view', view);
    closeMenu();
}
</script>

<style scoped>
/* Styles pour le bouton burger */
.burger-menu-container {
    display: none;
    /* Caché par défaut, affiché uniquement sur mobile */
}

.burger-button {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 30px;
    height: 21px;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0;
    z-index: 1001;
}

.burger-button span {
    display: block;
    width: 100%;
    height: 3px;
    background-color: white;
    border-radius: 3px;
    transition: all 0.3s ease;
}

/* Animation du bouton burger quand actif */
.burger-button.active span:nth-child(1) {
    transform: translateY(9px) rotate(45deg);
}

.burger-button.active span:nth-child(2) {
    opacity: 0;
}

.burger-button.active span:nth-child(3) {
    transform: translateY(-9px) rotate(-45deg);
}

/* Menu mobile */
.mobile-menu {
    position: fixed;
    top: 0;
    right: -100%;
    width: 75%;
    max-width: 300px;
    height: 100vh;
    background-color: var(--card-bg);
    z-index: 1000;
    transition: right 0.3s ease;
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
    padding-top: 70px;
    overflow-y: auto;
}

.mobile-menu.active {
    right: 0;
}

.mobile-nav-links {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 20px;
}

.mobile-nav-link {
    display: block;
    padding: 15px;
    text-decoration: none;
    color: var(--text-primary);
    border-radius: 8px;
    transition: all 0.2s ease;
}

.mobile-nav-link:hover {
    background-color: rgba(26, 115, 232, 0.1);
}

.mobile-nav-link.active {
    background-color: var(--primary-color);
    color: white;
}

/* Overlay de fond sombre */
.menu-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 999;
}

/* Media queries */
@media (max-width: 768px) {
    .burger-menu-container {
        display: block;
    }
}
</style>
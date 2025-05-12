/**
 * Utilitaire simple pour ajouter un cooldown aux fonctions
 * @param {Function} fn - La fonction à exécuter
 * @param {number} cooldownTime - Le temps de cooldown en millisecondes
 * @returns {Function} - La fonction avec cooldown
 */
export function withCooldown(fn, cooldownTime = 500) {
    let lastCall = 0;
    let isCoolingDown = false;

    return function (...args) {
        const now = Date.now();

        // Si nous sommes en cooldown, ne rien faire
        if (isCoolingDown) {
            return;
        }

        // Exécuter la fonction
        isCoolingDown = true;

        // Afficher un indicateur de chargement si nécessaire
        document.body.style.cursor = 'wait';

        // Exécuter la fonction après un très court délai
        setTimeout(() => {
            fn.apply(this, args);

            // Réinitialiser le cooldown après le temps spécifié
            setTimeout(() => {
                isCoolingDown = false;
                document.body.style.cursor = 'default';
            }, cooldownTime);
        }, 10);
    };
}
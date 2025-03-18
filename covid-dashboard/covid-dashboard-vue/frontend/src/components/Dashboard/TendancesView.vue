<template>
    <div id="tendances" class="visualization-section">
      <div class="coming-soon">
        <h2>Analyse des tendances</h2>
        <p>Cette fonctionnalité sera disponible prochainement.</p>
      </div>
    </div>
  </template>
  
  <script setup>
  // Aucun script nécessaire pour ce composant
  </script>
  
  <style scoped>
  .coming-soon {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    background-color: var(--card-bg);
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    text-align: center;
  }
  
  .coming-soon h2 {
    color: var(--primary-color);
    margin-bottom: 20px;
  }
  
  .coming-soon p {
    color: var(--text-secondary);
  }
  </style>
  
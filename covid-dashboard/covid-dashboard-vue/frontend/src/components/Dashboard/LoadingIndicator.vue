<template>
    <div class="loading" v-if="isLoading">
      Chargement des données...
    </div>
  </template>
  
  <script setup>
  import { defineProps } from 'vue';
  
  defineProps({
    isLoading: {
      type: Boolean,
      default: false
    }
  });
  </script>
  
  <style scoped>
  .loading {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: rgba(255, 255, 255, 0.8);
    padding: 15px 20px;
    border-radius: 5px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    font-weight: 500;
    color: var(--primary-color);
  }
  </style>
  
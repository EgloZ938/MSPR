<template>
    <div v-if="show" class="mini-loader-overlay">
      <div class="mini-loader"></div>
    </div>
  </template>
  
  <script setup>
  import { defineProps } from 'vue';
  
  defineProps({
    show: {
      type: Boolean,
      default: false
    }
  });
  </script>
  
  <style scoped>
  .mini-loader-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(255, 255, 255, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 999;
    pointer-events: none;
  }
  
  .mini-loader {
    width: 30px;
    height: 30px;
    border: 3px solid rgba(26, 115, 232, 0.2);
    border-top: 3px solid #1a73e8;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  </style>
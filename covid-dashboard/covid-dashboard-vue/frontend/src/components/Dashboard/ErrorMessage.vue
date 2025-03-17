<template>
    <div class="error" v-if="message" :style="errorStyle">
        {{ message }}
    </div>
</template>

<script>
export default {
    name: 'ErrorMessage',
    props: {
        message: {
            type: String,
            default: ''
        }
    },
    data() {
        return {
            errorStyle: `
          background-color: #f8d7da;
          color: #721c24;
          padding: 10px;
          margin: 10px 0;
          border-radius: 4px;
          text-align: center;
          position: fixed;
          top: 20px;
          left: 50%;
          transform: translateX(-50%);
          z-index: 1000;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        `
        }
    }
}
</script>
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'myth': {
          'cosmic': '#FF4500',    // Hot red-orange
          'mythic': '#FF6347',    // Tomato red
          'archetypal': '#FF8C00', // Dark orange
          'symbolic': '#FFA500',   // Orange
          'surface': '#FFD700',    // Gold
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'fire': 'fire 1.5s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          'from': { boxShadow: '0 0 10px rgba(255, 69, 0, 0.5)' },
          'to': { boxShadow: '0 0 20px rgba(255, 69, 0, 0.8), 0 0 30px rgba(255, 99, 71, 0.6)' }
        },
        fire: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(255, 69, 0, 0.8), 0 0 40px rgba(255, 69, 0, 0.5)' },
          '50%': { boxShadow: '0 0 30px rgba(255, 99, 71, 1), 0 0 60px rgba(255, 140, 0, 0.7)' }
        }
      }
    },
  },
  plugins: [],
}

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'holo': {
          'bg-primary': '#0d1117',
          'bg-secondary': '#161b22',
          'bg-tertiary': '#21262d',
          'text-primary': '#c9d1d9',
          'text-secondary': '#8b949e',
          'border': '#30363d',
          'success': '#238636',
          'warning': '#f0883e',
          'error': '#da3633',
          'info': '#58a6ff',
          'purple': '#8957e5'
        }
      }
    },
  },
  plugins: [],
}

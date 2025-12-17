/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],

  darkMode: 'class',

  theme: {
    extend: {
      colors: {
        // HoloLoom Brand Colors
        holo: {
          primary: '#6366f1',         // Indigo
          'primary-light': '#818cf8', // Indigo light (for hover)
          'primary-dark': '#4f46e5',  // Indigo dark
          secondary: '#a855f7',       // Purple
          'secondary-light': '#c47cf8', // Purple light
          accent: '#06b6d4',          // Cyan
          'accent-light': '#22d3ee',  // Cyan light
          success: '#10b981',         // Emerald
          warning: '#f59e0b',         // Amber
          danger: '#ef4444',          // Red
          neutral: '#6b7280',         // Gray
        },

        // Agent State Colors (Dark Mode Optimized)
        agent: {
          idle: '#6b7280',       // Gray (inactive)
          ready: '#10b981',      // Emerald (ready to run)
          running: '#06b6d4',    // Cyan (actively processing)
          success: '#34d399',    // Light Emerald (completed successfully)
          warning: '#f59e0b',    // Amber (warning/slow)
          error: '#ef4444',      // Red (failed)
          paused: '#f97316',     // Orange (paused/suspended)
        },

        // Background and Surface Colors
        surface: {
          primary: '#0f172a',    // Dark slate (main background)
          secondary: '#1e293b',  // Slightly lighter slate
          tertiary: '#334155',   // Even lighter slate
          elevated: '#475569',   // For elevated surfaces
        },

        // Text Colors
        text: {
          primary: '#f1f5f9',    // Almost white
          secondary: '#cbd5e1',  // Light gray
          tertiary: '#94a3b8',   // Medium gray
          muted: '#64748b',      // Muted gray
        },
      },

      backgroundColor: {
        dark: '#0f172a',
        'dark-secondary': '#1e293b',
      },

      textColor: {
        light: '#f1f5f9',
        'light-secondary': '#cbd5e1',
      },

      borderColor: {
        'dark-light': '#334155',
        'dark-lighter': '#475569',
      },

      animation: {
        'pulse-subtle': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
        'bounce-subtle': 'bounce 1s ease-in-out infinite',
        'fade-in': 'fadeIn 0.3s ease-in',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'slide-in-left': 'slideInLeft 0.3s ease-out',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideInRight: {
          '0%': {
            opacity: '0',
            transform: 'translateX(10px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateX(0)',
          },
        },
        slideInLeft: {
          '0%': {
            opacity: '0',
            transform: 'translateX(-10px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateX(0)',
          },
        },
      },

      boxShadow: {
        'glow-primary': '0 0 20px rgba(99, 102, 241, 0.5)',
        'glow-secondary': '0 0 20px rgba(168, 85, 247, 0.5)',
        'glow-accent': '0 0 20px rgba(6, 182, 212, 0.5)',
        'glow-success': '0 0 20px rgba(16, 185, 129, 0.5)',
        'glow-error': '0 0 20px rgba(239, 68, 68, 0.5)',
        'dark-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.5)',
        'dark-md': '0 4px 6px -1px rgba(0, 0, 0, 0.5)',
        'dark-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
      },

      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
      },

      spacing: {
        '128': '32rem',
        '144': '36rem',
      },

      zIndex: {
        'dropdown': '1000',
        'sticky': '1020',
        'fixed': '1030',
        'modal-backdrop': '1040',
        'modal': '1050',
        'popover': '1060',
        'tooltip': '1070',
      },
    },
  },

  plugins: [
    // Custom plugin for agent state badges
    function ({ addComponents, theme }) {
      const agentStatuses = {
        '.badge-agent-idle': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-900 text-gray-300 border border-gray-700': {},
        },
        '.badge-agent-ready': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-200 border border-emerald-700': {},
        },
        '.badge-agent-running': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-cyan-950 text-cyan-200 border border-cyan-700 animate-pulse-subtle': {},
        },
        '.badge-agent-success': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-900 text-emerald-100 border border-emerald-600': {},
        },
        '.badge-agent-warning': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-950 text-amber-200 border border-amber-700': {},
        },
        '.badge-agent-error': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-950 text-red-200 border border-red-700': {},
        },
        '.badge-agent-paused': {
          '@apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-950 text-orange-200 border border-orange-700': {},
        },
      }

      const cardComponent = {
        '.card': {
          '@apply bg-surface-secondary border border-dark-light rounded-lg shadow-dark-md': {},
        },
        '.card-elevated': {
          '@apply bg-surface-secondary border border-dark-lighter rounded-lg shadow-dark-lg': {},
        },
        '.card-interactive': {
          '@apply bg-surface-secondary border border-dark-light rounded-lg shadow-dark-md transition-all duration-200 hover:border-dark-lighter hover:shadow-dark-lg hover:bg-surface-tertiary cursor-pointer': {},
        },
      }

      addComponents({
        ...agentStatuses,
        ...cardComponent,
      })
    },
  ],
}

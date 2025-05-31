/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{rs,html,js}",
    "./dist/**/*.html"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Inter', 'Fira Sans', 'Ubuntu', 'Roboto', 'sans-serif'],
      },
      colors: {
        // Light mode colors
        primary: {
          DEFAULT: '#b0b91a',
          hover: '#bcc735',
        },
        accent: '#74a309',
        clear: '#991b1b',
        text: {
          DEFAULT: '#060504',
          muted: '#635c50',
        },
        bg: {
          DEFAULT: '#eee7d8',
          card: '#e0d6c8',
          'card-glass': 'rgba(255, 255, 255, 0.03)',
        },
        // Dark mode colors (will be used with dark: prefix)
        dark: {
          primary: {
            DEFAULT: '#4f46e5',
            hover: '#4338ca',
          },
          accent: '#8b5cf6',
          clear: '#ef4444',
          text: {
            DEFAULT: '#f9fafb',
            muted: '#9ca3af',
          },
          bg: {
            DEFAULT: '#111827',
            card: '#1f2937',
          },
        },
        // Status colors
        danger: {
          bg: 'rgba(239, 68, 68, 0.1)',
          border: 'rgba(239, 68, 68, 0.2)',
          text: '#d12525',
          'text-dark': '#fca5a5',
        },
        success: {
          bg: 'rgba(17, 177, 57, 0.1)',
          border: 'rgba(16, 185, 129, 0.2)',
          text: '#1fc080',
          'text-dark': '#6ee7b7',
        },
        error: {
          bg: 'rgba(239, 68, 68, 0.1)',
          border: 'rgba(239, 68, 68, 0.2)',
          text: '#d12525',
          'text-dark': 'rgba(239, 68, 68, 0.2)',
        },
        meter: {
          bg: 'rgba(255, 255, 255, 0.1)',
        },
        border: {
          DEFAULT: '#374151',
        },
      },
      backgroundImage: {
        'primary-gradient': 'linear-gradient(45deg, var(--tw-gradient-from), var(--tw-gradient-to))',
        'meter-gradient': 'linear-gradient(90deg, var(--tw-gradient-from), var(--tw-gradient-to))',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'expand-circle': 'expand-circle 0.5s ease-out forwards',
        'contract-circle': 'contract-circle 0.5s ease-in forwards',
      },
      keyframes: {
        fadeIn: {
          from: {
            opacity: '0',
            transform: 'translateY(20px)',
          },
          to: {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        'expand-circle': {
          from: {
            transform: 'translate(-50%, -50%) scale(0)',
          },
          to: {
            transform: 'translate(-50%, -50%) scale(1.5)',
          },
        },
        'contract-circle': {
          from: {
            transform: 'translate(-50%, -50%) scale(1.5)',
          },
          to: {
            transform: 'translate(-50%, -50%) scale(0)',
          },
        },
      },
      transitionProperty: {
        'wallpaper': 'background-image',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}

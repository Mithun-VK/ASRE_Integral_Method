/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ark: {
          red: '#DC2626',
          'red-light': '#FCA5A5',
          'red-dark': '#991B1B',
          'red-glow': '#EF4444',
          black: '#1F2937',
          'black-900': '#111827',
          'bg-primary': '#0F0F0F',
          'bg-secondary': '#1F1F1F',
          'bg-card': '#171717',
          'bg-card-hover': '#1E1E1E',
          gold: '#F59E0B',
          'gold-light': '#FCD34D',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'float': 'float 3s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'pulse-glow-green': 'pulse-glow-green 2s ease-in-out infinite',
        'pulse-glow-amber': 'pulse-glow-amber 2s ease-in-out infinite',
        'count-up': 'count-up 1.5s ease-out forwards',
        'slide-up': 'slide-up 0.6s ease-out forwards',
        'slide-in-right': 'slide-in-right 0.6s ease-out forwards',
        'fade-in': 'fade-in 0.8s ease-out forwards',
        'scale-in': 'scale-in 0.5s ease-out forwards',
        'shimmer': 'shimmer 2s linear infinite',
        'border-glow': 'border-glow 3s ease-in-out infinite',
        'gradient-shift': 'gradient-shift 8s ease infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 8px rgba(220, 38, 38, 0.3)' },
          '50%': { boxShadow: '0 0 24px rgba(220, 38, 38, 0.6)' },
        },
        'pulse-glow-green': {
          '0%, 100%': { boxShadow: '0 0 8px rgba(34, 197, 94, 0.3)' },
          '50%': { boxShadow: '0 0 24px rgba(34, 197, 94, 0.6)' },
        },
        'pulse-glow-amber': {
          '0%, 100%': { boxShadow: '0 0 8px rgba(245, 158, 11, 0.3)' },
          '50%': { boxShadow: '0 0 24px rgba(245, 158, 11, 0.6)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(30px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'border-glow': {
          '0%, 100%': { borderColor: 'rgba(220, 38, 38, 0.3)' },
          '50%': { borderColor: 'rgba(220, 38, 38, 0.8)' },
        },
        'gradient-shift': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-gradient': 'linear-gradient(135deg, #0F0F0F 0%, #1F1F1F 50%, #0F0F0F 100%)',
      },
    },
  },
  plugins: [
    // Hide scrollbars while keeping scroll functionality (for snap carousels)
    function({ addUtilities }) {
      addUtilities({
        '.scrollbar-hide': {
          '-ms-overflow-style': 'none',
          'scrollbar-width': 'none',
          '&::-webkit-scrollbar': { display: 'none' },
        },
      });
    },
  ],
}

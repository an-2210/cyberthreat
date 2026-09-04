/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        soc: {
          bg: '#0b0f19',
          card: '#111827',
          header: '#131b2e',
          border: '#1f293d',
          hover: '#1e293b',
          muted: '#64748b',
          text: '#f8fafc',
          subtext: '#94a3b8',
          accent: '#06b6d4',      // Cyan accent
          accentGlow: 'rgba(6, 182, 212, 0.15)',
          success: '#10b981',     // Emerald green
          warning: '#f59e0b',     // Amber
          danger: '#f43f5e',      // Rose red
          info: '#3b82f6',        // Blue
          purple: '#a855f7',      // Future modules
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'soc-card': '0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(31, 41, 61, 0.6)',
        'soc-glow-cyan': '0 0 15px -3px rgba(6, 182, 212, 0.3)',
        'soc-glow-emerald': '0 0 15px -3px rgba(16, 185, 129, 0.3)',
        'soc-glow-rose': '0 0 15px -3px rgba(244, 63, 94, 0.3)',
      }
    },
  },
  plugins: [],
}

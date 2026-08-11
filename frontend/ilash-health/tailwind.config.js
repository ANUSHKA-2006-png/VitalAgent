/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Segoe UI", "-apple-system", "sans-serif"],
      },
      colors: {
        wash: "#F7F7FB",
        border: "#E5E5EA",
        accent: {
          DEFAULT: "#6C5CE7",
          light: "#8B7CF6",
          soft: "#EEEDFE",
        },
        info: "#378ADD",
        success: { bg: "#E1F5EE", text: "#0F6E56", dot: "#1D9E75" },
        attention: { bg: "#E6F1FB", text: "#0C447C", dot: "#378ADD" },
        danger: { bg: "#FCEBEB", text: "#A32D2D", dot: "#E24B4A" },
        ink: { DEFAULT: "#1C1C1E", muted: "#6B6B75" },
      },
      borderRadius: {
        card: "12px",
        control: "8px",
      },
    },
  },
  plugins: [],
};

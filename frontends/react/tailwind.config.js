/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        // Pop + glow effect for the Analyze/Route/Generate stage badges.
        // Used by TypingIndicator.tsx via the `animate-stage-complete` class.
        //
        // CSS keyframes work like a timeline - the percentages are points in time
        // during the animation. For a 0.5s animation:
        //   0%  = start (0ms)      - normal size, no glow
        //   30% = peak  (150ms)    - badge pops up to 1.25x size with green glow
        //   60% = bounce (300ms)   - slight bounce back below normal (0.95x)
        //   100% = settle (500ms)  - back to normal size, glow fades out
        //
        // The green glow uses rgba(34, 197, 94) which is Tailwind's green-500.
        // boxShadow "0 0 12px 4px" creates a soft halo around the badge.
        "stage-complete": {
          "0%": {
            transform: "scale(1)",
            boxShadow: "0 0 0 0 rgba(34, 197, 94, 0)",
          },
          "30%": {
            transform: "scale(1.25)",
            boxShadow: "0 0 12px 4px rgba(34, 197, 94, 0.4)",
          },
          "60%": {
            transform: "scale(0.95)",
            boxShadow: "0 0 6px 2px rgba(34, 197, 94, 0.2)",
          },
          "100%": {
            transform: "scale(1)",
            boxShadow: "0 0 0 0 rgba(34, 197, 94, 0)",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        // 0.5s total duration, ease-out means it starts fast and slows down
        "stage-complete": "stage-complete 0.5s ease-out",
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
}

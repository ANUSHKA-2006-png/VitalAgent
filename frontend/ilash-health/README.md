# ILASH Health — Frontend (React + TypeScript + Tailwind)

A full-screen, multi-page AI-powered community health screening dashboard,
matching the ILASH Health design system (purple/blue accents, flat cards,
12px radii). Includes SpO2 (oxygen saturation) as a full screening metric
throughout: upload, analysis, results, patient profile, alerts, and
community analytics.

## Pages included

- `/` — Dashboard
- `/screening/new/details` — New Screening: Patient Details (step 1)
- `/screening/new/upload` — New Screening: Data Upload (step 2) — now includes
  a dedicated **SpO2 Screening** upload card alongside Heart Rate, Stress,
  and Fall
- `/screening/new/analysis` — New Screening: Analysis (step 3)
- `/screening/new/results` — New Screening: Results (step 4) — now includes
  an **SpO2 (Oxygen Saturation)** result card with trend + status
- `/patients` — Patients list (SpO2 column added)
- `/patients/:id` — Patient Profile (SpO2 metric card + latest-screening row)
- `/alerts` — Alerts (includes a "Low Oxygen Saturation" alert)
- `/reports` — Reports
- `/analytics` — Community Analytics (SpO2 added to Top Risk Indicators, plus
  a new SpO2 Distribution donut chart)
- `/settings` — Settings placeholder

Every page fills the full browser viewport (sidebar + header + scrollable
content area) rather than floating as a centered card.

## Getting started

Requires Node.js 18+.

```bash
npm install
npm run dev
```

Then open the printed local URL (default `http://localhost:5173`).

To create a production build:

```bash
npm run build
npm run preview
```

## Project structure

```
src/
  components/     Shared UI: Sidebar, Header, Stepper, StatusBadge,
                  MetricCard, Sparkline, DonutChart, Layout
  pages/          One file per screen/route
  data/           Mock data (patients, alerts, risk indicators, trends)
  types.ts        Shared TypeScript types (Patient, AlertItem, etc.)
  App.tsx         Route definitions
  main.tsx        App entry point
index.html        Vite HTML entry
tailwind.config.js  Design tokens (colors, radii, font)
```

## Customizing data

All sample data lives in `src/data/mockData.ts` — swap this out for a real
API call (e.g. `fetch`/`axios` inside a `useEffect`, or React Query) when
wiring up a backend. The `Patient` type in `src/types.ts` already includes
an `spo2: number` field ready for real sensor data.

# ChannelHub Frontend

React + TypeScript + Ant Design + React Query + Plotly web UI for the ChannelHub
信道数据工场 platform. Talks to the FastAPI backend at `http://localhost:8000` via the
REST API described in Phase 6.3.

## Prerequisites

- Node.js 20+ and npm 10+
- A running backend at `http://localhost:8000` (see `platform/backend`)
- Optional: Redis + worker for jobs that actually execute (see `platform/worker`)

## Install

```bash
cd platform/frontend
npm install
cp .env.example .env      # override VITE_API_BASE_URL if the backend is elsewhere
```

## Commands

| Command            | Description                                         |
| ------------------ | --------------------------------------------------- |
| `npm run dev`      | Start Vite dev server on http://localhost:5175      |
| `npm run build`    | Typecheck + production build to `dist/`             |
| `npm run preview`  | Preview production build locally                    |
| `npm run test`     | Run Vitest suite                                    |
| `npm run lint`     | ESLint over `.ts`/`.tsx` files                      |
| `npm run typecheck`| `tsc --noEmit` without emit                         |

During `npm run dev`, requests to `/api/*` are proxied to
`http://localhost:8000` so the UI can use relative URLs with no CORS setup.

## Project structure

```
src/
  api/              # axios client, typed endpoint wrappers, React Query hooks
  components/
    Layout/         # AppSider, AppHeader
    ConfigForm/     # rjsf-powered Hydra config form
    JobProgress/    # progress card, log viewer
    Metrics/        # metrics tables
    Plots/          # Plotly loss chart, UMAP, SINR histogram
    Common/         # loading, empty state, error boundary
  pages/
    Dashboard.tsx
    Datasets.tsx / DatasetDetail.tsx
    Jobs.tsx / JobDetail.tsx / JobCreate.tsx
    Runs.tsx / RunDetail.tsx
    Compare.tsx
    Models.tsx
  hooks/            # useJobPolling, useConfigSchema
  utils/            # date/number formatters, job status -> color
  theme.ts          # AntD light/dark tokens
  store.ts          # Zustand UI store (theme, sider)
  main.tsx          # entry
  App.tsx           # router + layout shell
tests/              # vitest + RTL unit tests
```

## Polling model

`useJob(id)` and `useJobLogs(id)` poll every 2 seconds while the job is in
`queued` or `running`; React Query stops the interval automatically when the
job reaches a terminal state. Listing pages (`/jobs`) poll every 5 seconds.

## Theming

The header exposes a dark-mode toggle. Preference is persisted in
`localStorage` under `msg-ui`. AntD's `ConfigProvider` theme algorithm is
swapped at runtime via the `useUIStore` Zustand store.

## Deployment

Phase 6.3 ships dev-mode only. In Phase 6.5 the deploy agent wraps this UI in
an nginx container that:

1. Serves the static `dist/` output at `/`.
2. Reverse-proxies `/api/*` to the FastAPI service.
3. Co-locates the backend, worker, and Redis in a docker-compose stack.

Until then run the backend, worker, and Redis locally and start Vite with
`npm run dev`.

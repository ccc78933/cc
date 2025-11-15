Overview

- Stack: Flask REST API + WSGI/ASGI (gunicorn/uvicorn) + Inference (PyTorch/ONNX Runtime) + DB (SQLite/MySQL/PostgreSQL) + optional Redis (cache) + optional Celery/RQ (queue).
- Frontend: Vue 3 + Vite (this repo: `dish-nutrition-vue/`).
- Backend: Place server code in a separate folder (recommended: `server/`). Existing PoC is under `menu-nutrition/backend/` and can be migrated.

API Contract (v1)

- `GET /api/ping` → `{ ok: true, version: "1.0.0" }`
- `POST /api/predict` multipart form: `file`; optional: `mode=classifier|detector|auto`, `topk=5`, `return_vis=true` → result with `dish/score/topk/nutrition`, optional `det.items` with `bbox`.
- `GET /api/nutrition?dish=...&basis=...&serving_size_g=...`
- `GET /api/history?page=1&page_size=20`
- `DELETE /api/history/{id}`; `DELETE /api/history`
- `POST /api/favorite`; `GET /api/favorite`
- `POST /api/ai/advice` with `{ messages, context }`

Frontend Mapping

- App root: `dish-nutrition-vue/`
  - Services: `src/services/api.js` implements `/api/*` (predict/nutrition/history/favorite/ai) with cancellable uploads and progress.
  - Core flow: `src/pages/Home.vue` (upload/drag-drop/camera, progress+cancel, Top‑K, detection overlay, nutrition scaling, CSV export, send-to-chat).
  - Visuals: `DetectionOverlay.vue`, `MacroChart.vue` (doughnut), `TopKChart.vue` (horizontal bar), `NutritionTable.vue`.
  - State: `src/stores/app.js` (uid, settings, history, favorites, chat context).
  - Dev proxy: `vite.config.js` proxies `/api` to backend.

Backend Skeleton (suggested `server/`)

- `server/app.py` (create app, CORS, blueprints, logging)
- `server/config/` (env, .env loader: API_BASE, DB_URL, MODEL_PATH, MODEL_TYPE, DEVICE, etc.)
- `server/services/` (`infer_pipeline.py`, `detector.py`, `classifier.py`, `nutrition.py`, `ai_advice.py`, `history.py`)
- `server/models/` (`schemas.py` for request/response, `db.py` for ORM models: History, Favorite, FoodNutrition, etc.)
- `server/utils/` (image/exif, unit convert, limiter, cache, errors)
- `server/assets/` (label_map.json, nutrition seeds), `server/weights/`, `server/logs/`, `server/tests/`

Config & Ops

- Python ≥ 3.10; key env: `MODEL_TYPE`, `MODEL_PATH`, `DEVICE`, `DB_URL`, `CORS_ORIGINS`, `MAX_UPLOAD_SIZE_MB`, `TOPK`, `CONF_THRES`, `IOU_THRES`, `NUTRITION_SOURCE`, `LANG`.
- Deploy: gunicorn (WSGI) or uvicorn (ASGI). Frontend built and served statically (CDN/Nginx). Reverse proxy terminates TLS and proxies `/api/*`.

Compatibility Notes

- Frontend is backward-compatible with legacy endpoints (`/health`, `/predict_image`, `/predict_menu`) but prefers `/api/*` if available.
- Nutrition basis: frontend accepts `nutrition.basis=per_serving|per_100g` and `serving_size_g`; scales via `ServingControl`.


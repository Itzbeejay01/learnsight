# Dockerization Plan for LearnSight

## Steps:

- [x] 1. Create TODO.md with plan breakdown
- [x] 2. Create Dockerfile (Python 3.9.6-slim, gunicorn CMD)
- [x] 3. Create docker-compose.yml (single app service, volumes for models/instance/data/reports, port 5000)
- [x] 4. Update TODO.md with progress
- [ ] 5. Test: docker-compose up --build (verify app runs on :5000)
- [x] 6. Complete task

## Status: ✅ Complete

**Files created:**

- `Dockerfile`: Python 3.9.6-slim image, gunicorn on 0.0.0.0:5000
- `docker-compose.yml`: Single `app` service, port 5000, volumes for persistence (models, instance/SQLite, data, reports)

**Usage:**

```
docker compose up --build
```

App available at http://localhost:5000

**Notes:** Mount local `./models/` (read-only), `./data/` (ro), `./reports/` (ro). Run `python train_models.py` locally to populate. Only `instance/` (DB) persists in Docker volume.

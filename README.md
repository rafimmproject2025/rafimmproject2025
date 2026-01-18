# Fresh News Agent (MCP + FastAPI + Streamlit)

## Quick start
1) Copy your Excel files into `./data/`:
   - `Bangladesh_Election_Media_Monitoring_Codebook.xlsx`
   - `Prothom Alo 2024 Election News.xlsx`

2) Start Postgres (schema auto-loads):
```
docker compose up -d db
```

3) Load codebook & news (one-shot):
```
docker compose run --rm loader
```

4) Bring up the stack:
```
docker compose up -d mcp backend enricher streamlit
```

5) Open:
- Backend: http://localhost:8000/api/health
- Streamlit: http://localhost:8501
- Sign in: default admin creds are `admin` / `admin123` (override with env vars `DEFAULT_ADMIN_USER` / `DEFAULT_ADMIN_PASSWORD`; set `JWT_SECRET` too).
- Internal workers use a shared bearer `SERVICE_TOKEN` (set the same value for backend/enricher; default `service-token`).

### Notes
- MCP is internal-only (no host port). Services use `http://mcp:5000/mcp` on the compose network.
- Healthchecks are Python exec-form for Windows compatibility.
- Base image pinned: `python:3.11-slim-bookworm`.

# ── Base Image ────────────────────────────────────────────
FROM python:3.12-slim

# ── Working Directory ─────────────────────────────────────
WORKDIR /app

# ── Copy Requirements & Install ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Files ────────────────────────────────────
COPY api/      ./api/
COPY ml/       ./ml/
COPY model/    ./model/

# ── Expose Port ───────────────────────────────────────────
EXPOSE 8000

# ── Run FastAPI ───────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"]
# ── Dockerfile for Email Triage OpenEnv ──────────────────────────────────
# Deploys a FastAPI server on Hugging Face Spaces (Docker SDK).
# Port 7860 is the default exposed port on HF Spaces.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────
COPY models.py  .
COPY data.py    .
COPY grader.py  .
COPY env.py     .
COPY app.py     .
# inference.py and openenv.yaml are included for reference but NOT run here
COPY inference.py  .
COPY openenv.yaml  .

# ── Expose HF Spaces port ────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start the API server (NOT inference) ─────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

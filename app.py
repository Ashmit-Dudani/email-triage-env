"""
app.py - FastAPI server for the Email Triage OpenEnv Environment

Endpoints
---------
  POST /reset          → start a new episode
  POST /step           → submit an action
  GET  /state          → inspect current episode state
  GET  /health         → liveness probe (used by Docker / HF Spaces)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from models import (
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    EpisodeState,
)
from env import EmailTriageEnv

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world simulation environment for AI email-triage agents.",
    version="1.0.0",
)

# Allow all origins so the HF Spaces preview and external agents can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global environment instance ────────────────────────────────────────────
# NOTE: This is a single shared instance — fine for a hackathon / demo.
# In production you'd store one env per session / user.
env = EmailTriageEnv()

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Liveness probe — always returns 200 OK."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(body: ResetRequest):
    """
    Start a new episode.

    Body
    ----
    {
        "task": "easy" | "medium" | "hard"   (default: "easy")
    }

    Returns the first Observation and the episode_id you must
    pass to every /step call.
    """
    try:
        observation = env.reset(task=body.task)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ResetResponse(
        observation=observation,
        episode_id=observation.episode_id,
    )


@app.post("/step", response_model=StepResponse)
def step(body: StepRequest):
    """
    Submit the agent's action for the current email.

    Body
    ----
    {
        "episode_id": "<uuid from /reset>",
        "action": {
            "category":    "work" | "spam" | "personal" | "promo",
            "priority":    "high" | "medium" | "low",
            "action_type": "reply" | "ignore" | "escalate"
        }
    }

    Returns the next Observation (null when done), the Reward,
    and a done flag.
    """
    # Validate episode_id matches the active episode
    try:
        current_state = env.state()
    except RuntimeError:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )

    if current_state.episode_id != body.episode_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"episode_id mismatch: "
                f"expected '{current_state.episode_id}', got '{body.episode_id}'"
            ),
        )

    try:
        result = env.step(body.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state", response_model=EpisodeState)
def state():
    """
    Inspect the current episode state without consuming a step.
    Useful for debugging and monitoring.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
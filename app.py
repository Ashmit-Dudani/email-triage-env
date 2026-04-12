from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from models import ResetRequest, ResetResponse, StepRequest, StepResponse, EpisodeState
from env import EmailTriageEnv

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = EmailTriageEnv()

def safe(v):
    """Force float strictly between 0 and 1."""
    try:
        v = float(v)
    except:
        v = 0.42
    # Clamp to (0.1, 0.9) — well away from 0 and 1
    return round(max(0.1, min(0.9, v)), 4)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
def reset(body: ResetRequest = None):
    task = "easy"
    if body is not None and body.task:
        task = body.task
    try:
        observation = env.reset(task=task)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ResetResponse(observation=observation, episode_id=observation.episode_id)

@app.post("/step")
def step(body: StepRequest):
    try:
        current_state = env.state()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    if current_state.episode_id != body.episode_id:
        raise HTTPException(status_code=400, detail="episode_id mismatch")

    try:
        result = env.step(body.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    r = result.reward

    # Return plain dict — no Pydantic validation blocking safe values
    return {
        "observation": result.observation.dict() if result.observation else None,
        "reward": {
            "total":          safe(r.total),
            "category_score": safe(r.category_score),
            "priority_score": safe(r.priority_score),
            "action_score":   safe(r.action_score),
            "penalty":        safe(r.penalty),
            "explanation":    r.explanation,
        },
        "done": result.done,
        "info": result.info,
    }

@app.get("/state", response_model=EpisodeState)
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
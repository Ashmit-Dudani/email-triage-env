# 📧 Email Triage OpenEnv

A real-world simulation environment where an AI agent manages an inbox by
classifying emails, setting priorities, and choosing actions — with fully
deterministic grading.

---

## 📁 File Structure

```
email_triage_env/
├── models.py        # Pydantic schemas (Observation, Action, Reward, …)
├── data.py          # 10 emails + ground-truth labels
├── grader.py        # Deterministic scorer with partial credit & penalties
├── env.py           # Environment: reset() / step() / state()
├── app.py           # FastAPI server — /reset, /step, /state, /health
├── inference.py     # AI agent runner (strict [START]/[STEP]/[END] logs)
├── openenv.yaml     # OpenEnv spec
├── Dockerfile       # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API server
```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 3. Run the AI agent
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"          # any OpenAI-compatible model
export API_BASE_URL="http://localhost:7860"

python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

Expected log format:
```
[START] task: easy
[STEP] action: {"category": "work", "priority": "high", "action_type": "reply"} reward: 1.0
[STEP] action: {"category": "spam", "priority": "low", "action_type": "ignore"} reward: 1.0
...
[END] score: 0.9143
```

---

## 🌐 API Reference

### `POST /reset`
Start a new episode.
```json
{ "task": "easy" }
```
Returns:
```json
{
  "observation": { "email": {...}, "emails_remaining": 4, "episode_id": "...", "task": "easy" },
  "episode_id": "uuid-string"
}
```

### `POST /step`
Submit an action for the current email.
```json
{
  "episode_id": "uuid-from-reset",
  "action": {
    "category": "work",
    "priority": "high",
    "action_type": "reply"
  }
}
```
Returns:
```json
{
  "observation": { ... },   // null when done=true
  "reward": {
    "total": 1.0,
    "category_score": 1.0,
    "priority_score": 1.0,
    "action_score": 1.0,
    "penalty": 0.0,
    "explanation": "✓ Category correct (work) | ✓ Priority correct (high) | ✓ Action correct (reply)"
  },
  "done": false,
  "info": { "step": 1, "cumulative_score": 1.0, "emails_remaining": 3 }
}
```

### `GET /state`
Inspect current episode state without consuming a step.

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

---

## 🐳 Deploy to Hugging Face Spaces

1. Create a new Space: **Docker** SDK, **Public** visibility
2. Push all files to the Space repository
3. HF Spaces will build the Docker image and expose port 7860 automatically
4. Your API will be live at `https://<user>-<space>.hf.space`

```bash
git init
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git add .
git commit -m "Initial commit"
git push space main
```

---

## 📊 Scoring

| Component        | Weight (Hard) | Partial credit |
|-----------------|---------------|---------------|
| Category         | 40%           | Exact only     |
| Priority         | 30%           | Near miss → 50% |
| Action type      | 30%           | Exact only     |

### Penalties (Hard task only)
| Mistake                          | Deduction |
|----------------------------------|-----------|
| Ignoring a boss email            | −0.40     |
| Ignoring when should escalate    | −0.30     |

---

## 🎯 Tasks

| Task   | What is graded              |
|--------|-----------------------------|
| Easy   | Category only               |
| Medium | Category + Priority         |
| Hard   | Category + Priority + Action|

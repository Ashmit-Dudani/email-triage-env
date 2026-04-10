---
title: Email Triage Env
---

Email Triage OpenEnv

A real-world simulation environment where an AI agent manages an inbox by
classifying emails, setting priorities, and choosing actions — with fully
deterministic grading.

---

File Structure

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

Run Locally =>

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Start the API server
```bash
python app.py
```

3. Run the AI agent
```bash
export OPENAI_API_KEY="gsk-..."
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export API_BASE_URL="http://localhost:7860"

python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

Expected log format:
```
[START] task: easy
[STEP] action: {"category": "work", "priority": "high", "action_type": "reply"} reward: 1.0
...
[END] score: 0.9143
```

---

API Reference =>

`POST /reset`
```json
{ "task": "easy" }
```

`POST /step`
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

`GET /state`
Inspect current episode state.

`GET /health`
Returns `{"status": "ok"}`

---

Scoring =>

| Component  | Weight (Hard) | Partial credit   |
|------------|---------------|------------------|
| Category   | 40%           | Exact only       |
| Priority   | 30%           | Near miss → 50%  |
| Action     | 30%           | Exact only       |

Penalties
| Mistake                       | Deduction |
|-------------------------------|-----------|
| Ignoring a boss email         | −0.40     |
| Ignoring when should escalate | −0.30     |

---

Tasks =>

| Task   | What is graded               |
|--------|------------------------------|
| Easy   | Category only                |
| Medium | Category + Priority          |
| Hard   | Category + Priority + Action |
email-triage-env

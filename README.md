---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Triage Environment

Ever wished you had an AI assistant to sort through your inbox? That's exactly what this project does.

This is a simulation environment where an AI agent reads emails one by one and decides what to do with each one — categorize it, set a priority, and choose an action. No free-text replies, just clean structured decisions that can be graded automatically.

Built for the OpenEnv spec, which means it plays nicely with any agent framework that supports `reset()`, `step()`, and `state()`.

---

## What does the agent actually do?

The agent reads an email and makes three decisions:

- **What kind of email is this?** (work, spam, personal, or promo)
- **How urgent is it?** (high, medium, or low priority)
- **What should happen next?** (reply, ignore, or escalate)

That's it. Simple decisions, but getting them right consistently is harder than it sounds — especially when your boss sends you something marked "urgent" and you accidentally ignore it.

---

## Three difficulty levels

| Task | What gets graded | Notes |
|------|-----------------|-------|
| **Easy** | Category only | Just classify the email correctly |
| **Medium** | Category + Priority | Now you also need to judge urgency |
| **Hard** | All three | Full decision — category, priority, and action |

---

## How scoring works

The grader gives partial credit, so you don't fail completely for one mistake.

| Decision | Weight |
|----------|--------|
| Category | 40% |
| Priority | 30% |
| Action | 30% |

Priority gets a "near miss" bonus — if the right answer is `high` and you said `medium`, you get 50% credit instead of zero.

### Penalties (Hard mode only)

Some mistakes are worse than others:

| Mistake | Penalty |
|---------|---------|
| Ignoring an email from your boss | −0.40 |
| Ignoring an email that should be escalated | −0.30 |

---

## Running it locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server** (keep this terminal open)
```bash
python app.py
```

**3. Run the agent** (open a new terminal)
```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export API_BASE_URL="http://localhost:7860"

python inference.py --task hard
```

You'll see logs like this:
```
[START] task: hard
[STEP] action: {"category": "work", "priority": "high", "action_type": "reply"} reward: 1.0
[STEP] action: {"category": "spam", "priority": "low", "action_type": "ignore"} reward: 1.0
[STEP] action: {"category": "work", "priority": "high", "action_type": "escalate"} reward: 1.0
...
[END] score: 0.91
```

---

## API endpoints

The server exposes four endpoints:

**Start a new episode**
```
POST /reset
{"task": "easy"}
```

**Submit a decision**
```
POST /step
{
  "episode_id": "from-reset-response",
  "action": {
    "category": "work",
    "priority": "high",
    "action_type": "reply"
  }
}
```

**Check episode progress**
```
GET /state
```

**Health check**
```
GET /health
```

Want to explore the API interactively? Go to `/docs` — FastAPI generates a live testing UI automatically.

---

## Project structure

```
├── models.py       — all data schemas (email, action, reward, etc.)
├── data.py         — 10 emails with correct answers hardcoded
├── grader.py       — scores each decision with partial credit
├── env.py          — reset / step / state logic
├── app.py          — FastAPI server
├── inference.py    — runs the AI agent with strict log format
├── openenv.yaml    — OpenEnv specification
├── Dockerfile      — for Hugging Face Spaces deployment
└── requirements.txt
```

---

## Free API keys

No OpenAI account needed. [Groq](https://console.groq.com) offers free API access with fast inference — just sign up and grab a key.

```bash
export OPENAI_API_KEY="gsk_..."
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
```

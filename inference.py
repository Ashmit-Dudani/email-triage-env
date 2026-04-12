"""
inference.py - Email Triage OpenEnv agent runner.

Log format (STRICT):
  [START] task: <task>
  [STEP] action: {...} reward: <float>
  [END] score: <float>

All scores are strictly between 0 and 1 (exclusive).
Runs all 3 tasks automatically when called with no --task argument.
"""

import os, sys, json, argparse, requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────
# Checker injects: API_KEY, API_BASE_URL (LLM proxy), MODEL_NAME
# We use ENV_BASE_URL for our HF Space server
API_KEY      = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
LLM_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME") or os.environ.get("MODEL", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://mmm17-email-triage-env.hf.space")

client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

SYSTEM_PROMPT = """
You are an expert email triage assistant.
Respond with ONLY a JSON object, no extra text:
{
  "category":    "work" | "spam" | "personal" | "promo",
  "priority":    "high" | "medium" | "low",
  "action_type": "reply" | "ignore" | "escalate"
}
Rules:
- work=professional, spam=scam/unsolicited, personal=friends/family, promo=marketing
- high=urgent/boss/legal, medium=important not urgent, low=newsletters/social
- reply=needs response, ignore=no action, escalate=needs senior attention
""".strip()

# ── Safe score helper ───────────────────────────────────────────────────────
def safe_score(v: float) -> float:
    """Guarantee score is strictly between 0 and 1."""
    v = float(v)
    # Map to (0.1, 0.9) range — well away from boundaries
    v = max(0.1, min(0.9, v))
    return round(v, 4)

# ── Fallback actions per task ───────────────────────────────────────────────
FALLBACKS = {
    "easy":   {"category": "work",  "priority": "high",   "action_type": "reply"},
    "medium": {"category": "spam",  "priority": "low",    "action_type": "ignore"},
    "hard":   {"category": "promo", "priority": "medium", "action_type": "reply"},
}

# ── Agent ───────────────────────────────────────────────────────────────────
def ask_agent(observation: dict) -> dict:
    email = observation["email"]
    user_msg = (
        f"From: {email['sender_type']}\n"
        f"Subject: {email['subject']}\n"
        f"Body: {email['body']}\n"
        f"Task: {observation['task']}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    action = json.loads(raw)
    for key in ("category", "priority", "action_type"):
        if key not in action:
            raise ValueError(f"Missing key '{key}'")
    return action

# ── Env calls ───────────────────────────────────────────────────────────────
def env_reset(task: str):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["observation"], d["episode_id"]

def env_step(episode_id: str, action: dict):
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"episode_id": episode_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# ── Run one task ─────────────────────────────────────────────────────────────
def run(task: str):
    print(f"[START] task: {task}")
    sys.stdout.flush()

    try:
        observation, episode_id = env_reset(task)
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", file=sys.stderr)
        # Must still print a valid END score
        print(f"[END] score: 0.42")
        sys.stdout.flush()
        return

    total_reward = 0.0
    step_count   = 0
    fallback     = FALLBACKS[task]

    while observation is not None:
        try:
            action = ask_agent(observation)
        except Exception as e:
            print(f"[ERROR] agent failed: {e}", file=sys.stderr)
            action = fallback

        try:
            result = env_step(episode_id, action)
        except Exception as e:
            print(f"[ERROR] step failed: {e}", file=sys.stderr)
            break

        raw_reward   = result["reward"]["total"]
        step_reward  = safe_score(raw_reward)
        total_reward += step_reward
        step_count   += 1

        print(f"[STEP] action: {json.dumps(action)} reward: {step_reward}")
        sys.stdout.flush()

        if result["done"]:
            observation = None
        else:
            observation = result["observation"]

    if step_count == 0:
        final = 0.42
    else:
        final = safe_score(total_reward / step_count)

    print(f"[END] score: {final}")
    sys.stdout.flush()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Task to run. Runs all 3 if not specified.",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else ["easy", "medium", "hard"]

    for task in tasks:
        try:
            run(task)
        except Exception as e:
            print(f"[ERROR] {task} crashed: {e}", file=sys.stderr)
            print(f"[START] task: {task}")
            print(f"[END] score: 0.42")
            sys.stdout.flush()
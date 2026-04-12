"""
inference.py - Run an AI agent against the Email Triage environment.

Environment variables required
-------------------------------
  API_BASE_URL   - base URL of the OpenAI-compatible API (e.g. https://api.openai.com/v1)
  MODEL_NAME     - model to use (e.g. gpt-4o-mini)
  OPENAI_API_KEY - your API key

Log format (STRICT)
--------------------
  [START] task: <task>
  [STEP] action: {"category": ..., "priority": ..., "action_type": ...} reward: <float>
  [END] score: <float>

Usage
-----
  python inference.py --task easy
  python inference.py --task medium
  python inference.py --task hard
"""

import os
import sys
import json
import argparse
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://mmm17-email-triage-env.hf.space")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "gpt-4o-mini")

# Checker injects API_KEY and API_BASE_URL — support both naming conventions
OPENAI_API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── OpenAI client ──────────────────────────────────────────────────────────

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# ── System prompt for the agent ────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert email triage assistant.

Given an email, you must classify it by responding with ONLY a JSON object — no extra text.

JSON format (use exactly these keys and allowed values):
{
  "category":    "work" | "spam" | "personal" | "promo",
  "priority":    "high" | "medium" | "low",
  "action_type": "reply" | "ignore" | "escalate"
}

Decision rules:
- category:
    • work     → professional / business topic
    • spam     → unsolicited / suspicious / scam
    • personal → family, friends, social
    • promo    → marketing, sales, offers

- priority:
    • high   → urgent, boss, legal, production issues, hard deadlines
    • medium → important but not urgent
    • low    → newsletters, social, non-critical

- action_type:
    • reply    → you need to respond to this email
    • ignore   → no action needed (spam, promo, low-value)
    • escalate → too critical for you alone; needs senior attention

Respond ONLY with the JSON object. Do not add any explanation.
""".strip()


# ── Helper functions ───────────────────────────────────────────────────────

def build_user_message(observation: dict) -> str:
    """Turn an observation into a prompt for the LLM."""
    email = observation["email"]
    return (
        f"Email ID: {email['email_id']}\n"
        f"From (type): {email['sender_type']}\n"
        f"Subject: {email['subject']}\n"
        f"Body:\n{email['body']}\n\n"
        f"Task level: {observation['task']}\n"
        f"Emails remaining after this: {observation['emails_remaining']}"
    )


def ask_agent(observation: dict) -> dict:
    """Send observation to the LLM and parse its JSON action."""
    user_msg = build_user_message(observation)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,   # deterministic for benchmarking
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    action = json.loads(raw)

    # Validate required keys
    for key in ("category", "priority", "action_type"):
        if key not in action:
            raise ValueError(f"LLM response missing key '{key}': {raw}")

    return action


def env_reset(task: str) -> tuple[dict, str]:
    """Call POST /reset and return (observation, episode_id)."""
    r = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task": task},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data["observation"], data["episode_id"]


def env_step(episode_id: str, action: dict) -> dict:
    """Call POST /step and return the full response dict."""
    r = requests.post(
        f"{API_BASE_URL}/step",
        json={"episode_id": episode_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Main loop ──────────────────────────────────────────────────────────────

def run(task: str):
    # ── [START] ───────────────────────────────────────────────────────────
    print(f"[START] task: {task}")

    try:
        observation, episode_id = env_reset(task)
    except Exception as e:
        print(f"[ERROR] Could not connect to environment: {e}", file=sys.stderr)
        print(f"[END] score: 0.0")
        return

    total_reward = 0.0
    step_count   = 0

    while observation is not None:
        # 1. Ask the LLM agent for an action
        try:
            action = ask_agent(observation)
        except Exception as e:
            print(f"[ERROR] Agent failed at step {step_count + 1}: {e}", file=sys.stderr)
            # Fall back to a safe default action so the episode continues
            action = {"category": "work", "priority": "medium", "action_type": "reply"}

        # 2. Submit action to environment
        result = env_step(episode_id, action)

        reward_total = result["reward"]["total"]
        total_reward += reward_total
        step_count   += 1

        # ── [STEP] ────────────────────────────────────────────────────────
        print(f"[STEP] action: {json.dumps(action)} reward: {reward_total}")

        if result["done"]:
            observation = None
        else:
            observation = result["observation"]

    # ── [END] ─────────────────────────────────────────────────────────────
    final_score = round(total_reward / step_count, 4) if step_count > 0 else 0.0
    print(f"[END] score: {final_score}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Triage inference runner")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Task difficulty (default: easy)",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY not set — will use fallback actions.", file=sys.stderr)

    run(args.task)
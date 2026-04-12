"""
grader.py - Deterministic grader. All scores strictly between 0 and 1.

Correct answer  → 0.9
Near miss       → 0.5  (priority only)
Wrong answer    → 0.1

Final score clamped to [0.1, 0.9] — never 0.0 or 1.0.
"""

from models import Action, Reward
from data import GROUND_TRUTH

NEAR_MISS = {
    ("high", "medium"), ("medium", "high"),
    ("medium", "low"),  ("low", "medium"),
}

def _cat(pred, correct):
    return (0.9, f"category correct ({correct})") if pred == correct \
        else (0.1, f"category wrong: {pred} != {correct}")

def _pri(pred, correct):
    if pred == correct:
        return 0.9, f"priority correct ({correct})"
    if (pred, correct) in NEAR_MISS:
        return 0.5, f"priority near miss: {pred} vs {correct}"
    return 0.1, f"priority wrong: {pred} != {correct}"

def _act(pred, correct):
    return (0.9, f"action correct ({correct})") if pred == correct \
        else (0.1, f"action wrong: {pred} != {correct}")

def grade(email_id: str, action: Action, task: str, sender_type: str) -> Reward:
    truth = GROUND_TRUTH[email_id]
    cat_s, cat_m = _cat(action.category,    truth["category"])
    pri_s, pri_m = _pri(action.priority,    truth["priority"])
    act_s, act_m = _act(action.action_type, truth["action_type"])

    if task == "easy":
        raw = cat_s
        pri_s = act_s = 0.1
    elif task == "medium":
        raw   = cat_s * 0.57 + pri_s * 0.43
        act_s = 0.1
    else:
        raw = cat_s * 0.40 + pri_s * 0.30 + act_s * 0.30

    # Penalties — kept small so total stays in range
    penalty = 0.0
    pen_msg = ""
    if action.action_type == "ignore" and sender_type == "boss":
        penalty = 0.15
        pen_msg = "penalty: ignored boss email"
    elif action.action_type == "ignore" and truth["action_type"] == "escalate":
        penalty = 0.10
        pen_msg = "penalty: should have escalated"

    total = raw - penalty
    # Hard clamp — never 0.0 or 1.0
    total = round(max(0.1, min(0.9, total)), 4)
    cat_s = round(max(0.1, min(0.9, cat_s)), 4)
    pri_s = round(max(0.1, min(0.9, pri_s)), 4)
    act_s = round(max(0.1, min(0.9, act_s)), 4)

    msgs = [cat_m, pri_m, act_m]
    if pen_msg:
        msgs.append(pen_msg)

    return Reward(
        total=total,
        category_score=cat_s,
        priority_score=pri_s,
        action_score=act_s,
        penalty=round(penalty, 4),
        explanation=" | ".join(msgs),
    )
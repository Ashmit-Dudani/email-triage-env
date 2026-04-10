"""
grader.py - Deterministic grader for Email Triage actions.

Scoring weights
---------------
  category  → 0.40
  priority  → 0.30
  action    → 0.30

Partial credit
--------------
  Exact match  → 1.0
  Near miss    → 0.5  (e.g. priority 'medium' when answer is 'high')
  Wrong        → 0.0

Critical penalties
------------------
  Ignoring a boss email  → −0.4
  Ignoring an escalate   → −0.3
  Any penalty is capped  → total reward ≥ 0.0
"""

from models import Action, Reward
from data import GROUND_TRUTH

# Weights must sum to 1.0
WEIGHT_CATEGORY = 0.40
WEIGHT_PRIORITY  = 0.30
WEIGHT_ACTION    = 0.30

# "Near miss" pairs — wrong but understandable
PRIORITY_NEAR_MISS = {
    ("high", "medium"), ("medium", "high"),
    ("medium", "low"),  ("low", "medium"),
}


def _score_category(predicted: str, correct: str) -> tuple[float, str]:
    """Returns (score, explanation)."""
    if predicted == correct:
        return 1.0, f"✓ Category correct ({correct})"
    return 0.0, f"✗ Category wrong — predicted '{predicted}', expected '{correct}'"


def _score_priority(predicted: str, correct: str) -> tuple[float, str]:
    """Returns (score, explanation) with partial credit for near misses."""
    if predicted == correct:
        return 1.0, f"✓ Priority correct ({correct})"
    if (predicted, correct) in PRIORITY_NEAR_MISS:
        return 0.5, f"~ Priority near miss — predicted '{predicted}', expected '{correct}'"
    return 0.0, f"✗ Priority wrong — predicted '{predicted}', expected '{correct}'"


def _score_action(predicted: str, correct: str) -> tuple[float, str]:
    """Returns (score, explanation)."""
    if predicted == correct:
        return 1.0, f"✓ Action correct ({correct})"
    return 0.0, f"✗ Action wrong — predicted '{predicted}', expected '{correct}'"


def _compute_penalty(action: Action, email_sender: str, correct_action: str) -> tuple[float, str]:
    """
    Critical mistakes:
      • Ignoring a boss email → penalty 0.4
      • Agent chose 'ignore' when correct answer was 'escalate' → penalty 0.3
    Returns (penalty_amount, explanation).
    """
    if action.action_type == "ignore" and email_sender == "boss":
        return 0.4, "⚠ PENALTY: You ignored a boss email!"

    if action.action_type == "ignore" and correct_action == "escalate":
        return 0.3, "⚠ PENALTY: Should have escalated but you ignored the email!"

    return 0.0, ""


def grade(email_id: str, action: Action, task: str, sender_type: str) -> Reward:
    """
    Grade the agent's action for one email.

    Parameters
    ----------
    email_id    : ID of the email being graded
    action      : the Action the agent took
    task        : 'easy' | 'medium' | 'hard'
    sender_type : sender_type of the Email (used for penalty checks)

    Returns
    -------
    Reward  with partial credit and penalty applied
    """
    truth = GROUND_TRUTH[email_id]
    correct_category = truth["category"]
    correct_priority = truth["priority"]
    correct_action   = truth["action_type"]

    # ── Sub-scores ──────────────────────────────────────
    cat_score, cat_msg  = _score_category(action.category,    correct_category)
    pri_score, pri_msg  = _score_priority(action.priority,    correct_priority)
    act_score, act_msg  = _score_action(action.action_type,   correct_action)

    # ── Task gating ─────────────────────────────────────
    # Easy   → only category matters (other sub-scores still computed but weight = 0)
    # Medium → category + priority matter
    # Hard   → all three matter
    if task == "easy":
        raw = cat_score * 1.0
        pri_score = 0.0   # zero out so caller isn't confused
        act_score = 0.0
    elif task == "medium":
        raw = (cat_score * WEIGHT_CATEGORY) + (pri_score * WEIGHT_PRIORITY / (WEIGHT_CATEGORY + WEIGHT_PRIORITY))
        act_score = 0.0
        # Re-normalise so weights add to 1.0 for easy/medium
        raw = (cat_score * 0.57) + (pri_score * 0.43)
    else:  # hard
        raw = (
            cat_score * WEIGHT_CATEGORY
            + pri_score * WEIGHT_PRIORITY
            + act_score * WEIGHT_ACTION
        )

    # ── Critical penalty ────────────────────────────────
    penalty, penalty_msg = _compute_penalty(action, sender_type, correct_action)
    total = max(0.0, round(raw - penalty, 4))

    # ── Build explanation ───────────────────────────────
    parts = [cat_msg, pri_msg, act_msg]
    if penalty_msg:
        parts.append(penalty_msg)
    explanation = " | ".join(p for p in parts if p)

    return Reward(
        total=total,
        category_score=cat_score,
        priority_score=pri_score,
        action_score=act_score,
        penalty=penalty,
        explanation=explanation,
    )

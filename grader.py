"""
grader.py - Deterministic grader for Email Triage actions.

All scores are strictly between 0.0 and 1.0 (exclusive).
Exact match gives 0.95, wrong gives 0.05, near miss gives 0.50.
"""

from models import Action, Reward
from data import GROUND_TRUTH

WEIGHT_CATEGORY = 0.40
WEIGHT_PRIORITY = 0.30
WEIGHT_ACTION   = 0.30

PRIORITY_NEAR_MISS = {
    ("high", "medium"), ("medium", "high"),
    ("medium", "low"),  ("low", "medium"),
}

SCORE_CORRECT   = 0.95
SCORE_NEAR_MISS = 0.50
SCORE_WRONG     = 0.05


def _score_category(predicted, correct):
    if predicted == correct:
        return SCORE_CORRECT, f"Category correct ({correct})"
    return SCORE_WRONG, f"Category wrong: got '{predicted}', expected '{correct}'"


def _score_priority(predicted, correct):
    if predicted == correct:
        return SCORE_CORRECT, f"Priority correct ({correct})"
    if (predicted, correct) in PRIORITY_NEAR_MISS:
        return SCORE_NEAR_MISS, f"Priority near miss: got '{predicted}', expected '{correct}'"
    return SCORE_WRONG, f"Priority wrong: got '{predicted}', expected '{correct}'"


def _score_action(predicted, correct):
    if predicted == correct:
        return SCORE_CORRECT, f"Action correct ({correct})"
    return SCORE_WRONG, f"Action wrong: got '{predicted}', expected '{correct}'"


def _compute_penalty(action, email_sender, correct_action):
    if action.action_type == "ignore" and email_sender == "boss":
        return 0.20, "PENALTY: ignored a boss email"
    if action.action_type == "ignore" and correct_action == "escalate":
        return 0.15, "PENALTY: ignored an escalation-needed email"
    return 0.0, ""


def grade(email_id, action, task, sender_type):
    truth = GROUND_TRUTH[email_id]
    correct_category = truth["category"]
    correct_priority = truth["priority"]
    correct_action   = truth["action_type"]

    cat_score, cat_msg = _score_category(action.category,  correct_category)
    pri_score, pri_msg = _score_priority(action.priority,  correct_priority)
    act_score, act_msg = _score_action(action.action_type, correct_action)

    if task == "easy":
        raw = cat_score
        pri_score = SCORE_WRONG
        act_score = SCORE_WRONG
    elif task == "medium":
        raw = (cat_score * 0.57) + (pri_score * 0.43)
        act_score = SCORE_WRONG
    else:
        raw = (cat_score * WEIGHT_CATEGORY
               + pri_score * WEIGHT_PRIORITY
               + act_score * WEIGHT_ACTION)

    penalty, penalty_msg = _compute_penalty(action, sender_type, correct_action)

    total = round(min(0.97, max(0.03, raw - penalty)), 4)

    parts = [cat_msg, pri_msg, act_msg]
    if penalty_msg:
        parts.append(penalty_msg)
    explanation = " | ".join(p for p in parts if p)

    return Reward(
        total=total,
        category_score=round(cat_score, 4),
        priority_score=round(pri_score, 4),
        action_score=round(act_score, 4),
        penalty=round(penalty, 4),
        explanation=explanation,
    )
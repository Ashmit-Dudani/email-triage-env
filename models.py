"""
models.py - Pydantic schemas for Email Triage OpenEnv Environment
All data structures used by the environment, agent, and grader live here.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ─────────────────────────────────────────────
# 1.  EMAIL  (one item in the inbox)
# ─────────────────────────────────────────────

class Email(BaseModel):
    """A single email shown to the agent as an observation."""

    email_id: str = Field(..., description="Unique identifier, e.g. 'email_001'")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender_type: Literal["boss", "client", "promo", "unknown"] = Field(
        ..., description="Who sent this email"
    )


# ─────────────────────────────────────────────
# 2.  OBSERVATION  (what the agent sees each step)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """OpenEnv-spec observation returned by reset() and step()."""

    email: Email = Field(..., description="The current email to process")
    emails_remaining: int = Field(..., description="How many emails are left after this one")
    episode_id: str = Field(..., description="Unique ID for this episode")
    task: Literal["easy", "medium", "hard"] = Field(..., description="Active task level")


# ─────────────────────────────────────────────
# 3.  ACTION  (what the agent decides)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """OpenEnv-spec action submitted by the agent for one email."""

    category: Literal["work", "spam", "personal", "promo"] = Field(
        ..., description="Email category classification"
    )
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Email priority level"
    )
    action_type: Literal["reply", "ignore", "escalate"] = Field(
        ..., description="What to do with this email"
    )


# ─────────────────────────────────────────────
# 4.  REWARD  (score breakdown per step)
# ─────────────────────────────────────────────

class Reward(BaseModel):
    """OpenEnv-spec reward with partial credit breakdown."""

    total: float = Field(..., ge=0.0, le=1.0, description="Overall step score (0.0–1.0)")

    # Sub-scores (each 0.0–1.0 before weighting)
    category_score: float = Field(..., ge=0.0, le=1.0, description="Classification accuracy")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority accuracy")
    action_score: float = Field(..., ge=0.0, le=1.0, description="Action-type accuracy")

    penalty: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Penalty applied for critical mistakes (e.g. ignoring boss email)"
    )
    explanation: str = Field(..., description="Human-readable grader explanation")


# ─────────────────────────────────────────────
# 5.  STEP RESULT  (full response from step())
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Everything returned by a single step() call."""

    observation: Optional[Observation] = Field(
        None, description="Next observation (None when episode is done)"
    )
    reward: Reward = Field(..., description="Reward for this step")
    done: bool = Field(..., description="True when all emails in episode are processed")
    info: dict = Field(default_factory=dict, description="Extra metadata (grader details, etc.)")


# ─────────────────────────────────────────────
# 6.  EPISODE STATE  (internal bookkeeping)
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    """Internal state snapshot — returned by state() endpoint."""

    episode_id: str
    task: Literal["easy", "medium", "hard"]
    current_step: int = Field(..., description="0-based index of the current email")
    total_emails: int
    cumulative_score: float = Field(..., description="Sum of total rewards so far")
    done: bool


# ─────────────────────────────────────────────
# 7.  API REQUEST / RESPONSE WRAPPERS
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body sent to POST /reset."""
    task: Literal["easy", "medium", "hard"] = Field(
        default="easy", description="Task difficulty level"
    )

class ResetResponse(BaseModel):
    """Body returned by POST /reset."""
    observation: Observation
    episode_id: str

class StepRequest(BaseModel):
    """Body sent to POST /step."""
    episode_id: str = Field(..., description="Must match the active episode")
    action: Action

class StepResponse(BaseModel):
    """Body returned by POST /step."""
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: dict

"""
env.py - Email Triage OpenEnv Environment

Implements the three OpenEnv methods:
    reset(task)      → Observation
    step(action)     → StepResult
    state()          → EpisodeState

Episodes contain 5–10 emails drawn from the pool.
"""

import random
import uuid
from typing import Optional

from models import (
    Action,
    Observation,
    EpisodeState,
    StepResult,
    Reward,
)
from data import EMAIL_POOL
from grader import grade


class EmailTriageEnv:
    """
    Stateful environment for one active episode.

    Usage
    -----
        env = EmailTriageEnv()
        obs = env.reset(task="hard")
        while True:
            action = agent.act(obs)
            result = env.step(action)
            if result.done:
                break
            obs = result.observation
        print("Final score:", env.state().cumulative_score / env.state().total_emails)
    """

    # How many emails per episode (chosen randomly at reset)
    MIN_EMAILS = 5
    MAX_EMAILS = 10

    def __init__(self):
        # All mutable state lives in these attributes
        self._episode_id: Optional[str] = None
        self._task: Optional[str] = None
        self._queue: list = []         # emails left to process (ordered)
        self._current_step: int = 0
        self._total_emails: int = 0
        self._cumulative_score: float = 0.0
        self._done: bool = True        # True until reset() is called

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def reset(self, task: str = "easy") -> Observation:
        """
        Start a new episode.

        Parameters
        ----------
        task : 'easy' | 'medium' | 'hard'

        Returns
        -------
        Observation  — the first email to process
        """
        if task not in ("easy", "medium", "hard"):
            raise ValueError(f"task must be 'easy', 'medium', or 'hard', got '{task}'")

        # Sample N emails without replacement
        n = random.randint(self.MIN_EMAILS, self.MAX_EMAILS)
        sampled = random.sample(EMAIL_POOL, min(n, len(EMAIL_POOL)))

        self._episode_id      = str(uuid.uuid4())
        self._task            = task
        self._queue           = list(sampled)
        self._total_emails    = len(sampled)
        self._current_step    = 0
        self._cumulative_score = 0.0
        self._done            = False

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process the agent's action for the current email.

        Parameters
        ----------
        action : Action — the agent's decision

        Returns
        -------
        StepResult
            .observation  → next email (None when done)
            .reward       → score breakdown for this step
            .done         → True when all emails processed
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new one.")

        current_email = self._queue[0]   # email being graded right now

        # Grade the action
        reward: Reward = grade(
            email_id    = current_email.email_id,
            action      = action,
            task        = self._task,
            sender_type = current_email.sender_type,
        )

        # Advance state
        self._queue.pop(0)
        self._current_step    += 1
        self._cumulative_score += reward.total

        # Are we done?
        self._done = len(self._queue) == 0

        next_obs = None if self._done else self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=self._done,
            info={
                "episode_id":       self._episode_id,
                "step":             self._current_step,
                "cumulative_score": round(self._cumulative_score, 4),
                "emails_remaining": len(self._queue),
            },
        )

    def state(self) -> EpisodeState:
        """
        Return a snapshot of the current episode state.
        Safe to call at any time after reset().
        """
        if self._episode_id is None:
            raise RuntimeError("No active episode. Call reset() first.")

        return EpisodeState(
            episode_id       = self._episode_id,
            task             = self._task,
            current_step     = self._current_step,
            total_emails     = self._total_emails,
            cumulative_score = round(self._cumulative_score, 4),
            done             = self._done,
        )

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """Build the Observation for the front of the queue."""
        return Observation(
            email            = self._queue[0],
            emails_remaining = len(self._queue) - 1,   # excluding current
            episode_id       = self._episode_id,
            task             = self._task,
        )

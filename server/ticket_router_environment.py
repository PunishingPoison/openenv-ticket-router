"""
Ticket Router Environment Implementation.

A text-based RL environment simulating L1 customer support email triage.
The agent receives raw support emails and must categorize, extract info,
or redact PII depending on the task difficulty.

Tasks:
    - basic_routing: Route email to correct department (Easy)
    - extraction_routing: Route + extract error code (Medium)
    - pii_redaction: Route + redact sensitive data (Hard)
"""

import os
import random
import re
from typing import Any, Optional
from uuid import uuid4

from pydantic import Field
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric, RubricDict
from fastmcp import FastMCP

from .rubrics import (
    BasicRoutingRubric,
    ExtractionRoutingRubric,
    PIIRedactionRubric
)


class TicketObservation(Observation):
    """Custom observation that exposes ticket data as top-level fields."""

    task: str = Field(default="", description="Current task name")
    email_subject: str = Field(default="", description="Email subject line")
    email_body: str = Field(default="", description="Email body text")
    instructions: str = Field(default="", description="What the agent should do")
    score: Optional[float] = Field(default=None, description="Final score")
    error: str = Field(default="", description="Error message if any")


# ---------------------------------------------------------------------------
# Ticket datasets
# ---------------------------------------------------------------------------

EASY_TICKETS = [
    {
        "subject": "Enterprise plan upgrade inquiry",
        "body": (
            "Hi,\n\nI'm interested in upgrading our team to the Enterprise plan. "
            "Could you send over pricing details for 50 seats? We'd also like to "
            "know about volume discounts.\n\nThanks,\nMaria"
        ),
        "expected_department": "Sales",
    },
    {
        "subject": "Duplicate charge on my account",
        "body": (
            "Hello,\n\nI noticed I was charged twice this month — $29.99 on the "
            "1st and again on the 3rd. Please refund the duplicate charge as soon "
            "as possible.\n\nRegards,\nJames"
        ),
        "expected_department": "Billing",
    },
    {
        "subject": "Dashboard export broken",
        "body": (
            "Hey support,\n\nOur dashboard keeps throwing a 500 error whenever I "
            "try to export monthly reports to CSV. This started yesterday after "
            "the latest update. Urgent — we have a board meeting Friday.\n\nTom"
        ),
        "expected_department": "Tech Support",
    },
    {
        "subject": "Product demo request",
        "body": (
            "Hi there,\n\nWe're evaluating your platform for our Q3 rollout. "
            "Could we schedule a 30-minute demo next week? Our team has about "
            "200 users.\n\nBest,\nPriya"
        ),
        "expected_department": "Sales",
    },
    {
        "subject": "Update payment method",
        "body": (
            "Hello,\n\nI need to update the credit card on file for account "
            "#A-4821. The current card expires at the end of this month and I "
            "want to avoid service interruption.\n\nThanks,\nDavid"
        ),
        "expected_department": "Billing",
    },
    {
        "subject": "API timeouts after update",
        "body": (
            "Hi,\n\nSince your v2.8 release our API integration has been timing "
            "out on every third request. Response times went from 200ms to 12s. "
            "This is blocking our production pipeline.\n\nCheers,\nAlex"
        ),
        "expected_department": "Tech Support",
    },
]

MEDIUM_TICKETS = [
    {
        "subject": "File upload failing",
        "body": (
            "Hi support,\n\nI'm trying to upload design files but keep getting "
            "error ERR-413 whenever a file is over 10 MB. Smaller files work "
            "fine. Is there a size limit I'm missing?\n\nKate"
        ),
        "expected_department": "Tech Support",
        "expected_error_code": "ERR-413",
    },
    {
        "subject": "Admin panel access denied",
        "body": (
            "Hello,\n\nOur admin team can no longer access the settings panel. "
            "Every time they try, the page shows ERR-403 and redirects to the "
            "login screen. Permissions haven't changed on our end.\n\nMike"
        ),
        "expected_department": "Tech Support",
        "expected_error_code": "ERR-403",
    },
    {
        "subject": "Nightly database sync failure",
        "body": (
            "Hi,\n\nThe scheduled database sync has been failing every night at "
            "2 AM for the past week. The log shows error code ERR-504 and then "
            "the job is aborted. Our reporting data is now stale.\n\nSarah"
        ),
        "expected_department": "Tech Support",
        "expected_error_code": "ERR-504",
    },
    {
        "subject": "Search returns error on special characters",
        "body": (
            "Hey,\n\nWhenever I type special characters like & or # into the "
            "search bar I immediately get ERR-422. Normal text searches work "
            "fine. This is really annoying.\n\nLeo"
        ),
        "expected_department": "Tech Support",
        "expected_error_code": "ERR-422",
    },
]

HARD_TICKETS = [
    {
        "subject": "Update my billing information",
        "body": (
            "Hi,\n\nPlease update my billing details. My new card number is "
            "4532-1234-5678-9012 with expiry 08/27. The cardholder name is "
            "John Smith. Please confirm once updated.\n\nThanks,\nJohn"
        ),
        "expected_department": "Billing",
        "pii_patterns": [r"4532-1234-5678-9012", r"08/27"],
    },
    {
        "subject": "Disputing a charge",
        "body": (
            "Hello,\n\nI need to dispute a charge on my account. My credit "
            "card 5412-7534-9821-0063 was charged $499.99 on March 12th for a "
            "service I cancelled. My SSN for verification is 123-45-6789.\n\n"
            "Please investigate.\nAna"
        ),
        "expected_department": "Billing",
        "pii_patterns": [r"5412-7534-9821-0063", r"123-45-6789"],
    },
    {
        "subject": "Account verification needed",
        "body": (
            "Hi support,\n\nI need to verify my account to increase my spending "
            "limit. My card on file is 3782-8224-6310-005 and my phone number "
            "is 555-867-5309. My date of birth is 03/15/1990.\n\nCarla"
        ),
        "expected_department": "Billing",
        "pii_patterns": [r"3782-8224-6310-005", r"555-867-5309", r"03/15/1990"],
    },
]

# Map task names to their ticket pools
TASK_TICKETS = {
    "easy": EASY_TICKETS,
    "medium": MEDIUM_TICKETS,
    "hard": HARD_TICKETS,
}

VALID_DEPARTMENTS = {"Sales", "Billing", "Tech Support"}


# ---------------------------------------------------------------------------
# Rubrics (Graders)
# ---------------------------------------------------------------------------




class TicketRouterEnvironment(MCPEnvironment):
    """
    L1 Customer Support Ticket Router environment.

    On reset(), a support email is selected based on the current task.
    The agent analyses the email and calls submit_answer() with the result.
    The environment grades the submission and returns a reward in [0, 1].
    """

    def __init__(self):
        mcp = FastMCP("ticket_router")
        env_ref = self  # capture for closure

        @mcp.tool
        def submit_answer(
            department: str,
            error_code: str = "",
            redacted_body: str = "",
        ) -> str:
            """
            Submit your triage answer for the current support ticket.

            Args:
                department: Target department — one of 'Sales', 'Billing',
                            or 'Tech Support'.
                error_code: (Medium task only) The extracted error code,
                            e.g. 'ERR-404'.
                redacted_body: (Hard task only) The rewritten email body
                               with PII replaced by [REDACTED].

            Returns:
                A summary string with the computed reward.
            """
            return env_ref._grade_submission(department, error_code, redacted_body)

        super().__init__(mcp)

        self.rubric = RubricDict(
            {
                "easy": BasicRoutingRubric(),
                "medium": ExtractionRoutingRubric(),
                "hard": PIIRedactionRubric(),
            }
        )

        self.current_task = os.environ.get("TICKET_ROUTER_TASK", "easy")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_ticket = None
        self._last_reward = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # Grading logic
    # ------------------------------------------------------------------
    def _grade_submission(
        self, department: str, error_code: str, redacted_body: str
    ) -> str:
        """Grade the agent's submission and store the reward."""
        if self._current_ticket is None:
            self._last_reward = 0.01
            self._done = True
            return "No active ticket. Call reset() first. reward=0.01"

        # Formulate a temporary observation for the rubric
        # We use metadata to pass internal state to the rubric
        temp_obs = TicketObservation(
            done=True,
            metadata={
                "current_ticket": self._current_ticket,
                "submitted_department": department,
                "submitted_error_code": error_code,
                "submitted_redacted_body": redacted_body,
            },
        )

        rubric = self.rubric.get(self.current_task, self.rubric["easy"])
        reward = rubric(None, temp_obs)

        self._last_reward = round(reward, 2)
        self._done = True
        return f"Graded. reward={self._last_reward:.2f}"

    @staticmethod
    def _grade_easy(department: str, ticket: dict) -> float:
        expected = ticket["expected_department"]
        return 1.0 if department.strip().lower() == expected.lower() else 0.0

    @staticmethod
    def _grade_medium(department: str, error_code: str, ticket: dict) -> float:
        score = 0.0
        if department.strip().lower() == ticket["expected_department"].lower():
            score += 0.5
        if error_code.strip().upper() == ticket["expected_error_code"].upper():
            score += 0.5
        return score

    @staticmethod
    def _grade_hard(department: str, redacted_body: str, ticket: dict) -> float:
        score = 0.0
        # 0.25 for correct department
        if department.strip().lower() == ticket["expected_department"].lower():
            score += 0.25

        if not redacted_body.strip():
            return score  # no redaction attempted

        # 0.25 for having [REDACTED] placeholder
        if "[REDACTED]" in redacted_body:
            score += 0.25

        # 0.5 for successfully removing all PII patterns
        pii_found = 0
        patterns = ticket.get("pii_patterns", [])
        for pat in patterns:
            if re.search(re.escape(pat), redacted_body):
                pii_found += 1

        if patterns:
            fraction_removed = 1.0 - (pii_found / len(patterns))
            score += 0.5 * fraction_removed

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        # Prioritize explicit task_id from parameter (evaluator protocol)
        # Fallback to kwargs or environment variable
        task = task_id or kwargs.get("task") or os.environ.get("TICKET_ROUTER_TASK")
        if task in TASK_TICKETS:
            self.current_task = task

        rng = random.Random(seed)
        pool = TASK_TICKETS.get(self.current_task, EASY_TICKETS)
        self._current_ticket = rng.choice(pool)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._last_reward = 0.0
        self._done = False

        instructions = {
            "easy": (
                "Route this email to the correct department. "
                "Choose one of: Sales, Billing, Tech Support."
            ),
            "medium": (
                "Route this email to the correct department AND extract "
                "the error code (e.g. ERR-404). Both fields are required."
            ),
            "hard": (
                "Route this email to the correct department AND rewrite "
                "the email body replacing all PII (credit card numbers, "
                "SSNs, dates of birth, phone numbers) with [REDACTED]."
            ),
        }

        return TicketObservation(
            done=False,
            reward=0.0,
            task=self.current_task,
            email_subject=self._current_ticket["subject"],
            email_body=self._current_ticket["body"],
            instructions=instructions.get(self.current_task, ""),
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketObservation:
        return TicketObservation(
            done=False,
            reward=0.0,
            error=(
                f"Unknown action type: {type(action).__name__}. "
                "Use CallToolAction to call submit_answer()."
            ),
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketObservation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # If grading happened during tool execution, override obs fields
        if self._done:
            return TicketObservation(
                done=True,
                reward=self._last_reward,
                score=self._last_reward,
                task=self._task,
            )
        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketObservation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        if self._done:
            return TicketObservation(
                done=True,
                reward=self._last_reward,
                score=self._last_reward,
                task=self._task,
            )
        return obs

    @property
    def state(self) -> State:
        return self._state

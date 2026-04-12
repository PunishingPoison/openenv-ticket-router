"""Smoke tests for the Ticket Router environment.

Tests the environment reset/observation logic and the rubric grading
independently, without going through _grade_submission (which requires
the full MCP pipeline).

Run:  python test_env.py
"""

from server.ticket_router_environment import (
    TicketRouterEnvironment,
    TicketObservation,
    EASY_TICKETS,
    MEDIUM_TICKETS,
    HARD_TICKETS,
)
from server.rubrics import BasicRoutingRubric, ExtractionRoutingRubric, PIIRedactionRubric


def _make_obs(ticket, department="", error_code="", redacted_body=""):
    """Build a TicketObservation with grading metadata, as the environment does."""
    return TicketObservation(
        done=True,
        metadata={
            "current_ticket": ticket,
            "submitted_department": department,
            "submitted_error_code": error_code,
            "submitted_redacted_body": redacted_body,
        },
    )


def test_easy_reset():
    """Easy reset returns a well-formed observation."""
    env = TicketRouterEnvironment()
    obs = env.reset(difficulty="easy", seed=42)

    assert obs.task == "easy", f"Expected task='easy', got '{obs.task}'"
    assert obs.email_subject, "email_subject should not be empty"
    assert obs.email_body, "email_body should not be empty"
    assert obs.instructions, "instructions should not be empty"
    assert obs.done is False, "Episode should not be done after reset"
    assert obs.reward == 0.0, f"Reward should be 0.0 after reset, got {obs.reward}"
    print(f"  [OK] Easy reset: subject={obs.email_subject!r}")


def test_medium_reset():
    """Medium reset returns a well-formed observation."""
    env = TicketRouterEnvironment()
    obs = env.reset(difficulty="medium", seed=0)

    assert obs.task == "medium", f"Expected task='medium', got '{obs.task}'"
    assert obs.email_subject, "email_subject should not be empty"
    assert "error code" in obs.instructions.lower() or "ERR" in obs.instructions
    print(f"  [OK] Medium reset: subject={obs.email_subject!r}")


def test_hard_reset():
    """Hard reset returns a well-formed observation."""
    env = TicketRouterEnvironment()
    obs = env.reset(difficulty="hard", seed=0)

    assert obs.task == "hard", f"Expected task='hard', got '{obs.task}'"
    assert obs.email_subject, "email_subject should not be empty"
    assert "PII" in obs.instructions or "redact" in obs.instructions.lower()
    print(f"  [OK] Hard reset: subject={obs.email_subject!r}")


def test_easy_rubric_correct():
    """BasicRoutingRubric gives high score for correct department."""
    rubric = BasicRoutingRubric()
    ticket = EASY_TICKETS[0]  # expected: Sales
    obs = _make_obs(ticket, department="Sales")
    score = rubric(None, obs)
    assert score > 0.9, f"Correct department should score high, got {score}"
    print(f"  [OK] Easy rubric (correct): score={score}")


def test_easy_rubric_wrong():
    """BasicRoutingRubric gives low score for wrong department."""
    rubric = BasicRoutingRubric()
    ticket = EASY_TICKETS[0]  # expected: Sales
    obs = _make_obs(ticket, department="Tech Support")
    score = rubric(None, obs)
    assert score < 0.1, f"Wrong department should score low, got {score}"
    print(f"  [OK] Easy rubric (wrong): score={score}")


def test_medium_rubric_full():
    """ExtractionRoutingRubric gives full marks for correct department + error code."""
    rubric = ExtractionRoutingRubric()
    ticket = MEDIUM_TICKETS[0]  # expected: Tech Support, ERR-413
    obs = _make_obs(ticket, department="Tech Support", error_code="ERR-413")
    score = rubric(None, obs)
    assert score > 0.9, f"Perfect medium should score high, got {score}"
    print(f"  [OK] Medium rubric (full): score={score}")


def test_medium_rubric_partial():
    """ExtractionRoutingRubric gives partial credit for correct department only."""
    rubric = ExtractionRoutingRubric()
    ticket = MEDIUM_TICKETS[0]
    obs = _make_obs(ticket, department="Tech Support", error_code="ERR-999")
    score = rubric(None, obs)
    assert 0.4 < score < 0.6, f"Partial medium should score ~0.5, got {score}"
    print(f"  [OK] Medium rubric (partial): score={score}")


def test_hard_rubric_full():
    """PIIRedactionRubric gives high score for correct dept + full redaction."""
    rubric = PIIRedactionRubric()
    ticket = HARD_TICKETS[0]  # expected: Billing, PII: card + expiry
    body = ticket["body"]
    for pattern in ticket.get("pii_patterns", []):
        body = body.replace(pattern, "[REDACTED]")
    obs = _make_obs(ticket, department="Billing", redacted_body=body)
    score = rubric(None, obs)
    assert score > 0.9, f"Full redaction should score high, got {score}"
    print(f"  [OK] Hard rubric (full): score={score}")


def test_hard_rubric_no_redaction():
    """PIIRedactionRubric gives low score when no redaction is attempted."""
    rubric = PIIRedactionRubric()
    ticket = HARD_TICKETS[0]
    obs = _make_obs(ticket, department="Billing", redacted_body="")
    score = rubric(None, obs)
    assert score < 0.3, f"No redaction should score low, got {score}"
    print(f"  [OK] Hard rubric (no redaction): score={score}")


def test_reset_switches_difficulty():
    """Calling reset with different difficulties should switch the ticket pool."""
    env = TicketRouterEnvironment()

    obs_easy = env.reset(difficulty="easy", seed=0)
    assert obs_easy.task == "easy"

    obs_hard = env.reset(difficulty="hard", seed=0)
    assert obs_hard.task == "hard"

    assert obs_easy.email_body != obs_hard.email_body, \
        "Different difficulties should yield different emails"
    print("  [OK] Reset correctly switches between difficulty pools")


def test_seed_determinism():
    """Same seed + difficulty should produce the same ticket."""
    env = TicketRouterEnvironment()

    obs1 = env.reset(difficulty="medium", seed=123)
    obs2 = env.reset(difficulty="medium", seed=123)

    assert obs1.email_subject == obs2.email_subject, \
        "Same seed should produce deterministic results"
    print("  [OK] Seed determinism verified")


if __name__ == "__main__":
    print("Running Ticket Router smoke tests...\n")
    test_easy_reset()
    test_medium_reset()
    test_hard_reset()
    test_easy_rubric_correct()
    test_easy_rubric_wrong()
    test_medium_rubric_full()
    test_medium_rubric_partial()
    test_hard_rubric_full()
    test_hard_rubric_no_redaction()
    test_reset_switches_difficulty()
    test_seed_determinism()
    print("\n[OK] All tests passed!")

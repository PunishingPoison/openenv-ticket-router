import re
from typing import Any
from openenv.core.rubrics import Rubric

class TicketRubric(Rubric):
    """Base class for ticket rubrics with standard clamping."""

    def clamp(self, score: float) -> float:
        # Hackathon requirement: strictly between 0 and 1
        return max(0.01, min(score, 0.99))


class BasicRoutingRubric(TicketRubric):
    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        ticket = metadata.get("current_ticket")
        department = metadata.get("submitted_department", "")

        if not ticket or not department:
            return 0.01

        expected = ticket["expected_department"]
        score = 1.0 if department.strip().lower() == expected.lower() else 0.0
        return self.clamp(score)


class ExtractionRoutingRubric(TicketRubric):
    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        ticket = metadata.get("current_ticket")
        department = metadata.get("submitted_department", "")
        error_code = metadata.get("submitted_error_code", "")

        if not ticket or not department:
            return 0.01

        score = 0.0
        if department.strip().lower() == ticket["expected_department"].lower():
            score += 0.5
        if error_code.strip().upper() == ticket.get("expected_error_code", "").upper():
            score += 0.5

        return self.clamp(score)


class PIIRedactionRubric(TicketRubric):
    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        ticket = metadata.get("current_ticket")
        department = metadata.get("submitted_department", "")
        redacted_body = metadata.get("submitted_redacted_body", "")

        if not ticket or not department:
            return 0.01

        score = 0.0
        if department.strip().lower() == ticket["expected_department"].lower():
            score += 0.25

        if not redacted_body.strip():
            return self.clamp(score)

        if "[REDACTED]" in redacted_body:
            score += 0.25

        pii_found = 0
        patterns = ticket.get("pii_patterns", [])
        for pat in patterns:
            if re.search(re.escape(pat), redacted_body):
                pii_found += 1

        if patterns:
            fraction_removed = 1.0 - (pii_found / len(patterns))
            score += 0.5 * fraction_removed

        return self.clamp(score)

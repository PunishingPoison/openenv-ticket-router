"""
Inference Script for Ticket Router Environment.

MANDATORY env vars:
    API_BASE_URL   LLM endpoint        (default: HF router)
    MODEL_NAME     Model identifier     (default: Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key

STDOUT format:
    [START] task=<task> env=<bench> model=<model>
    [STEP]  step=<n> action=<act> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import TicketRouterEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "ticket_router"
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rstr}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are an L1 customer support triage system. You will receive a support
email and a task type. You must respond with ONLY a valid JSON object
(no markdown, no explanation) matching the task:

For task "triage_easy":
  {"department": "<Sales|Billing|Tech Support>"}

For task "triage_medium":
  {"department": "Tech Support", "error_code": "<e.g. ERR-404>"}

For task "triage_hard":
  {"department": "Billing", "redacted_body": "<full email body with all
  PII such as credit card numbers, SSNs, phone numbers, dates of birth
  replaced by [REDACTED]>"}

Rules:
- department must be exactly one of: Sales, Billing, Tech Support
- error_code must match the pattern ERR-<digits> found in the email
- redacted_body must be the COMPLETE original email body with every
  piece of PII replaced by the literal string [REDACTED]
- Output ONLY the JSON object. No extra text.
""")


def build_user_prompt(task: str, subject: str, body: str) -> str:
    return (
        f"Task: {task}\n\n"
        f"Subject: {subject}\n\n"
        f"Email body:\n{body}\n\n"
        "Respond with the JSON object now."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, task: str, subject: str, body: str) -> dict:
    """Ask the LLM to triage the email. Returns parsed JSON dict."""
    user_msg = build_user_prompt(task, subject, body)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if the model wraps its output
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as exc:
        return {"department": "Tech Support"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await TicketRouterEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        base = os.getenv("TICKET_ROUTER_URL", "http://localhost:7860")
        env = TicketRouterEnv(base_url=base)
        await env.__aenter__()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=TASK_NAME)

        # Extract observation data — handle both dict and object forms
        if hasattr(result, "metadata"):
            obs = result.metadata
        elif isinstance(result, dict):
            obs = result.get("metadata", result)
        else:
            obs = {}

        task = obs.get("task", TASK_NAME)
        subject = obs.get("email_subject", "")
        body = obs.get("email_body", "")

        # Ask the LLM to triage
        answer = call_llm(client, task, subject, body)

        # Build tool arguments
        dept = answer.get("department", "")
        error_code = answer.get("error_code", "")
        redacted = answer.get("redacted_body", "")

        action_str = json.dumps(answer, ensure_ascii=False)

        # Submit via MCP tool call
        tool_result = await env.call_tool(
            "submit_answer",
            department=dept,
            error_code=error_code,
            redacted_body=redacted,
        )

        # Parse reward from tool result string
        reward = 0.0
        if isinstance(tool_result, str) and "reward=" in tool_result:
            try:
                reward = float(tool_result.split("reward=")[1])
            except (ValueError, IndexError):
                pass

        steps_taken = 1
        rewards.append(reward)
        done = True
        error = None

        log_step(step=1, action=action_str, reward=reward,
                 done=done, error=error)

        score = reward  # single-step episode, score = reward
        success = score >= 0.5

    except Exception as exc:
        log_step(step=steps_taken + 1, action="error", reward=0.0,
                 done=True, error=str(exc))
        rewards.append(0.0)
        steps_taken += 1

    finally:
        try:
            await env.close()
        except Exception:
            pass
        
        # Enforce hackathon rule strictly
        score = max(0.01, min(score, 0.99))
        rewards = [max(0.01, min(r, 0.99)) for r in rewards]

        log_end(success=success, steps=steps_taken, score=score,
                rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

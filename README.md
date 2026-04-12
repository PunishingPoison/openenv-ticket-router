---
title: OpenEnv Ticket Router
emoji: 🎟️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Ticket Router — L1 Customer Support Triage Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates
L1 customer-support email triage. An LLM agent receives raw support emails and
must categorise, extract structured information, or redact PII depending on the
task difficulty.

## Why This Environment?

Real-world support triage is deceptively subtle: emails are ambiguous, contain
mixed signals, and carry sensitive data that must be handled correctly.  This
environment captures three progressively harder facets of that problem so that
RL or tool-calling agents can be trained and benchmarked end-to-end.

## Tasks

| ID | Difficulty | Objective | Scoring |
|----|-----------|-----------|---------|
| `triage_easy_001` | **Easy** | Route email to the correct department | 0.99 correct · 0.01 wrong |
| `triage_medium_001` | **Medium** | Route **and** extract the technical error code | 0.5 routing + 0.5 extraction |
| `triage_hard_001` | **Hard** | Route **and** redact all PII from the email body | Multi-axis partial credit (see below) |

### PII Redaction Scoring (Hard)

| Component | Weight | Description |
|-----------|--------|-------------|
| Correct department | 0.25 | Must route to the right department |
| `[REDACTED]` placeholder present | 0.25 | Agent attempted redaction |
| PII patterns removed | 0.50 | Proportional to fraction of PII successfully redacted |

All rewards are clamped to `(0.01, 0.99)` to satisfy the hackathon's
strictly-between-0-and-1 constraint.

## Observation Space

On `reset(difficulty="easy")` the environment returns a `TicketObservation`
Pydantic model with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | Current difficulty: `"easy"`, `"medium"`, or `"hard"` |
| `email_subject` | `str` | Subject line of the support email |
| `email_body` | `str` | Full body text of the support email |
| `instructions` | `str` | Human-readable instructions for the agent |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Current reward (0.0 on reset) |

### Example observation (JSON)

```json
{
  "task": "easy",
  "email_subject": "Enterprise plan upgrade inquiry",
  "email_body": "Hi, I'm interested in upgrading our team to the Enterprise plan...",
  "instructions": "Route this email to the correct department. Choose one of: Sales, Billing, Tech Support.",
  "done": false,
  "reward": 0.0
}
```

## Action Space

The agent interacts via the `submit_answer` MCP tool:

| Argument | Type | When Required | Description |
|----------|------|---------------|-------------|
| `department` | `str` | **Always** | One of `Sales`, `Billing`, or `Tech Support` |
| `error_code` | `str` | Medium only | The extracted error code, e.g. `ERR-413` |
| `redacted_body` | `str` | Hard only | Full email body with all PII replaced by `[REDACTED]` |

## Quick Start

### Local (no Docker)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t ticket-router .
docker run -p 7860:7860 ticket-router
```

### Run Inference

```bash
export HF_TOKEN="hf_..."
export TICKET_ROUTER_URL="http://localhost:7860"
python inference.py
```

The inference script loops over all three difficulties (`easy`, `medium`,
`hard`) and logs structured `[START]` / `[STEP]` / `[END]` output for each
episode.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK_NAME` | `easy` | Difficulty level injected by the evaluator |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `LOCAL_IMAGE_NAME` | — | Docker image (if using `from_docker_image`) |

## Baseline Scores

| Difficulty | Expected Score |
|------------|---------------|
| Easy | 0.99 |
| Medium | 0.99 |
| Hard | 0.75 – 0.99 |

## Project Structure

```
├── openenv.yaml                        # OpenEnv manifest (tasks + rubrics)
├── pyproject.toml                      # Python package configuration
├── Dockerfile                          # Container image definition
├── inference.py                        # Baseline inference script
├── client.py                           # TicketRouterEnv client (MCPToolClient)
├── __init__.py                         # Package exports
├── test_env.py                         # Smoke tests
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                          # FastAPI application entry point
    ├── rubrics.py                      # Grader implementations (Rubric subclasses)
    └── ticket_router_environment.py    # Core environment logic
```

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  Agent (LLM)                       │
│  Reads observation → Decides action → Calls tool   │
└──────────────────────┬─────────────────────────────┘
                       │  MCP tool call: submit_answer(...)
                       ▼
┌────────────────────────────────────────────────────┐
│            TicketRouterEnvironment                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  reset() │  │ submit_answer│  │  RubricDict  │ │
│  │ selects  │  │  grades via  │  │  easy/medium │ │
│  │ ticket   │  │  rubric      │  │  /hard       │ │
│  └──────────┘  └──────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────┘
```

## License

MIT

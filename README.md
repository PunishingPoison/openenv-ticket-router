---
title: OpenEnv Ticket Router
emoji: 🎟️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Ticket Router — L1 Customer Support Ticket Router

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates
L1 customer support email triage. An LLM agent receives raw support emails and
must categorize, extract information, or redact PII depending on the task.

## Tasks

| Task | Difficulty | Objective | Scoring |
|------|-----------|-----------|---------|
| `basic_routing` | Easy | Route email to correct department | 1.0 correct, 0.0 wrong |
| `extraction_routing` | Medium | Route to Tech Support + extract error code | 0.5 per correct field |
| `pii_redaction` | Hard | Route to Billing + redact all PII from body | Partial credit (see below) |

### PII Redaction Scoring (Hard)

| Component | Weight |
|-----------|--------|
| Correct department ("Billing") | 0.25 |
| `[REDACTED]` placeholder present | 0.25 |
| Each PII pattern removed | 0.50 (proportional) |

## Action / Observation Space

### Observation (returned on `reset()`)

```json
{
  "task": "basic_routing",
  "email_subject": "Enterprise plan upgrade inquiry",
  "email_body": "Hi, I'm interested in upgrading...",
  "instructions": "Route this email to the correct department..."
}
```

### Action (via `submit_answer` MCP tool)

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `department` | `str` | Yes | `Sales`, `Billing`, or `Tech Support` |
| `error_code` | `str` | Medium task | e.g. `ERR-404` |
| `redacted_body` | `str` | Hard task | Email body with PII → `[REDACTED]` |

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
export TICKET_ROUTER_TASK="basic_routing"   # or extraction_routing / pii_redaction
export TICKET_ROUTER_URL="http://localhost:7860"
python inference.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKET_ROUTER_TASK` | `basic_routing` | Which task to run |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `HF_TOKEN` | — | API key |
| `IMAGE_NAME` | — | Docker image (if using `from_docker_image`) |

## Baseline Scores

| Task | Expected Score |
|------|---------------|
| `basic_routing` | 1.00 |
| `extraction_routing` | 1.00 |
| `pii_redaction` | 0.75 – 1.00 |

## Project Structure

```
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml          # Python package config
├── Dockerfile              # Container image
├── inference.py            # Inference script
├── __init__.py             # Package exports
├── client.py               # TicketRouterEnv client
├── README.md
└── server/
    ├── app.py              # FastAPI application
    └── ticket_router_environment.py  # Environment logic
```

## License

MIT

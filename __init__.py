"""
Ticket Router — L1 Customer Support Triage OpenEnv Environment.

An MCP-based environment where an LLM agent triages customer support emails
by routing them to the correct department, extracting error codes, or
redacting personally identifiable information (PII).

Tasks:
    - easy:   Route email to the correct department (Sales / Billing / Tech Support)
    - medium: Route email + extract the technical error code (e.g. ERR-413)
    - hard:   Route email + redact all PII from the email body

Example:
    >>> from ticket_router import TicketRouterEnv
    >>>
    >>> with TicketRouterEnv(base_url="http://localhost:7860").sync() as env:
    ...     env.reset(difficulty="easy")
    ...     result = env.call_tool("submit_answer", department="Sales")
    ...     print(result)
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import TicketRouterEnv

__all__ = ["TicketRouterEnv", "CallToolAction", "ListToolsAction"]

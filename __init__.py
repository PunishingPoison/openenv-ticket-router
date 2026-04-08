"""
Ticket Router — L1 Customer Support Ticket Router OpenEnv Environment.

Tasks:
    - basic_routing: Route email to correct department (Easy)
    - extraction_routing: Route + extract error code (Medium)
    - pii_redaction: Route + redact sensitive data (Hard)

Example:
    >>> from ticket_router import TicketRouterEnv
    >>>
    >>> with TicketRouterEnv(base_url="http://localhost:7860").sync() as env:
    ...     env.reset()
    ...     result = env.call_tool("submit_answer", department="Sales")
    ...     print(result)
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import TicketRouterEnv

__all__ = ["TicketRouterEnv", "CallToolAction", "ListToolsAction"]

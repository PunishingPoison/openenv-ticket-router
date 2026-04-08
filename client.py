"""
Ticket Router Environment Client.

Connects to a running Ticket Router server via WebSocket.

Example:
    >>> from ticket_router import TicketRouterEnv
    >>>
    >>> with TicketRouterEnv(base_url="http://localhost:7860").sync() as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool(
    ...         "submit_answer", department="Sales"
    ...     )
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class TicketRouterEnv(MCPToolClient):
    """Client for the Ticket Router Environment."""

    pass

"""
FastAPI application for the Ticket Router Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .ticket_router_environment import TicketRouterEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.ticket_router_environment import TicketRouterEnvironment

app = create_app(
    TicketRouterEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="ticket_router",
)

from fastapi.responses import RedirectResponse
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


def main():
    """Entry point for direct execution."""
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

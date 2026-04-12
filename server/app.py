import os

try:
    from openenv.core.env_server.http_server import create_app
    from .ticket_router_environment import TicketRouterEnvironment, TicketAction, TicketObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.ticket_router_environment import TicketRouterEnvironment, TicketAction, TicketObservation

app = create_app(
    TicketRouterEnvironment,
    TicketAction,
    TicketObservation,
    env_name="ticket_router",
)

from fastapi.responses import RedirectResponse
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

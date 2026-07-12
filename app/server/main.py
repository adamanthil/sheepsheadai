import os

from server.app import create_app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    # Dev entrypoint (hot reload). Production runs uvicorn via the container
    # CMD with --workers 1: all game state is in-process, so exactly one
    # worker is a hard requirement, not a tuning choice.
    uvicorn.run(
        "server.app:create_app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "9000")),
        reload=True,
        factory=True,
    )

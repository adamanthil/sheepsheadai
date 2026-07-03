from server.app import create_app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:create_app", host="0.0.0.0", port=9000, reload=True, factory=True
    )

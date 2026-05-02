from server.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=9000, reload=True)

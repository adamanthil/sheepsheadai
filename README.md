# sheepsheadai
Deep learning AI for the Sheepshead cardgame.

# Running

Requires `uv`: https://docs.astral.sh/uv/getting-started/installation/

Install dependencies (optional)
```
uv sync
```

Running a game played by the agent:
```
uv run play.py
```

# Web UI and Multiplayer Tables

Install extra dependencies for web UI
```
uv sync --extra server
```

Running the web server backend:
```
./server/run_server.sh --model [path/to/model/file.pt]
```

Running the web server frontend:
```
npm run dev
```

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sheepshead_model_path: str
    sheepshead_model_label: str
    database_url: str
    sheepshead_cors_origins: str = ""
    # Deploy-time convention mask for TABLE agents ("", "c1", "c2", "c1c2"):
    # masks convention-violating defender leads; the policy still picks the
    # card within the convention. /analyze stays raw by design — the research
    # scanners measure the unwrapped policy. See
    # sheepshead/agent/convention_wrapper.py + Convention_Optimality notebook.
    sheepshead_convention_wrap: str = ""
    # Open tables one creator may hold at once (server.runtime.manager). The
    # IP allowance is higher: several people behind one home or office
    # network may each open a table.
    sheepshead_max_tables_per_player: int = 2
    sheepshead_max_tables_per_ip: int = 5
    # Seconds a human has to move before the AI moves for them
    # (server.runtime.turn_timer).
    sheepshead_turn_timeout_seconds: float = 20.0
    env: str = "development"
    log_format: str = "text"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    # The required fields are supplied by the environment and .env, which
    # pydantic-settings reads inside __init__; a type checker only sees a
    # dataclass-like constructor with three missing arguments.
    return Settings()  # pyright: ignore[reportCallIssue]

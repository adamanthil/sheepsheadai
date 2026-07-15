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
    env: str = "development"
    log_format: str = "text"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

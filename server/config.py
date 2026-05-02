from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sheepshead_model_path: str
    database_url: str
    sheepshead_cors_origins: str = ""
    env: str = "development"
    log_format: str = "text"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

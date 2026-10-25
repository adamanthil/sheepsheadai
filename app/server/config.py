from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
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
    # Deployment sampling for TABLE agents (sheepshead/agent/sampling.py):
    # per-head softmax temperatures (0 = argmax) and an optional logit band
    # in nats. Tuned per checkpoint (Training_Program_Redesign §4.5); the
    # five values are also the AI player's identity row (ai_player).
    sheepshead_temperature_pick: float = 0.0
    sheepshead_temperature_partner: float = 0.0
    sheepshead_temperature_bury: float = 0.0
    sheepshead_temperature_play: float = 0.0
    sheepshead_logit_band: float | None = None

    @field_validator("sheepshead_logit_band", mode="before")
    @classmethod
    def _empty_band_is_none(cls, v):
        # Compose passes the variable through unset as "", which means no band.
        return None if v in ("", None) else v

    # Open tables one creator may hold at once (server.runtime.manager). The
    # IP allowance is higher: several people behind one home or office
    # network may each open a table.
    sheepshead_max_tables_per_player: int = 2
    sheepshead_max_tables_per_ip: int = 5
    # Seconds a human has to move before the AI moves for them
    # (server.runtime.turn_timer).
    sheepshead_turn_timeout_seconds: float = 30.0
    # Transactional email via Resend (server.services.email). With no key
    # (development, tests) messages and their links are logged instead of
    # sent; production refuses to boot without one.
    resend_api_key: str = ""
    email_from: str = "Sheepshead <noreply@localhost>"
    # Origin of the web app, for links in emails (verify, password reset).
    public_base_url: str = "http://localhost:3000"
    env: str = "development"
    log_format: str = "text"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    # The required fields are supplied by the environment and .env, which
    # pydantic-settings reads inside __init__; a type checker only sees a
    # dataclass-like constructor with three missing arguments.
    return Settings()  # pyright: ignore[reportCallIssue]

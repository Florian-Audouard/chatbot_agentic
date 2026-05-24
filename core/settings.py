from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Optional so the server can start and still serve /v1/models even if key is missing.
    # /v1/chat/completions will return a 500 until configured.
    openrouter_api_key: Annotated[
        str | None, Field(default=None, alias="OPENROUTER_API_KEY")
    ]
    openrouter_model: Annotated[
        str, Field(default="openai/gpt-4o-mini", alias="OPENROUTER_MODEL")
    ]
    openrouter_temperature: Annotated[
        float, Field(default=0.2, alias="OPENROUTER_TEMPERATURE")
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()

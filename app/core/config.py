from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ClinicalOps AI"
    app_version: str = "0.2.0"

    # Local default.
    # Docker can override this with:
    # OLLAMA_BASE_URL=http://host.docker.internal:11434
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_base_url(cls, value: str) -> str:
        """Return a usable Ollama server URL without a trailing slash."""

        value = value.strip()

        if not value:
            raise ValueError(
                "OLLAMA_BASE_URL must not be empty"
            )

        if not value.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must include "
                "http:// or https://"
            )

        return value.rstrip("/")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
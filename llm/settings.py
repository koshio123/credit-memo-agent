from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """環境変数（と .env）から読むLLM関連の設定。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: Literal["claude_code", "local", "anthropic_api"] = "local"
    llm_cache_enabled: bool = True
    llm_cache_dir: Path = Path(".cache/llm")

    ollama_base_url: str = "http://localhost:11434"
    local_model_fast: str = "qwen3:8b"
    local_model_standard: str = "qwen3:14b"
    local_model_strong: str = "qwen3:14b"

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL の接続先。環境変数か .env から読む。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://credit_memo:credit_memo@127.0.0.1:5433/credit_memo"

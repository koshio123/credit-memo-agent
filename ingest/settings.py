from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdinetSettings(BaseSettings):
    """EDINET API の設定。キーは環境変数か .env から読む（コミットしない）。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    edinet_api_key: SecretStr = SecretStr("")

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

# 役割ごとのモデル割当。呼び出し側は段階だけを指定し、実際のモデル名はバックエンドが決める。
#   fast: 分類・抽出などの軽い処理 / standard: 分析・起草 / strong: 計画・検証
Tier = Literal["fast", "standard", "strong"]


class LLMBackendError(Exception):
    """バックエンドの呼び出し失敗。リトライ可否は呼び出し側が判断する。"""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str = ""
    messages: list[Message] = Field(min_length=1)
    tier: Tier = "standard"
    max_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    backend: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False
    # max_tokens で途中で切れた応答。呼び出し側が検知して扱いを決める（キャッシュはしない）。
    truncated: bool = False


class LLMBackend(Protocol):
    """LLMの呼び出し先。実装は llm/local.py など。"""

    @property
    def name(self) -> str:
        """バックエンドの名前。

        Returns:
            キャッシュのキーや応答に記録する名前。
        """
        ...

    @property
    def cache_salt(self) -> str:
        """モデル名以外で出力に影響する設定（コンテキスト長など）。キャッシュのキーに混ぜる。

        Returns:
            設定を表す文字列。
        """
        ...

    def model_for(self, tier: Tier) -> str:
        """段階に対応する実モデル名。キャッシュのキーにも使う。

        Args:
            tier: モデルの段階（fast / standard / strong）。

        Returns:
            実際のモデル名。
        """
        ...

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """LLM を1回呼ぶ。

        Args:
            request: 呼び出しの内容。

        Returns:
            LLM の応答。
        """
        ...

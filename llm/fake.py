from collections import deque
from collections.abc import Iterable

from llm.types import LLMBackendError, LLMRequest, LLMResponse, Tier

_DEFAULT_MODELS: dict[Tier, str] = {"fast": "fake", "standard": "fake", "strong": "fake"}


class ScriptedBackend:
    """テスト用。あらかじめ渡した応答を順に返し、受け取ったリクエストを記録する。

    文字列は応答の本文として、LLMResponse はそのまま返し、例外は送出する。
    """

    name = "scripted"

    def __init__(
        self,
        script: Iterable[str | LLMResponse | LLMBackendError],
        models: dict[Tier, str] | None = None,
        salt: str = "",
    ) -> None:
        """あらかじめ決めた応答を返すバックエンドを作る。

        Args:
            script: 順に返す応答。文字列は本文、LLMResponse はそのまま、例外は送出する。
            models: 段階ごとのモデル名。省略した段階は "fake"。
            salt: キャッシュのキーに混ぜる文字列。
        """
        self.cache_salt = salt
        self._script = deque(script)
        self._models = {**_DEFAULT_MODELS, **(models or {})}
        self.requests: list[LLMRequest] = []

    def model_for(self, tier: Tier) -> str:
        """段階に対応する実モデル名。

        Args:
            tier: モデルの段階（fast / standard / strong）。

        Returns:
            実際のモデル名。
        """
        return self._models[tier]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """リクエストを記録し、次の応答を返す。

        Args:
            request: 呼び出しの内容。

        Returns:
            LLM の応答。

        Raises:
            AssertionError: 用意した応答を使い切ったとき。
            LLMBackendError: 次の項目が例外のとき（その例外）。
        """
        self.requests.append(request)
        if not self._script:
            raise AssertionError("ScriptedBackend: 用意した応答を使い切りました")
        item = self._script.popleft()
        if isinstance(item, LLMBackendError):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, backend=self.name, model=self.model_for(request.tier))

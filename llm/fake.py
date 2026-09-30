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
    ) -> None:
        self._script = deque(script)
        self._models = {**_DEFAULT_MODELS, **(models or {})}
        self.requests: list[LLMRequest] = []

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError("ScriptedBackend: 用意した応答を使い切りました")
        item = self._script.popleft()
        if isinstance(item, LLMBackendError):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, backend=self.name, model=self.model_for(request.tier))

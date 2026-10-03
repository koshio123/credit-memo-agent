"""LLM の呼び出し回数とトークン数を数える包み。アブレーション（構成ごとの費用）の比較に使う。

Pro の利用枠が制約なので、費用は金額ではなく、呼び出し回数とトークン数で示す。
"""

from llm.types import LLMBackend, LLMRequest, LLMResponse, Tier


class CountingBackend:
    def __init__(self, inner: LLMBackend) -> None:
        """数える対象のバックエンドを受け取る。

        Args:
            inner: 実際に呼び出すバックエンド。
        """
        self._inner = inner
        self.calls = 0
        self.cached_calls = 0  # キャッシュから返った呼び出し（利用枠を使わない）
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def name(self) -> str:
        """バックエンドの名前。

        Returns:
            キャッシュのキーや応答に記録する名前。
        """
        return self._inner.name

    @property
    def cache_salt(self) -> str:
        """モデル名以外で出力に影響する設定。キャッシュのキーに混ぜる。

        Returns:
            設定を表す文字列。
        """
        return self._inner.cache_salt

    def model_for(self, tier: Tier) -> str:
        """段階に対応する実モデル名。

        Args:
            tier: モデルの段階（fast / standard / strong）。

        Returns:
            実際のモデル名。
        """
        return self._inner.model_for(tier)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """内側のバックエンドを呼び、回数とトークン数を数える。キャッシュから返った分のトークンは数えない。

        Args:
            request: 呼び出しの内容。

        Returns:
            LLM の応答。
        """
        response = await self._inner.complete(request)
        self.calls += 1
        if response.cached:
            self.cached_calls += 1
        else:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        return response

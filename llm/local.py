import httpx
from pydantic import BaseModel, ValidationError

from llm.types import LLMBackendError, LLMRequest, LLMResponse, Tier, Usage


class _ChatMessage(BaseModel):
    content: str


class _ChatResponse(BaseModel):
    """Ollama の /api/chat の応答のうち、使う部分だけ。"""

    message: _ChatMessage
    model: str | None = None
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str | None = None  # "length" は max_tokens での打ち切り


class OllamaBackend:
    """ローカルLLM（Ollama）。開発中の反復と大量の評価実行に使う。"""

    name = "local"

    def __init__(
        self,
        base_url: str,
        models: dict[Tier, str],
        num_ctx: int = 16384,
        think: bool = False,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
    ) -> None:
        """
        num_ctx: コンテキスト長。Ollama の既定は短く、超えた分のプロンプトは黙って切り捨てられる。
        think: Qwen3 などの思考モード。思考トークンも max_tokens を消費するので、既定はオフ。
        """
        self._base_url = base_url.rstrip("/")
        self._models = models
        self.num_ctx = num_ctx
        self.think = think
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def cache_salt(self) -> str:
        return f"num_ctx={self.num_ctx};think={self.think}"

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = self.model_for(request.tier)
        messages = [m.model_dump() for m in request.messages]
        if request.system:
            messages = [{"role": "system", "content": request.system}, *messages]
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            res = await self._client.post(f"{self._base_url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMBackendError(f"Ollama に接続できません ({self._base_url}): {e}") from e
        if res.status_code != 200:
            raise LLMBackendError(
                f"Ollama がエラーを返しました: {res.status_code} {res.text[:200]}"
            )

        try:
            parsed = _ChatResponse.model_validate_json(res.content)
        except ValidationError as e:
            raise LLMBackendError(f"Ollama の応答を解釈できません: {e}") from e
        if parsed.prompt_eval_count >= self.num_ctx:
            # 入力が上限に達している。先頭が切り捨てられた状態で、もっともらしい答えが返っている。
            # あくまで補助的な検知: Ollama がプロンプトのキャッシュを再利用すると、この値は
            # 実際より小さく出て見逃す。確実に防ぐには、呼び出し側で入力量を制御する。
            raise LLMBackendError(
                f"プロンプトが num_ctx={self.num_ctx} に達しました。入力が切り捨てられた可能性が"
                "あるため応答を採用しません。num_ctx を上げるか、入力を減らしてください"
            )
        return LLMResponse(
            text=parsed.message.content,
            backend=self.name,
            model=parsed.model or model,
            usage=Usage(input_tokens=parsed.prompt_eval_count, output_tokens=parsed.eval_count),
            truncated=parsed.done_reason == "length",
        )

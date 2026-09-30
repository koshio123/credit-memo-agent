import math

import httpx
from pydantic import BaseModel, ValidationError

from llm.types import LLMBackendError, LLMRequest, LLMResponse, Tier, Usage

# 入力の大きさは送る前に見積もる。Ollama は num_ctx に収まらない入力を黙って切り詰め、
# 応答の prompt_eval_count も切り詰め後の値を返すため、応答からは切り捨てを確実に検知できない
# （実機で確認: 4426トークンの入力が、警告ログだけで258トークンとして「成功」した）。
# Qwen3 のトークナイザでの実測（1文字あたり）: 日本語の散文 0.69 / 数値の表 0.89 /
# 有報風の文章 0.95。1.0 を上限として見積もれば、これらを過小評価しない。
# 英数字が多い入力は過大評価になる（安全側）。
TOKENS_PER_CHAR_UPPER_BOUND = 1.0
# チャットのテンプレート（役割タグなど）の分の余裕
PROMPT_OVERHEAD_TOKENS = 64


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

    def _check_fits(self, request: LLMRequest) -> None:
        chars = len(request.system) + sum(len(m.content) for m in request.messages)
        estimated = math.ceil(chars * TOKENS_PER_CHAR_UPPER_BOUND) + PROMPT_OVERHEAD_TOKENS
        if estimated + request.max_tokens > self.num_ctx:
            raise LLMBackendError(
                f"入力が num_ctx={self.num_ctx} に収まらない可能性があります"
                f"（入力の見積もり {estimated} + max_tokens {request.max_tokens}）。"
                "Ollama は収まらない入力を黙って切り捨てるため、送信を中止しました。"
                "num_ctx を上げるか、入力を減らしてください"
            )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_fits(request)
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
        return LLMResponse(
            text=parsed.message.content,
            backend=self.name,
            model=parsed.model or model,
            usage=Usage(input_tokens=parsed.prompt_eval_count, output_tokens=parsed.eval_count),
            truncated=parsed.done_reason == "length",
        )

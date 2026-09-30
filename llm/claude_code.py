"""Claude Code（Agent SDK）経由のバックエンド。自分の Claude サブスクリプションのログインで動く。

規約上の位置づけ: Pro/Max の利用枠は「Claude Code と Agent SDK の通常の個人利用」を前提とする。
そのため、少量・手動の実行（最終メモの生成、検証、少数の judge）にだけ使う。CI・常時稼働・
他人への提供には使わない。認証情報は一切扱わず、Claude Code 自身のログインに任せる。
詳細は docs/decisions.md を参照。
"""

import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    Message,
    RateLimitEvent,
    ResultMessage,
    SystemMessage,
    query,
)

from llm.types import LLMBackendError, LLMRequest, LLMResponse, Tier, Usage

logger = logging.getLogger(__name__)

# claude_agent_sdk.query と同じ形。テストで差し替える
QueryFn = Callable[[str, ClaudeAgentOptions], AsyncIterator[Message]]

# これらが環境にあると Claude Code はサブスクリプションではなく API 課金で動いてしまう
_BILLING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

# system が空のときに Claude Code の既定（コーディング用）プロンプトへ落ちないようにする
_DEFAULT_SYSTEM = "あなたは丁寧で正確なアシスタントです。"


def _default_query(prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Message]:
    return query(prompt=prompt, options=options)


class ClaudeCodeBackend:
    """Agent SDK を「ツールなしの単発テキスト生成」として使う。

    できないこと（Agent SDK に該当オプションがない）:
    - temperature の指定。0 以外はエラーにする。
    - 複数ターンの履歴の受け渡し。単発の呼び出しだけ対応する。
    max_tokens は環境変数で上限にできるが、超えると途中で切れた応答ではなくエラーで返る。
    """

    name = "claude_code"

    def __init__(
        self,
        models: dict[Tier, str],
        think: bool = False,
        query_fn: QueryFn | None = None,
    ) -> None:
        for var in _BILLING_ENV_VARS:
            if os.environ.get(var):
                raise LLMBackendError(
                    f"環境変数 {var} が設定されています。この状態で Claude Code を呼ぶと"
                    "サブスクリプションではなく API 課金になるため、作成を中止しました。"
                    "unset してから実行してください"
                )
        self._models = models
        self._think = think
        self._query = query_fn or _default_query

    @property
    def cache_salt(self) -> str:
        return f"think={self._think}"

    def model_for(self, tier: Tier) -> str:
        return self._models[tier]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.temperature > 0:
            raise LLMBackendError(
                "claude_code バックエンドは temperature を指定できません（0 のみ）。"
                "Agent SDK に該当オプションがないためです"
            )
        if len(request.messages) != 1:
            raise LLMBackendError("claude_code バックエンドは複数ターンの履歴に未対応です")

        model = self.model_for(request.tier)
        # 作業ディレクトリをリポジトリの外にして、CLAUDE.md や .claude/ の hook を拾わせない
        with tempfile.TemporaryDirectory(prefix="credit-memo-llm-") as workdir:
            options = ClaudeAgentOptions(
                system_prompt=request.system or _DEFAULT_SYSTEM,
                model=model,
                tools=[],  # 組み込みツールは使わない
                setting_sources=[],  # ユーザー設定・CLAUDE.md を読み込まない
                max_turns=1,
                cwd=workdir,
                # アカウントに連携した claude.ai のコネクタ（Drive など）を生成に混ぜない
                settings=json.dumps({"disableClaudeAiConnectors": True}),
                thinking={"type": "adaptive"} if self._think else {"type": "disabled"},
                env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(request.max_tokens)},
            )
            return await self._run(request.messages[0].content, options, model)

    async def _run(self, prompt: str, options: ClaudeAgentOptions, model: str) -> LLMResponse:
        served_model = model
        result: ResultMessage | None = None
        try:
            # 結果メッセージの後にプロセスの異常終了が来ることがあるので、最後まで読み切る
            async for message in self._query(prompt, options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    self._check_auth(message.data)
                elif isinstance(message, RateLimitEvent):
                    self._check_rate_limit(message)
                elif isinstance(message, AssistantMessage):
                    served_model = message.model
                elif isinstance(message, ResultMessage):
                    result = message
        except ClaudeSDKError as e:
            raise LLMBackendError(f"Claude Code の呼び出しに失敗しました: {e}") from e

        if result is None:
            raise LLMBackendError("Claude Code から結果が返りませんでした")
        return self._to_response(result, served_model)

    @staticmethod
    def _check_auth(init: dict[str, Any]) -> None:
        source = init.get("apiKeySource")
        if source != "none":
            raise LLMBackendError(
                f"サブスクリプションのログインではない認証（apiKeySource={source!r}）で動いています。"
                "API 課金を避けるため中止しました"
            )

    @staticmethod
    def _check_rate_limit(event: RateLimitEvent) -> None:
        info = event.rate_limit_info
        if info.status == "rejected":
            raise LLMBackendError(
                f"Claude の利用枠の上限に達しています（{info.rate_limit_type}、"
                f"リセット: {info.resets_at}）。リセットまで待ってください"
            )
        if info.status == "allowed_warning":
            logger.warning(
                "Claude の利用枠が上限に近づいています（%s、utilization=%s）",
                info.rate_limit_type,
                info.utilization,
            )

    @staticmethod
    def _to_response(result: ResultMessage, model: str) -> LLMResponse:
        text = result.result or ""
        if result.is_error:
            raise LLMBackendError(
                f"Claude Code がエラーを返しました: {result.subtype} {result.errors}"
            )
        if text.startswith("API Error:"):
            # 実機では max_tokens 超過が、成功の結果の本文として "API Error: ..." で返ってきた
            raise LLMBackendError(f"Claude Code が API エラーを返しました: {text[:200]}")

        usage = result.usage or {}
        prompt_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        return LLMResponse(
            text=text,
            backend=ClaudeCodeBackend.name,
            model=model,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=usage.get("output_tokens", 0)),
            truncated=result.stop_reason == "max_tokens",
        )

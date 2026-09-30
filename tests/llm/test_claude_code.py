from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ProcessError,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from llm.claude_code import ClaudeCodeBackend
from llm.types import LLMBackendError, LLMRequest, Tier
from llm.types import Message as LLMMessage

pytestmark = pytest.mark.anyio

MODELS: dict[Tier, str] = {"fast": "haiku-x", "standard": "sonnet-x", "strong": "opus-x"}


def _init(api_key_source: str = "none", model: str = "sonnet-x") -> SystemMessage:
    return SystemMessage(
        subtype="init", data={"apiKeySource": api_key_source, "model": model, "tools": []}
    )


def _result(
    text: str = "自己資本比率は45.3%です。",
    *,
    is_error: bool = False,
    subtype: str = "success",
    stop_reason: str | None = "end_turn",
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s",
        stop_reason=stop_reason,
        usage={
            "input_tokens": 10,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 1000,
            "output_tokens": 7,
        },
        result=text,
    )


def _assistant(
    text: str = "自己資本比率は45.3%です。", model: str = "sonnet-x-20260101"
) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model=model)


def _rate_limit(status: str) -> RateLimitEvent:
    info = RateLimitInfo(
        status=status,  # type: ignore[arg-type]
        resets_at=1790755800,
        rate_limit_type="five_hour",
        utilization=None,
        overage_status=None,
        overage_resets_at=None,
        overage_disabled_reason=None,
        raw={},
    )
    return RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s")


class FakeQuery:
    """claude_agent_sdk.query の代わり。用意したメッセージを返し、渡された引数を記録する。"""

    def __init__(self, messages: list[Message], then_raise: Exception | None = None) -> None:
        self.messages = messages
        self.then_raise = then_raise
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(self, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Message]:
        for m in self.messages:
            yield m
        if self.then_raise is not None:
            raise self.then_raise


def _backend(fake: FakeQuery, **kwargs: Any) -> ClaudeCodeBackend:
    return ClaudeCodeBackend(models=MODELS, query_fn=fake, **kwargs)


def _req(**kwargs: Any) -> LLMRequest:
    return LLMRequest(
        system=kwargs.pop("system", "あなたは与信担当者です。"),
        messages=[LLMMessage(role="user", content="要約して")],
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


async def test_成功した応答を読む() -> None:
    fake = FakeQuery([_init(), _assistant(), _result()])

    res = await _backend(fake).complete(_req(tier="standard"))

    assert res.text == "自己資本比率は45.3%です。"
    assert (res.backend, res.model) == ("claude_code", "sonnet-x-20260101")
    # 入力トークンはキャッシュ作成・読み出し分を含めたプロンプト全体
    assert (res.usage.input_tokens, res.usage.output_tokens) == (1110, 7)
    assert (res.cached, res.truncated) == (False, False)


async def test_ツールも設定もコネクタも使わない安全な設定で呼ぶ(tmp_path: Path) -> None:
    fake = FakeQuery([_init(), _assistant(), _result()])

    await _backend(fake).complete(_req(tier="strong", max_tokens=300))

    prompt, opts = fake.calls[0]
    assert prompt == "要約して"
    assert opts.system_prompt == "あなたは与信担当者です。"
    assert opts.model == "opus-x"
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.max_turns == 1
    assert opts.thinking == {"type": "disabled"}
    assert opts.settings is not None and '"disableClaudeAiConnectors": true' in opts.settings
    assert opts.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "300"
    # このリポジトリの CLAUDE.md や hook を拾わないよう、作業ディレクトリはリポジトリの外
    assert opts.cwd is not None and Path(opts.cwd).resolve() != Path.cwd().resolve()


async def test_systemが空でもClaude_Codeの既定プロンプトに任せない() -> None:
    fake = FakeQuery([_init(), _assistant(), _result()])

    await _backend(fake).complete(_req(system=""))

    system_prompt = fake.calls[0][1].system_prompt
    assert isinstance(system_prompt, str) and system_prompt.strip()


async def test_temperatureは指定できないので明示的にエラーにする() -> None:
    fake = FakeQuery([_init(), _assistant(), _result()])

    with pytest.raises(LLMBackendError, match="temperature"):
        await _backend(fake).complete(_req(temperature=0.7))
    assert fake.calls == []


async def test_複数ターンの履歴は未対応としてエラーにする() -> None:
    fake = FakeQuery([_init(), _assistant(), _result()])
    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="a"),
            LLMMessage(role="assistant", content="b"),
            LLMMessage(role="user", content="c"),
        ]
    )

    with pytest.raises(LLMBackendError, match="複数ターン"):
        await _backend(fake).complete(request)


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_APIキーが環境にあると課金の取り違えを防ぐため作らせない(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "dummy")

    with pytest.raises(LLMBackendError, match=name):
        ClaudeCodeBackend(models=MODELS, query_fn=FakeQuery([]))


async def test_サブスクリプション以外の認証で動いていたらエラーにする() -> None:
    fake = FakeQuery([_init(api_key_source="ANTHROPIC_API_KEY"), _assistant(), _result()])

    with pytest.raises(LLMBackendError, match="サブスクリプション"):
        await _backend(fake).complete(_req())


async def test_エラー結果はLLMBackendErrorにする() -> None:
    fake = FakeQuery([_init(), _result("", is_error=True, subtype="error_during_execution")])

    with pytest.raises(LLMBackendError, match="error_during_execution"):
        await _backend(fake).complete(_req())


async def test_エラー文が本文として返ってきても回答として採用しない() -> None:
    # 実機では max_tokens 超過が「成功」の結果に "API Error: ..." という本文で返ってきた
    text = "API Error: Claude's response exceeded the 16 output token maximum."
    fake = FakeQuery([_init(), _assistant(text), _result(text, stop_reason="stop_sequence")])

    with pytest.raises(LLMBackendError, match="API Error"):
        await _backend(fake).complete(_req())


async def test_結果の後にプロセスが異常終了したら回答を採用しない() -> None:
    fake = FakeQuery([_init(), _assistant(), _result()], then_raise=ProcessError("exit 1"))

    with pytest.raises(LLMBackendError, match="Claude Code"):
        await _backend(fake).complete(_req())


async def test_利用枠の上限に達していたらエラーにする() -> None:
    fake = FakeQuery([_init(), _rate_limit("rejected")])

    with pytest.raises(LLMBackendError, match="利用枠"):
        await _backend(fake).complete(_req())


async def test_利用枠の警告は続行するが記録する(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeQuery([_init(), _rate_limit("allowed_warning"), _assistant(), _result()])

    with caplog.at_level("WARNING"):
        res = await _backend(fake).complete(_req())

    assert res.text
    assert "利用枠" in caplog.text


async def test_max_tokensで止まったらtruncatedを立てる() -> None:
    fake = FakeQuery([_init(), _assistant(), _result(stop_reason="max_tokens")])

    res = await _backend(fake).complete(_req())

    assert res.truncated is True


def test_cache_saltに思考の有無が入る() -> None:
    off = _backend(FakeQuery([]), think=False).cache_salt
    on = _backend(FakeQuery([]), think=True).cache_salt

    assert off != on
    assert _backend(FakeQuery([])).model_for("fast") == "haiku-x"

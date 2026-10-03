"""LLM の出力を JSON で受け、Pydantic で検証する。

失敗したら、理由を添えて作り直させる（既定は 1 回）。`claude_code` バックエンドは単発の呼び出しだけ
なので、作り直しは「元の依頼 + 直前の失敗の理由」を 1 通のメッセージにして送る。
途中で切れた出力は、作り直しても同じ長さで切れるため、作り直さずにエラーにする。
"""

import json

from pydantic import BaseModel, ValidationError

from llm.types import LLMBackend, LLMRequest, Message


class StructuredOutputError(Exception):
    """決まった形の出力を得られなかった。"""


def extract_json(text: str) -> str:
    """出力から、最初の JSON（オブジェクトか配列）を取り出す。フェンスや前後の説明は無視する。

    Args:
        text: LLM の出力。

    Returns:
        最初の JSON の文字列。

    Raises:
        ValueError: JSON が見つからないとき。
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return text[start:end]
    raise ValueError("出力に JSON が見つかりません")


def json_schema_hint(model: type[BaseModel]) -> str:
    """プロンプトに添える、出力の形の説明（JSON スキーマ）。

    Args:
        model: 出力の形を表す Pydantic モデル。

    Returns:
        整形した JSON スキーマの文字列。
    """
    return json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)


def _describe(error: ValidationError) -> str:
    """検証エラーを、場所と理由を並べた1行にする。

    Args:
        error: Pydantic の検証エラー。

    Returns:
        例: "claims.0.text: Field required"。
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )


def _retry_request(request: LLMRequest, error: str) -> LLMRequest:
    """元の依頼に、前回の失敗の理由を添えた作り直しの依頼を作る。

    Args:
        request: 元の依頼。
        error: 前回の出力を受け付けなかった理由。

    Returns:
        1通のメッセージにまとめた依頼。
    """
    original = request.messages[-1].content
    content = (
        f"{original}\n\n"
        f"【前回の出力は、次の理由で受け付けられませんでした】\n{error}\n"
        "指定された形の JSON だけを、説明やコードフェンスなしで出力してください。"
    )
    return request.model_copy(update={"messages": [Message(role="user", content=content)]})


async def complete_structured[T: BaseModel](
    backend: LLMBackend, request: LLMRequest, model: type[T], retries: int = 1
) -> T:
    """LLM の出力を JSON で受け、モデルで検証する。失敗したら理由を添えて作り直させる。

    Args:
        backend: LLM のバックエンド。
        request: 呼び出しの内容。
        model: 出力の形を表す Pydantic モデル。
        retries: 作り直す回数の上限。

    Returns:
        検証を通ったモデルのインスタンス。

    Raises:
        StructuredOutputError: 出力が途中で切れたとき、または作り直しても形が合わなかったとき。
    """
    current = request
    errors: list[str] = []
    for _ in range(retries + 1):
        response = await backend.complete(current)
        if response.truncated:
            raise StructuredOutputError(
                "出力が max_tokens で途中で切れました（作り直しても同じ長さで切れます）"
            )
        try:
            return model.model_validate_json(extract_json(response.text))
        except ValidationError as e:
            error = f"JSON の内容が指定の形と合いません: {_describe(e)}"
        except ValueError as e:
            error = f"JSON として読めません: {e}"
        errors.append(error)
        current = _retry_request(request, error)
    raise StructuredOutputError("\n".join(errors))

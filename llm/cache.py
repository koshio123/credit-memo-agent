import contextlib
import hashlib
import json
import logging
import uuid
from pathlib import Path

from pydantic import ValidationError

from llm.types import LLMBackend, LLMRequest, LLMResponse, Tier

logger = logging.getLogger(__name__)


class CachedBackend:
    """入力が同じなら、LLMを呼ばずに保存済みの応答を返すバックエンド。

    評価の再実行を無料にするためのもの。キーにはバックエンド名とモデル名も含めるので、
    モデルの割当を変えたときに古い応答を使い回すことはない。
    温度が0より大きい呼び出しも、最初に得た1つの応答を再現する（再現性を優先する）。

    次のものは保存しない。保存すると、以後の同じ呼び出しがずっと同じ不完全な応答を返すため。
    - 失敗した呼び出し
    - max_tokens で途中で切れた応答（truncated）
    - 空の応答

    キャッシュの読み書きに失敗しても、LLMの呼び出し自体は成功として扱う。
    """

    def __init__(self, inner: LLMBackend, cache_dir: Path) -> None:
        self.inner = inner
        self._dir = cache_dir

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def cache_salt(self) -> str:
        return self.inner.cache_salt

    def model_for(self, tier: Tier) -> str:
        return self.inner.model_for(tier)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        path = self._dir / f"{self._key(request)}.json"
        hit = self._read(path)
        if hit is not None:
            return hit.model_copy(update={"cached": True})

        response = await self.inner.complete(request)
        if self._is_cacheable(response):
            self._write(path, response)
        return response

    def _key(self, request: LLMRequest) -> str:
        payload = {
            "backend": self.inner.name,
            "model": self.inner.model_for(request.tier),
            "salt": self.inner.cache_salt,
            "request": request.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_cacheable(response: LLMResponse) -> bool:
        return bool(response.text.strip()) and not response.truncated

    @staticmethod
    def _read(path: Path) -> LLMResponse | None:
        if not path.exists():
            return None
        try:
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError, ValueError, OSError:
            # UnicodeDecodeError は ValueError の一種
            logger.warning("壊れたキャッシュを無視します: %s", path)
            return None

    @staticmethod
    def _write(path: Path, response: LLMResponse) -> None:
        # 書きかけのファイルを読まれないよう、他の書き手と重ならない名前の一時ファイルに
        # 書いてから置き換える
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(response.model_dump_json(), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            logger.warning("キャッシュに書き込めませんでした: %s", path, exc_info=True)
            # 後始末の失敗（親がディレクトリでない場合など）で、成功した応答まで失わない
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

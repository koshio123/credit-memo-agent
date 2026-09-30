import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from llm.types import LLMBackend, LLMRequest, LLMResponse, Tier

logger = logging.getLogger(__name__)


class CachedBackend:
    """入力が同じなら、LLMを呼ばずに保存済みの応答を返すバックエンド。

    評価の再実行を無料にするためのもの。キーにはバックエンド名とモデル名も含めるので、
    モデルの割当を変えたときに古い応答を使い回すことはない。
    温度が0より大きい呼び出しも、最初に得た1つの応答を再現する（再現性を優先する）。
    失敗した呼び出しは保存しない。
    """

    def __init__(self, inner: LLMBackend, cache_dir: Path) -> None:
        self.inner = inner
        self._dir = cache_dir

    @property
    def name(self) -> str:
        return self.inner.name

    def model_for(self, tier: Tier) -> str:
        return self.inner.model_for(tier)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        path = self._dir / f"{self._key(request)}.json"
        hit = self._read(path)
        if hit is not None:
            return hit.model_copy(update={"cached": True})

        response = await self.inner.complete(request)
        self._write(path, response)
        return response

    def _key(self, request: LLMRequest) -> str:
        payload = {
            "backend": self.inner.name,
            "model": self.inner.model_for(request.tier),
            "request": request.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _read(path: Path) -> LLMResponse | None:
        if not path.exists():
            return None
        try:
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, OSError):
            logger.warning("壊れたキャッシュを無視します: %s", path)
            return None

    @staticmethod
    def _write(path: Path, response: LLMResponse) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 書きかけのファイルを読まれないよう、一時ファイルに書いてから置き換える
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json(), encoding="utf-8")
        tmp.replace(path)

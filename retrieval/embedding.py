"""文章の埋め込み（ベクトル化）。検索のベクトル側で使う。

- モデルごとに、クエリと文書に付ける接頭辞が違う。付け忘れると精度が落ちるので、モデルの仕様として
  ここに固定する（値はモデルカードで確認した）。
- 入力の上限を超える文書は、モデルが黙って切り捨てる。over_limit で検出できるようにする。
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    query_prefix: str
    document_prefix: str
    dim: int
    max_tokens: int  # 入力の上限。これを超えると、モデルは黙って切り捨てる


# 接頭辞・次元・最大長は、それぞれのモデルカードで確認した値
MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec(
        key="e5-small",
        hf_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        document_prefix="passage: ",
        dim=384,
        max_tokens=512,
    ),
    "ruri-v3-30m": ModelSpec(
        key="ruri-v3-30m",
        hf_id="cl-nagoya/ruri-v3-30m",
        query_prefix="検索クエリ: ",
        document_prefix="検索文書: ",
        dim=256,
        max_tokens=8192,
    ),
}


class Embedder(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        """文書のうち、モデルの入力上限を超えるもの（黙って切り捨てられる）。"""
        ...


class EncoderModel(Protocol):
    """SentenceTransformer と同じ形の、差し替え可能なモデル。"""

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...

    def count_tokens(self, texts: list[str]) -> list[int]: ...


class _SentenceTransformerModel:
    """実際の SentenceTransformer を、EncoderModel の形に合わせる。"""

    def __init__(self, hf_id: str, device: str | None = None) -> None:
        # PyTorch の読み込みは重いので、実際に使うときまで遅らせる
        from sentence_transformers import SentenceTransformer

        # encode の型が複雑で推論できないため、境界では Any として扱う
        self._model: Any = SentenceTransformer(hf_id, device=device)

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        return self._model.encode(texts, **kwargs)

    def count_tokens(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in cast(list[list[int]], encoded["input_ids"])]


def _to_lists(vectors: Any) -> list[list[float]]:
    return [[float(x) for x in v] for v in vectors]


class SentenceTransformerEmbedder:
    def __init__(
        self,
        spec: ModelSpec,
        model: EncoderModel | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self._spec = spec
        self._model = model or _SentenceTransformerModel(spec.hf_id, device)
        self._batch_size = batch_size

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def dim(self) -> int:
        return self._spec.dim

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return _to_lists(vectors)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self._spec.query_prefix)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self._spec.document_prefix)

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        if not documents:
            return []
        prefixed = [self._spec.document_prefix + d for d in documents]
        return [n > self._spec.max_tokens for n in self._model.count_tokens(prefixed)]


class HashEmbedder:
    """テスト用の決定的な埋め込み。文字の2-gramをハッシュして次元に振り分け、正規化する。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def key(self) -> str:
        return "hash"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for i in range(len(text) - 1):
            digest = hashlib.sha256(text[i : i + 2].encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        return [False for _ in documents]

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
    def key(self) -> str:
        """埋め込みモデルの識別子。

        Returns:
            DB に保存するモデルの名前。
        """
        ...

    @property
    def dim(self) -> int:
        """埋め込みベクトルの次元。

        Returns:
            ベクトルの要素数。
        """
        ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """質問文をベクトルにする。

        Args:
            texts: 質問文のリスト。

        Returns:
            入力と同じ順の、正規化済みのベクトルのリスト。
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """文書をベクトルにする。

        Args:
            texts: 文書の本文のリスト。

        Returns:
            入力と同じ順の、正規化済みのベクトルのリスト。
        """
        ...

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        """文書のうち、モデルの入力上限を超えるもの（黙って切り捨てられる）を調べる。

        Args:
            documents: 文書の本文のリスト。

        Returns:
            入力と同じ順の真偽値のリスト。上限を超える文書は True。
        """
        ...


class EncoderModel(Protocol):
    """SentenceTransformer と同じ形の、差し替え可能なモデル。"""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """文字列をベクトルにする。

        Args:
            texts: 入力の文字列のリスト。
            **kwargs: モデルにそのまま渡す引数。

        Returns:
            ベクトルの並び（モデルが返す配列のまま）。
        """
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """文字列ごとのトークン数を数える。

        Args:
            texts: 入力の文字列のリスト。

        Returns:
            入力と同じ順のトークン数。特殊トークンを含む。
        """
        ...


class _SentenceTransformerModel:
    """実際の SentenceTransformer を、EncoderModel の形に合わせる。"""

    def __init__(self, hf_id: str, device: str | None = None) -> None:
        # PyTorch の読み込みは重いので、実際に使うときまで遅らせる
        """SentenceTransformer を読み込む。

        Args:
            hf_id: Hugging Face のモデルID。
            device: 実行する装置。None ならライブラリの既定。
        """
        from sentence_transformers import SentenceTransformer

        # encode の型が複雑で推論できないため、境界では Any として扱う
        self._model: Any = SentenceTransformer(hf_id, device=device)

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """文字列をベクトルにする。

        Args:
            texts: 入力の文字列のリスト。
            **kwargs: SentenceTransformer.encode にそのまま渡す引数。

        Returns:
            ベクトルの並び（モデルが返す配列のまま）。
        """
        return self._model.encode(texts, **kwargs)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """文字列ごとのトークン数を数える。

        Args:
            texts: 入力の文字列のリスト。

        Returns:
            入力と同じ順のトークン数。特殊トークンを含み、切り捨てはしない。
        """
        encoded = self._model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in cast(list[list[int]], encoded["input_ids"])]


def _to_lists(vectors: Any) -> list[list[float]]:
    """モデルが返す配列を、float の二重リストにする。

    Args:
        vectors: ベクトルの並び（numpy の配列など）。

    Returns:
        float のリストのリスト。
    """
    return [[float(x) for x in v] for v in vectors]


class SentenceTransformerEmbedder:
    def __init__(
        self,
        spec: ModelSpec,
        model: EncoderModel | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """埋め込みの器を作る。

        Args:
            spec: 使うモデルの仕様。
            model: 差し替え用のモデル。None なら実際の SentenceTransformer を読み込む。
            batch_size: 一度にベクトルにする件数。
            device: 実行する装置。None ならライブラリの既定。
        """
        self._spec = spec
        self._model = model or _SentenceTransformerModel(spec.hf_id, device)
        self._batch_size = batch_size

    @property
    def key(self) -> str:
        """埋め込みモデルの識別子。

        Returns:
            モデル仕様のキー。
        """
        return self._spec.key

    @property
    def dim(self) -> int:
        """埋め込みベクトルの次元。

        Returns:
            モデル仕様の次元。
        """
        return self._spec.dim

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        """接頭辞を付けて、正規化したベクトルにする。

        Args:
            texts: 入力の文字列。
            prefix: 各文字列の前に付ける接頭辞。

        Returns:
            入力と同じ順のベクトル。入力が空なら空のリスト。
        """
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
        """質問文をベクトルにする。

        Args:
            texts: 質問文のリスト。

        Returns:
            入力と同じ順の、正規化済みのベクトルのリスト。
        """
        return self._encode(texts, self._spec.query_prefix)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """文書をベクトルにする。

        Args:
            texts: 文書の本文のリスト。

        Returns:
            入力と同じ順の、正規化済みのベクトルのリスト。
        """
        return self._encode(texts, self._spec.document_prefix)

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        """文書のうち、モデルの入力上限を超えるものを調べる。

        Args:
            documents: 文書の本文のリスト。接頭辞を付けた状態で数える。

        Returns:
            入力と同じ順の真偽値のリスト。上限を超える文書は True。
        """
        if not documents:
            return []
        prefixed = [self._spec.document_prefix + d for d in documents]
        return [n > self._spec.max_tokens for n in self._model.count_tokens(prefixed)]


class HashEmbedder:
    """テスト用の決定的な埋め込み。文字の2-gramをハッシュして次元に振り分け、正規化する。"""

    def __init__(self, dim: int = 64) -> None:
        """テスト用の埋め込みを作る。

        Args:
            dim: ベクトルの次元。
        """
        self._dim = dim

    @property
    def key(self) -> str:
        """埋め込みモデルの識別子。

        Returns:
            固定の "hash"。
        """
        return "hash"

    @property
    def dim(self) -> int:
        """埋め込みベクトルの次元。

        Returns:
            作成時に指定した次元。
        """
        return self._dim

    def _embed(self, text: str) -> list[float]:
        """文字の2-gramをハッシュして次元に振り分け、正規化する。

        Args:
            text: 入力の文字列。

        Returns:
            長さ 1 に正規化したベクトル。2文字未満なら零ベクトル。
        """
        vector = [0.0] * self._dim
        for i in range(len(text) - 1):
            digest = hashlib.sha256(text[i : i + 2].encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """質問文をベクトルにする。

        Args:
            texts: 質問文のリスト。

        Returns:
            入力と同じ順のベクトルのリスト。
        """
        return [self._embed(t) for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """文書をベクトルにする。

        Args:
            texts: 文書の本文のリスト。

        Returns:
            入力と同じ順のベクトルのリスト。
        """
        return [self._embed(t) for t in texts]

    def over_limit(self, documents: Sequence[str]) -> list[bool]:
        """入力上限を超える文書を調べる。上限は無いので常に False。

        Args:
            documents: 文書の本文のリスト。

        Returns:
            入力と同じ長さの、すべて False のリスト。
        """
        return [False for _ in documents]

"""検索の入口: 全文（pg_bigm）・BM25・ベクトル（pgvector）と、語句側とベクトル側の順位融合（RRF）。

方式ごとにスコアの尺度が違うので、融合はスコアではなく順位で行う。
検索の前に、クエリを NFKC で正規化する（本文が NFKC で正規化されているため）。
"""

import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from retrieval.bm25 import Bm25Index
from retrieval.chunker import Chunk
from retrieval.embedding import Embedder
from retrieval.fusion import rrf
from retrieval.store import ChunkStore, Hit

# lexical = pg_bigm の類似度 / bm25 = 文字2-gram の BM25
# hybrid = lexical + vector / hybrid_bm25 = bm25 + vector（L2 評価で最良。既定）
Mode = Literal["lexical", "bm25", "vector", "hybrid", "hybrid_bm25"]

# 融合の前に、各方式から取る候補の数。k より多く取って、融合で上位を選ぶ
_CANDIDATES = 50


@dataclass(frozen=True)
class _Snapshot:
    index: Bm25Index
    chunks: dict[str, Chunk]
    by_doc: dict[str, list[str]]
    fingerprint: tuple[int, int]


class LexicalIndex:
    """全文書のチャンクから作る BM25 の索引（IDF は全文書で計算）。

    取り込みでチャンクが変わったら作り直す。複数の Retriever・スレッドで共有できる
    （作り直しは1つのスナップショットの入れ替えで行い、読み手は途中の状態を見ない）。
    """

    def __init__(self, store: ChunkStore) -> None:
        self._store = store
        self._snapshot: _Snapshot | None = None
        self._lock = threading.Lock()

    @property
    def index(self) -> Bm25Index:
        return self._current().index

    def search(self, query: str, k: int, doc_ids: Sequence[str] | None) -> list[Hit]:
        snap = self._current()
        restrict = None
        if doc_ids is not None:
            restrict = [cid for doc_id in set(doc_ids) for cid in snap.by_doc.get(doc_id, [])]
        ranked = snap.index.search(query, k, restrict)
        return [
            Hit(snap.chunks[chunk_id], score, rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def _current(self) -> _Snapshot:
        with self._lock:
            fingerprint = self._store.corpus_fingerprint()
            snap = self._snapshot
            if snap is not None and snap.fingerprint == fingerprint:
                return snap
            chunks = self._store.all_chunks()
            by_doc: dict[str, list[str]] = {}
            for c in chunks:
                by_doc.setdefault(c.doc_id, []).append(c.chunk_id)
            snap = _Snapshot(
                index=Bm25Index({c.chunk_id: c.text for c in chunks}),
                chunks={c.chunk_id: c for c in chunks},
                by_doc=by_doc,
                fingerprint=fingerprint,
            )
            self._snapshot = snap
            return snap


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        lexical_index: LexicalIndex | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lexical = lexical_index or LexicalIndex(store)

    def search(
        self,
        query: str,
        k: int,
        mode: Mode = "hybrid_bm25",
        doc_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        query = unicodedata.normalize("NFKC", query)
        if mode == "lexical":
            return self._store.search_lexical(query, k, doc_ids)
        if mode == "bm25":
            return self._lexical.search(query, k, doc_ids)
        if mode == "vector":
            return self._vector(query, k, doc_ids)
        if mode in ("hybrid", "hybrid_bm25"):
            n = max(k, _CANDIDATES)
            if mode == "hybrid":
                lexical = self._store.search_lexical(query, n, doc_ids)
            else:
                lexical = self._lexical.search(query, n, doc_ids)
            vector = self._vector(query, n, doc_ids)
            by_id = {h.chunk.chunk_id: h for h in [*lexical, *vector]}
            fused = rrf([[h.chunk.chunk_id for h in lexical], [h.chunk.chunk_id for h in vector]])
            return [
                Hit(by_id[chunk_id].chunk, score, rank)
                for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
            ]
        raise ValueError(f"知らない検索モードです: {mode}")

    def is_indexed(self, doc_id: str) -> bool:
        """この書類を、このインスタンスの埋め込みモデルで検索できるか（埋め込みが保存されているか）。"""
        return self._store.count_embeddings(self._embedder.key, doc_id) > 0

    def _vector(self, query: str, k: int, doc_ids: Sequence[str] | None) -> list[Hit]:
        (vector,) = self._embedder.embed_queries([query])
        return self._store.search_vector(self._embedder.key, vector, k, doc_ids)

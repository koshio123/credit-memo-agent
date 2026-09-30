"""検索の入口: 全文（pg_bigm）・BM25・ベクトル（pgvector）と、語句側とベクトル側の順位融合（RRF）。

方式ごとにスコアの尺度が違うので、融合はスコアではなく順位で行う。
検索の前に、クエリを NFKC で正規化する（本文が NFKC で正規化されているため）。
"""

import unicodedata
from collections.abc import Sequence
from typing import Literal

from retrieval.bm25 import Bm25Index
from retrieval.chunker import Chunk
from retrieval.embedding import Embedder
from retrieval.fusion import rrf
from retrieval.store import ChunkStore, Hit

# lexical = pg_bigm の類似度 / bm25 = 文字2-gram の BM25
# hybrid = lexical + vector / hybrid_bm25 = bm25 + vector
Mode = Literal["lexical", "bm25", "vector", "hybrid", "hybrid_bm25"]

# 融合の前に、各方式から取る候補の数。k より多く取って、融合で上位を選ぶ
_CANDIDATES = 50


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder
        self._bm25: Bm25Index | None = None
        self._bm25_chunks: dict[str, Chunk] = {}
        self._bm25_fingerprint: tuple[int, int] | None = None

    def search(
        self,
        query: str,
        k: int,
        mode: Mode = "hybrid",
        doc_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        query = unicodedata.normalize("NFKC", query)
        if mode == "lexical":
            return self._store.search_lexical(query, k, doc_ids)
        if mode == "bm25":
            return self._bm25_search(query, k, doc_ids)
        if mode == "vector":
            return self._vector(query, k, doc_ids)
        if mode in ("hybrid", "hybrid_bm25"):
            n = max(k, _CANDIDATES)
            if mode == "hybrid":
                lexical = self._store.search_lexical(query, n, doc_ids)
            else:
                lexical = self._bm25_search(query, n, doc_ids)
            vector = self._vector(query, n, doc_ids)
            by_id = {h.chunk.chunk_id: h for h in [*lexical, *vector]}
            fused = rrf([[h.chunk.chunk_id for h in lexical], [h.chunk.chunk_id for h in vector]])
            return [
                Hit(by_id[chunk_id].chunk, score, rank)
                for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
            ]
        raise ValueError(f"知らない検索モードです: {mode}")

    def _vector(self, query: str, k: int, doc_ids: Sequence[str] | None) -> list[Hit]:
        (vector,) = self._embedder.embed_queries([query])
        return self._store.search_vector(self._embedder.key, vector, k, doc_ids)

    def _bm25_search(self, query: str, k: int, doc_ids: Sequence[str] | None) -> list[Hit]:
        index = self._bm25_index()
        restrict = None
        if doc_ids is not None:
            wanted = set(doc_ids)
            restrict = [cid for cid, c in self._bm25_chunks.items() if c.doc_id in wanted]
        ranked = index.search(query, k, restrict)
        return [
            Hit(self._bm25_chunks[chunk_id], score, rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def _bm25_index(self) -> Bm25Index:
        """全文書のチャンクから索引を作る（IDF は全文書で計算）。取り込みで変わったら作り直す。"""
        fingerprint = self._store.corpus_fingerprint()
        if self._bm25 is None or fingerprint != self._bm25_fingerprint:
            chunks = self._store.all_chunks()
            self._bm25_chunks = {c.chunk_id: c for c in chunks}
            self._bm25 = Bm25Index({c.chunk_id: c.text for c in chunks})
            self._bm25_fingerprint = fingerprint
        return self._bm25

"""文書の取り込み: チャンク分け → PostgreSQL に保存 → 埋め込み。

埋め込みは CPU で時間がかかるので、内容が変わっていない文書は再実行でやり直さない。
モデルが黙って切り捨てる長すぎるチャンクは、数を報告する（チャンクの大きさを見直す材料）。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.chunker import DEFAULT_MAX_CHARS, chunk_pages
from retrieval.embedding import Embedder
from retrieval.store import ChunkStore, DocumentRecord


@dataclass(frozen=True)
class IngestResult:
    n_chunks: int
    embedded: dict[str, int]  # モデル -> 今回埋め込んだチャンク数（取り込み済みなら0）
    over_limit: dict[str, int]  # モデル -> 入力上限を超えて、黙って切り捨てられるチャンク数


def ingest_document(
    store: ChunkStore,
    doc: DocumentRecord,
    pages: list[str],
    embedders: Sequence[Embedder],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IngestResult:
    chunks = chunk_pages(doc.doc_id, pages, max_chars=max_chars)
    unchanged = store.chunks_unchanged(doc.doc_id, chunks)
    if not unchanged:
        # 古いチャンクと、その埋め込みは、置き換えで消える
        store.upsert_document(doc, chunks)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    embedded: dict[str, int] = {}
    over_limit: dict[str, int] = {}
    for embedder in embedders:
        over_limit[embedder.key] = sum(embedder.over_limit(texts))
        if chunks and store.count_embeddings(embedder.key, doc.doc_id) == len(chunks):
            embedded[embedder.key] = 0
            continue
        vectors = embedder.embed_documents(texts)
        store.add_embeddings(embedder.key, ids, vectors)
        embedded[embedder.key] = len(chunks)
    return IngestResult(len(chunks), embedded, over_limit)

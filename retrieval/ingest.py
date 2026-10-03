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
    over_limit: dict[str, int]  # 今回埋め込んだモデル -> 入力上限を超えて、黙って切り捨てられる数


def ingest_document(
    store: ChunkStore,
    doc: DocumentRecord,
    pages: list[str],
    embedders: Sequence[Embedder],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IngestResult:
    """文書をチャンクに分けて保存し、渡されたモデルで埋め込む。

    内容が変わっていなければ保存し直さず、埋め込み済みのモデルはやり直さない。

    Args:
        store: チャンクの保存先。
        doc: 文書の情報。
        pages: ページごとの本文。
        embedders: 埋め込みに使うモデル。置き換えで消える埋め込みのモデルをすべて含める。
        max_chars: 1チャンクの最大文字数。

    Returns:
        チャンク数と、モデルごとの埋め込み数・入力上限を超えた数。

    Raises:
        ValueError: 置き換えると消える埋め込みのモデルが embedders に無いとき、
            またはベクトルの次元が宣言と違うとき。
    """
    chunks = chunk_pages(doc.doc_id, pages, max_chars=max_chars)
    unchanged = store.get_document(doc.doc_id) == doc and store.chunks_unchanged(doc.doc_id, chunks)
    if not unchanged:
        # 置き換えると、古いチャンクと、全モデルの埋め込みが連鎖削除で消える。
        # 渡していないモデルの分は作り直せないので、消す前に断る
        missing = sorted(set(store.embedded_models(doc.doc_id)) - {e.key for e in embedders})
        if missing:
            raise ValueError(
                f"文書 {doc.doc_id} を置き換えると、次のモデルの埋め込みが消える: "
                f"{', '.join(missing)}。これらのモデルも渡すこと"
            )
        store.upsert_document(doc, chunks)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    embedded: dict[str, int] = {}
    over_limit: dict[str, int] = {}
    for embedder in embedders:
        if chunks and store.count_embeddings(embedder.key, doc.doc_id) == len(chunks):
            embedded[embedder.key] = 0
            continue
        vectors = embedder.embed_documents(texts)
        if any(len(v) != embedder.dim for v in vectors):
            raise ValueError(f"{embedder.key}: ベクトルの次元が宣言（{embedder.dim}）と違う")
        store.add_embeddings(embedder.key, ids, vectors)
        embedded[embedder.key] = len(chunks)
        over_limit[embedder.key] = sum(embedder.over_limit(texts))
    return IngestResult(len(chunks), embedded, over_limit)

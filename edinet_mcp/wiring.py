"""実データ（data/、PostgreSQL、埋め込みモデル）につないで、サービスを組み立てる。"""

import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from edinet_mcp.service import EdinetMcpError, EdinetService, Searcher
from evals.companies import load_dataset
from ingest.pdf_baseline import cached_pages
from ingest.xbrl_facts import read_facts
from retrieval.embedding import MODELS, SentenceTransformerEmbedder
from retrieval.search import Retriever
from retrieval.settings import DatabaseSettings
from retrieval.store import ChunkStore, Hit


class LazySearcher:
    """最初の検索まで、DB への接続と埋め込みモデルの読み込みを遅らせる。

    財務数値・比率・ページの取得は、DB もモデルも要らない。検索の準備に失敗しても
    サーバーは起動し、他のツールは使える。失敗は次の呼び出しでもう一度試す。
    """

    def __init__(self, factory: Callable[[], Searcher]) -> None:
        """最初に使われるまで、検索の準備を遅らせる。

        Args:
            factory: 検索を作る関数。DB への接続とモデルの読み込みを含むので、遅らせる。
        """
        self._factory = factory
        self._searcher: Searcher | None = None
        self._lock = threading.Lock()

    def _get(self) -> Searcher:
        """検索を作る（作成済みならそれを返す）。

        Returns:
            検索。

        Raises:
            EdinetMcpError: DB に接続できない、またはモデルを読み込めないとき。
        """
        with self._lock:
            if self._searcher is None:
                try:
                    self._searcher = self._factory()
                except (OSError, psycopg.Error) as e:
                    raise EdinetMcpError(
                        "検索を使えません（PostgreSQL に接続できないか、モデルを読み込めません）。"
                        f"docker compose up -d --wait で DB を起動してください: {type(e).__name__}"
                    ) from e
            return self._searcher

    def search(self, query: str, k: int, *, doc_ids: Sequence[str] | None = None) -> list[Hit]:
        """本文のチャンクを検索する。

        Args:
            query: 質問文。
            k: 返す件数の上限。
            doc_ids: 対象にする書類ID。None なら全書類。

        Returns:
            順位つきの検索結果。
        """
        return self._get().search(query, k, doc_ids=doc_ids)

    def is_indexed(self, doc_id: str) -> bool:
        """書類が検索の索引に入っているか。

        Args:
            doc_id: 書類ID。

        Returns:
            索引に入っていれば True。
        """
        return self._get().is_indexed(doc_id)


DEFAULT_MODEL = "ruri-v3-30m"  # L2 評価で、BM25 との融合の MRR が最も高かった（docs/decisions.md）


def build_service(
    data_dir: Path = Path("data"),
    database_url: str | None = None,
    model: str = DEFAULT_MODEL,
    datasets: tuple[str, ...] = ("dev", "heldout"),
) -> EdinetService:
    """実際のデータと DB につなげたサービス層を作る。

    Args:
        data_dir: 取得した書類とキャッシュのあるディレクトリ。
        database_url: PostgreSQL の接続URL。None なら設定から読む。
        model: 埋め込みモデルのキー。
        datasets: 対象にする会社の組（dev / heldout）。

    Returns:
        サービス層。
    """
    edinet_dir = data_dir / "edinet"
    text_cache = data_dir / "pdf_text"

    def make_retriever() -> Retriever:
        """DB に接続し、検索の入口を作る。

        Returns:
            検索の入口。
        """
        store = ChunkStore.connect(database_url or DatabaseSettings().database_url)
        return Retriever(store, SentenceTransformerEmbedder(MODELS[model]))

    return EdinetService(
        searcher=LazySearcher(make_retriever),
        companies=[c for name in datasets for c in load_dataset(name)],
        load_facts=lambda doc_id: read_facts(edinet_dir / doc_id / f"{doc_id}.csv.zip"),
        load_pages=lambda doc_id: cached_pages(edinet_dir / doc_id / f"{doc_id}.pdf", text_cache),
    )

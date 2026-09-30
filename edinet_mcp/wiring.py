"""実データ（data/、PostgreSQL、埋め込みモデル）につないで、サービスを組み立てる。"""

from pathlib import Path

from edinet_mcp.service import EdinetService
from evals.companies import load_dataset
from ingest.pdf_baseline import cached_pages
from ingest.xbrl_facts import read_facts
from retrieval.embedding import MODELS, SentenceTransformerEmbedder
from retrieval.search import Retriever
from retrieval.settings import DatabaseSettings
from retrieval.store import ChunkStore

DEFAULT_MODEL = "ruri-v3-30m"  # L2 評価で、BM25 との融合の MRR が最も高かった（docs/decisions.md）


def build_service(
    data_dir: Path = Path("data"),
    database_url: str | None = None,
    model: str = DEFAULT_MODEL,
    datasets: tuple[str, ...] = ("dev", "heldout"),
) -> EdinetService:
    edinet_dir = data_dir / "edinet"
    text_cache = data_dir / "pdf_text"
    store = ChunkStore.connect(database_url or DatabaseSettings().database_url)
    retriever = Retriever(store, SentenceTransformerEmbedder(MODELS[model]))
    return EdinetService(
        searcher=retriever,
        companies=[c for name in datasets for c in load_dataset(name)],
        load_facts=lambda doc_id: read_facts(edinet_dir / doc_id / f"{doc_id}.csv.zip"),
        load_pages=lambda doc_id: cached_pages(edinet_dir / doc_id / f"{doc_id}.pdf", text_cache),
    )

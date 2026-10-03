"""対象企業の有価証券報告書（当期）をチャンクに分け、PostgreSQL に保存して、埋め込みを作る。

  uv run python -m scripts.ingest_index [--dataset dev heldout] [--models e5-small ruri-v3-30m]

先に docker compose up -d --wait で DB を起動し、scripts.fetch_filings で書類を取得しておく。
内容が変わっていない文書は、埋め込みをやり直さない。
"""

import argparse
import logging
import time
from pathlib import Path

from evals.companies import DATASETS, load_dataset
from ingest.pdf_baseline import cached_pages
from retrieval.chunker import DEFAULT_MAX_CHARS
from retrieval.embedding import MODELS, Embedder, SentenceTransformerEmbedder
from retrieval.ingest import ingest_document
from retrieval.settings import DatabaseSettings
from retrieval.store import ChunkStore, DocumentRecord

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ingest_index")
# モデルの取得で huggingface_hub / httpx が出す大量のリクエストログを黙らせる
for _name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)

EDINET_DIR = Path("data/edinet")
TEXT_CACHE = Path("data/pdf_text")


def main() -> int:
    """対象の書類をチャンクに分け、埋め込んで DB に保存する。

    Returns:
        終了コード。正常終了は 0。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", nargs="+", choices=sorted(DATASETS), default=["dev", "heldout"]
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    embedders: list[Embedder] = []
    for key in args.models:
        started = time.time()
        embedders.append(SentenceTransformerEmbedder(MODELS[key]))
        log.info("モデルを読み込みました: %s（%.0f 秒）", key, time.time() - started)

    store = ChunkStore.connect(DatabaseSettings().database_url)
    total_chunks = 0
    for name in args.dataset:
        for company in load_dataset(name):
            filing = company.filings.current
            pages = cached_pages(EDINET_DIR / filing.doc_id / f"{filing.doc_id}.pdf", TEXT_CACHE)
            doc = DocumentRecord(
                filing.doc_id, company.sec_code, company.name, filing.period_end, len(pages)
            )
            started = time.time()
            result = ingest_document(store, doc, pages, embedders, max_chars=args.max_chars)
            total_chunks += result.n_chunks
            over = ", ".join(f"{k}: {v}" for k, v in result.over_limit.items())
            emb = ", ".join(f"{k}: {v}" for k, v in result.embedded.items())
            log.info(
                "%s %s: %d チャンク（今回の埋め込み %s / 入力上限超え %s）%.0f 秒",
                company.sec_code,
                company.name,
                result.n_chunks,
                emb,
                over,
                time.time() - started,
            )
    log.info("== 合計 %d チャンク", total_chunks)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""L2 評価: 質問に対して、根拠のある本文を上位で取れているかを、検索方式ごとに測る。

  uv run python -m scripts.eval_l2

先に scripts.ingest_index で取り込んでおく。検索は各質問の会社の文書に限る（メモを書く場面と同じ）。
レポートは evals/reports/l2_retrieval.md に書く。
"""

import logging
from collections.abc import Callable
from pathlib import Path

from evals.l2 import L2Metrics, Question, evaluate, load_questions, render_report
from retrieval.embedding import MODELS, Embedder, SentenceTransformerEmbedder
from retrieval.search import Mode, Retriever
from retrieval.settings import DatabaseSettings
from retrieval.store import ChunkStore, Hit

logging.basicConfig(level=logging.INFO, format="%(message)s")
for _name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("eval_l2")

KS = (1, 3, 5, 10)
REPORT = Path("evals/reports/l2_retrieval.md")
LABELS = {
    "dev": "開発用（50問）",
    "heldout": "保留データ（50問。設定を決めるときには使っていない）",
}


def main() -> int:
    store = ChunkStore.connect(DatabaseSettings().database_url)
    embedders: dict[str, Embedder] = {
        key: SentenceTransformerEmbedder(spec) for key, spec in MODELS.items()
    }
    retrievers = {key: Retriever(store, e) for key, e in embedders.items()}

    def system(key: str, mode: Mode) -> Callable[[Question], list[Hit]]:
        return lambda q: retrievers[key].search(
            q.question, k=max(KS), mode=mode, doc_ids=[q.doc_id]
        )

    systems: dict[str, Callable[[Question], list[Hit]]] = {
        "全文（pg_bigm）": system("e5-small", "lexical"),
        "全文（BM25 文字2-gram）": system("e5-small", "bm25"),
        "ベクトル: e5-small": system("e5-small", "vector"),
        "ベクトル: ruri-v3-30m": system("ruri-v3-30m", "vector"),
        "融合: 全文 + e5-small": system("e5-small", "hybrid"),
        "融合: 全文 + ruri-v3-30m": system("ruri-v3-30m", "hybrid"),
        "融合: BM25 + e5-small": system("e5-small", "hybrid_bm25"),
        "融合: BM25 + ruri-v3-30m": system("ruri-v3-30m", "hybrid_bm25"),
    }

    sections = [
        "# L2 評価: 検索（質問に対して、根拠のある本文を上位で取れているか）",
        "",
        "正解の取得は、同じ文書の、根拠のページと引用を含むチャンク（`evals/l2.py`）。"
        "検索は質問の会社の文書に限る。チャンクは最大600文字。",
        "",
    ]
    for split in ("dev", "heldout"):
        questions = load_questions(split)
        results: dict[str, L2Metrics] = {}
        for name, search in systems.items():
            results[name] = evaluate(questions, search, ks=KS)
            log.info(
                "[%s] %s: Recall@5=%.2f MRR=%.3f",
                split,
                name,
                results[name].recall_at[5],
                results[name].mrr,
            )
        sections.append(render_report(LABELS[split], results, ks=KS))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(sections), encoding="utf-8")
    log.info("== レポート: %s", REPORT)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

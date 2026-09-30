"""L2 評価: 質問に対して、根拠のある本文を上位で取れているか（検索の評価）。

- 正解の取得: 同じ文書の、根拠のページを含み、根拠の引用（本文の一節）を含むチャンク。
  チャンクの大きさを変えても成り立つよう、チャンクIDではなくページと引用で定義する。
- 指標: Recall@k（上位 k 件に正解があった質問の割合）と MRR（最初の正解の順位の逆数の平均）。
- 質問は開発用（規則や設定を決めるときに使う）と保留（最後に一度だけ測る）に分ける。
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from retrieval.chunker import Chunk
from retrieval.store import Hit

DATASET_PATH = Path(__file__).parent / "datasets" / "l2_questions.json"
MRR_DEPTH = 10  # MRR は上位10件までで数える


class Gold(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int  # 1 始まり
    evidence: str  # 本文の一節（1行）。チャンクの本文にそのまま含まれる


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    split: Literal["dev", "heldout"]
    sec_code: str
    doc_id: str
    category: str
    question: str
    gold: list[Gold]


class _Dataset(BaseModel):
    note: str
    questions: list[Question]


def load_questions(split: str | None = None, path: Path = DATASET_PATH) -> list[Question]:
    dataset = _Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return [q for q in dataset.questions if split is None or q.split == split]


def is_relevant(chunk: Chunk, question: Question) -> bool:
    """同じ文書で、根拠のページを含み、根拠の引用を含むチャンクなら正解。"""
    if chunk.doc_id != question.doc_id:
        return False
    return any(
        chunk.page_start <= g.page <= chunk.page_end and g.evidence in chunk.text
        for g in question.gold
    )


def is_on_gold_page(chunk: Chunk, question: Question) -> bool:
    """同じ文書で、根拠のページを含むチャンクなら、ページ単位の正解（引用は含まなくてもよい）。

    メモに出典として付けるのはページなので、根拠の一文が隣のチャンクにあっても、
    同じページを取れていれば出典としては足りる。厳密な指標（is_relevant）の補助として使う。
    """
    return chunk.doc_id == question.doc_id and any(
        chunk.page_start <= g.page <= chunk.page_end for g in question.gold
    )


@dataclass(frozen=True)
class CategoryMetrics:
    n: int
    recall_at: dict[int, float]
    mrr: float


@dataclass(frozen=True)
class L2Metrics:
    n: int
    recall_at: dict[int, float]
    page_recall_at: dict[int, float]  # ページ単位（根拠のページ上のチャンクが上位 k 件にあるか）
    mrr: float
    misses: list[str]  # 上位 max(ks) 件に正解が無かった質問の ID
    by_category: dict[str, CategoryMetrics]


def _first_rank(
    hits: Sequence[Hit], question: Question, match: Callable[[Chunk, Question], bool]
) -> int | None:
    for i, hit in enumerate(hits, start=1):
        if match(hit.chunk, question):
            return i
    return None


def _summarize(ranks: Sequence[int | None], ks: Sequence[int]) -> tuple[dict[int, float], float]:
    n = len(ranks)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    mrr = sum(1 / r for r in ranks if r is not None and r <= MRR_DEPTH) / n
    return recall, mrr


def evaluate(
    questions: Sequence[Question],
    search: Callable[[Question], list[Hit]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> L2Metrics:
    if not questions:
        raise ValueError("質問が0件です")
    ranks: dict[str, int | None] = {}
    page_ranks: dict[str, int | None] = {}
    for q in questions:
        hits = search(q)
        ranks[q.id] = _first_rank(hits, q, is_relevant)
        page_ranks[q.id] = _first_rank(hits, q, is_on_gold_page)
    recall, mrr = _summarize([ranks[q.id] for q in questions], ks)
    page_recall, _ = _summarize([page_ranks[q.id] for q in questions], ks)

    by_category: dict[str, CategoryMetrics] = {}
    for category in sorted({q.category for q in questions}):
        in_category = [ranks[q.id] for q in questions if q.category == category]
        cat_recall, cat_mrr = _summarize(in_category, ks)
        by_category[category] = CategoryMetrics(len(in_category), cat_recall, cat_mrr)

    limit = max(ks)
    misses = [q.id for q in questions if ranks[q.id] is None or ranks[q.id] > limit]  # type: ignore[operator]
    return L2Metrics(len(questions), recall, page_recall, mrr, misses, by_category)


def longest_common_substring(a: str, b: str) -> int:
    """2つの文字列の、最長の共通部分文字列の長さ。設問が根拠の写しになっていないかの目安に使う。"""
    best = 0
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def render_report(
    title: str, systems: dict[str, L2Metrics], ks: Sequence[int] = (1, 3, 5, 10)
) -> str:
    columns = " | ".join(f"Recall@{k}" for k in ks)
    mid_k = ks[len(ks) // 2]
    lines = [
        f"### {title}",
        "",
        f"| 方式 | {columns} | MRR | ページ単位 Recall@{mid_k} |",
        "| --- | " + " | ".join("---" for _ in ks) + " | --- | --- |",
    ]
    for name, m in systems.items():
        cells = " | ".join(f"{m.recall_at[k]:.2f}" for k in ks)
        lines.append(f"| {name} | {cells} | {m.mrr:.3f} | {m.page_recall_at[mid_k]:.2f} |")

    any_metrics = next(iter(systems.values()))
    mid = ks[len(ks) // 2]
    lines += ["", f"分類ごとの Recall@{mid}:", ""]
    lines += [
        "| 分類（件数） | " + " | ".join(systems) + " |",
        "| --- | " + " | ".join("---" for _ in systems) + " |",
    ]
    for category, cat in any_metrics.by_category.items():
        cells = " | ".join(
            f"{m.by_category[category].recall_at[mid]:.2f}" for m in systems.values()
        )
        lines.append(f"| {category}（{cat.n}） | {cells} |")

    lines += ["", f"上位 {max(ks)} 件に正解が無かった質問:", ""]
    for name, m in systems.items():
        lines.append(f"- {name}: {', '.join(m.misses) if m.misses else 'なし'}")
    return "\n".join(lines) + "\n"

"""L2 評価（検索）の物差しのテスト: 正解の取得の定義、Recall@k、MRR、設問データの整合。"""

import pytest

from evals.companies import load_dataset
from evals.l2 import (
    Gold,
    Question,
    evaluate,
    is_relevant,
    load_questions,
    longest_common_substring,
    render_report,
)
from retrieval.chunker import Chunk
from retrieval.store import Hit


def _q(**kw: object) -> Question:
    base: dict[str, object] = {
        "id": "dev-001",
        "split": "dev",
        "sec_code": "1111",
        "doc_id": "D1",
        "category": "リスク",
        "question": "原材料が高騰した場合の影響は。",
        "gold": [
            Gold(page=5, evidence="原材料価格が高騰した場合、収益を圧迫する可能性があります。")
        ],
    }
    base.update(kw)
    return Question(**base)  # type: ignore[arg-type]


def _chunk(doc: str = "D1", start: int = 5, end: int = 5, text: str = "") -> Chunk:
    return Chunk(f"{doc}:0", doc, ["h"], start, end, text)


EVIDENCE = "原材料価格が高騰した場合、収益を圧迫する可能性があります。"


# ---- 正解の取得の定義 ----


def test_同じ文書で_根拠のページを含み_根拠の引用を含むチャンクが正解() -> None:
    assert is_relevant(_chunk(text=f"前の文。{EVIDENCE}次の文。"), _q())


def test_ページ範囲に根拠のページが含まれていればよい() -> None:
    assert is_relevant(_chunk(start=4, end=6, text=EVIDENCE), _q())


def test_別の文書のチャンクは_同じ引用を含んでいても正解にならない() -> None:
    assert not is_relevant(_chunk(doc="D2", text=EVIDENCE), _q())


def test_引用を含まないチャンクは正解にならない() -> None:
    assert not is_relevant(_chunk(text="従業員の平均年齢は四十歳です。"), _q())


def test_ページが違えば_引用を含んでいても正解にならない() -> None:
    assert not is_relevant(_chunk(start=9, end=9, text=EVIDENCE), _q())


def test_根拠が複数ページにある設問は_どれかに合えば正解() -> None:
    q = _q(gold=[Gold(page=5, evidence=EVIDENCE), Gold(page=8, evidence=EVIDENCE)])
    assert is_relevant(_chunk(start=8, end=8, text=EVIDENCE), q)


# ---- 指標 ----


def _hit(chunk: Chunk, rank: int) -> Hit:
    return Hit(chunk, 1.0, rank)


def _hits_with_relevant_at(rank: int | None, n: int = 10) -> list[Hit]:
    hits: list[Hit] = []
    for i in range(1, n + 1):
        text = EVIDENCE if i == rank else f"無関係な文 {i}"
        hits.append(_hit(Chunk(f"D1:{i}", "D1", ["h"], 5, 5, text), i))
    return hits


def test_RecallとMRRを計算する() -> None:
    questions = [_q(id="a"), _q(id="b"), _q(id="c"), _q(id="d")]
    ranks = {"a": 1, "b": 3, "c": 8, "d": None}

    metrics = evaluate(questions, lambda q: _hits_with_relevant_at(ranks[q.id]), ks=(1, 3, 5, 10))

    assert metrics.n == 4
    assert metrics.recall_at == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}
    assert metrics.mrr == pytest.approx((1 + 1 / 3 + 1 / 8 + 0) / 4)
    assert metrics.misses == ["d"]  # 上位k件に正解が無かった設問


def test_kより下の順位の正解は_Recallに数えるが_MRRは上位10件までで数える() -> None:
    q = _q(id="a")
    metrics = evaluate([q], lambda _: _hits_with_relevant_at(7), ks=(5, 10))
    assert metrics.recall_at == {5: 0.0, 10: 1.0}


def test_分類ごとの結果も出す() -> None:
    questions = [_q(id="a", category="リスク"), _q(id="b", category="配当")]
    metrics = evaluate(
        questions, lambda q: _hits_with_relevant_at(1 if q.id == "a" else None), ks=(1,)
    )
    assert metrics.by_category["リスク"].recall_at[1] == 1.0
    assert metrics.by_category["配当"].recall_at[1] == 0.0


def test_設問が0件ならエラー() -> None:
    with pytest.raises(ValueError):
        evaluate([], lambda _: [], ks=(1,))


# ---- 設問と根拠の言葉の重なり（設問が根拠の写しになっていないかの目安） ----


def test_最長共通部分文字列() -> None:
    assert longest_common_substring("原材料価格の高騰", "原材料価格が高騰した") == len("原材料価格")
    assert longest_common_substring("あいう", "えお") == 0
    assert longest_common_substring("", "abc") == 0


# ---- レポート ----


def test_レポートは方式ごとの指標を表にする() -> None:
    q = _q(id="a")
    m = evaluate([q], lambda _: _hits_with_relevant_at(1), ks=(1, 3))
    md = render_report("開発用", {"全文": m, "ベクトル": m}, ks=(1, 3))
    assert "開発用" in md
    assert "全文" in md and "ベクトル" in md
    assert "Recall@1" in md and "MRR" in md


# ---- 設問データの整合（コミットされたデータ。CI でも検査する） ----


def test_設問データは_開発用と保留データが50問ずつ() -> None:
    questions = load_questions()
    assert len(questions) == 100
    assert sum(q.split == "dev" for q in questions) == 50
    assert sum(q.split == "heldout" for q in questions) == 50
    assert len({q.id for q in questions}) == 100


def test_設問の会社と文書は_対象企業のデータと一致する() -> None:
    for split in ("dev", "heldout"):
        companies = {c.sec_code: c.filings.current.doc_id for c in load_dataset(split)}
        for q in load_questions(split):
            assert companies[q.sec_code] == q.doc_id, q.id


def test_どの会社にも_開発用と保留の両方で5問ずつある() -> None:
    for split in ("dev", "heldout"):
        counts: dict[str, int] = {}
        for q in load_questions(split):
            counts[q.sec_code] = counts.get(q.sec_code, 0) + 1
        assert set(counts.values()) == {5}, (split, counts)


def test_根拠は十分な長さがあり_ページが正の整数() -> None:
    for q in load_questions():
        assert q.gold, q.id
        for g in q.gold:
            assert len(g.evidence) >= 15, q.id
            assert g.page >= 1, q.id


def test_設問は根拠の写しではない() -> None:
    # 設問と根拠の一節に、長い共通部分があれば、検索が語句の一致だけで解けてしまう
    too_close = [
        (q.id, longest_common_substring(q.question, g.evidence))
        for q in load_questions()
        for g in q.gold
        if longest_common_substring(q.question, g.evidence) >= 12
    ]
    assert too_close == []


# ---- ページ単位の Recall（出典として付けるのはページ） ----


def test_ページ単位の正解は_根拠の引用を含まなくても正解ページ上のチャンク() -> None:
    from evals.l2 import is_on_gold_page

    q = _q()
    assert is_on_gold_page(_chunk(start=5, end=5, text="同じページだが、別の文。"), q)
    assert is_on_gold_page(_chunk(start=4, end=6, text="範囲が正解ページを含む。"), q)
    assert not is_on_gold_page(_chunk(start=9, end=9, text=EVIDENCE), q)  # ページが違う
    assert not is_on_gold_page(_chunk(doc="D2", start=5, end=5, text=EVIDENCE), q)  # 文書が違う


def test_ページ単位のRecallは_厳密なRecall以上になる() -> None:
    def hits_on_page_without_evidence(_: Question) -> list[Hit]:
        # 正解ページ上のチャンクだが、根拠の引用は含まない（隣のチャンクにある）
        return [_hit(Chunk("D1:1", "D1", ["h"], 5, 5, "同じ節の別の文"), 1)]

    metrics = evaluate([_q()], hits_on_page_without_evidence, ks=(1, 5))

    assert metrics.recall_at == {1: 0.0, 5: 0.0}
    assert metrics.page_recall_at == {1: 1.0, 5: 1.0}

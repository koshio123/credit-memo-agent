"""PDF の本文を、見出しの階層とページ範囲つきのチャンクに分ける規則のテスト。

文面は実データ（有価証券報告書の PDF、NFKC で正規化済み）の書式に基づく。
"""

import pytest

from retrieval.chunker import Chunk, chunk_pages

HEADER = "EDINET提出書類\nサンプル株式会社(E00000)\n有価証券報告書\n"


def page(body: str, number: int = 1, total: int = 100) -> str:
    return f"{HEADER}{body}\n{number}/{total}"


PAGE_1 = page(
    """第一部 【企業情報】
第2 【事業の状況】
1 【経営方針、経営環境及び対処すべき課題等】
(1) 経営方針
当社グループは、社会の安全と安心に貢献することを経営の基本方針としております。
(2) 経営環境
国内の建設需要は堅調に推移しております。
2 【サステナビリティに関する考え方及び取組】
当社は、サステナビリティを重要な経営課題と位置づけております。""",
    1,
)

PAGE_2 = page(
    """3 【事業等のリスク】
(1) 原材料価格の変動
原材料価格が高騰した場合、収益を圧迫する可能性があります。""",
    2,
)


def by_text(chunks: list[Chunk], needle: str) -> Chunk:
    (found,) = [c for c in chunks if needle in c.text]
    return found


# ---- ページの体裁を除く ----


def test_ページの先頭3行と末尾のページ番号は本文に含めない() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    joined = "\n".join(c.text for c in chunks)
    assert "EDINET提出書類" not in joined
    assert "(E00000)" not in joined
    assert "1/100" not in joined
    assert "経営の基本方針" in joined


def test_本文が無いページは飛ばす() -> None:
    assert chunk_pages("D1", [page(""), page("   ")]) == []


# ---- 見出しの階層 ----


def test_見出しの階層をチャンクに持たせる() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    c = by_text(chunks, "経営の基本方針")
    assert c.heading_path == [
        "第一部 【企業情報】",
        "第2 【事業の状況】",
        "1 【経営方針、経営環境及び対処すべき課題等】",
        "(1) 経営方針",
    ]


def test_同じ階層の次の見出しで_下位の見出しは置き換わる() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    c = by_text(chunks, "建設需要")
    assert c.heading_path[-1] == "(2) 経営環境"
    assert "(1) 経営方針" not in c.heading_path


def test_上位の見出しが変わると_下位の見出しは消える() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    c = by_text(chunks, "サステナビリティを重要な経営課題")
    assert c.heading_path == [
        "第一部 【企業情報】",
        "第2 【事業の状況】",
        "2 【サステナビリティに関する考え方及び取組】",
    ]


def test_見出しの階層はページをまたいで引き継ぐ() -> None:
    chunks = chunk_pages("D1", [PAGE_1, PAGE_2])
    c = by_text(chunks, "原材料価格が高騰")
    assert c.heading_path[:2] == ["第一部 【企業情報】", "第2 【事業の状況】"]
    assert c.heading_path[-2:] == ["3 【事業等のリスク】", "(1) 原材料価格の変動"]


def test_見出しの行そのものはチャンクの本文に入れない() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    assert all(not c.text.startswith("第2 【") for c in chunks)
    assert all("【事業等のリスク】" not in c.text for c in chunks)


def test_チャンクは見出しをまたがない() -> None:
    chunks = chunk_pages("D1", [PAGE_1])
    for c in chunks:
        assert not ("経営の基本方針" in c.text and "建設需要" in c.text)


# ---- ページ範囲（出典） ----


def test_ページ範囲を1始まりで持つ() -> None:
    chunks = chunk_pages("D1", [PAGE_1, PAGE_2])
    assert (
        by_text(chunks, "経営の基本方針").page_start,
        by_text(chunks, "経営の基本方針").page_end,
    ) == (1, 1)
    risk = by_text(chunks, "原材料価格が高騰")
    assert (risk.page_start, risk.page_end) == (2, 2)


def test_同じ節が2ページにまたがるときはページ範囲が広がる() -> None:
    p1 = page("1 【事業等のリスク】\n" + "A" * 50 + "。", 1)
    p2 = page("B" * 50 + "。", 2)
    (c,) = chunk_pages("D1", [p1, p2], max_chars=1000)
    assert (c.page_start, c.page_end) == (1, 2)


# ---- 長さ ----


def test_チャンクは最大文字数を超えず_行の途中では切らない() -> None:
    lines = [f"表の行{i:03d} 1,000 2,000" for i in range(50)]
    p = page("1 【表】\n" + "\n".join(lines))
    chunks = chunk_pages("D1", [p], max_chars=200)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    rebuilt = "\n".join(c.text for c in chunks).split("\n")
    assert rebuilt == lines  # 行の欠落・重複・分断がない


def test_1行が最大文字数より長いときは_句点で切る() -> None:
    sentence = "これは長い文です。" * 30  # 9文字 × 30 = 270文字
    p = page("1 【本文】\n" + sentence)
    chunks = chunk_pages("D1", [p], max_chars=100)

    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == sentence


def test_句点が無い長すぎる行も_最大文字数で切って落とさない() -> None:
    line = "あ" * 250
    chunks = chunk_pages("D1", [page("1 【本文】\n" + line)], max_chars=100)
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == line


# ---- 識別子・再現性 ----


def test_チャンクIDは文書IDと連番で_同じ入力なら同じ結果() -> None:
    a = chunk_pages("D1", [PAGE_1, PAGE_2])
    b = chunk_pages("D1", [PAGE_1, PAGE_2])
    assert a == b
    assert [c.chunk_id for c in a] == [f"D1:{i}" for i in range(len(a))]


def test_文書IDが空なら失敗する() -> None:
    with pytest.raises(ValueError):
        chunk_pages("", [PAGE_1])


# ---- 見出しの誤検出 ----


def test_数値の行や本文中の括弧は見出しにしない() -> None:
    body = (
        "1 【表】\n(1) 経営方針\n売上高 (百万円) 25,678 57,020\n"
        "(注) 1.前期比較を示しています。\n通常の文。"
    )
    (c,) = chunk_pages("D1", [page(body)], max_chars=1000)
    assert c.heading_path == ["1 【表】", "(1) 経営方針"]
    assert "売上高 (百万円) 25,678 57,020" in c.text
    assert "(注) 1.前期比較を示しています。" in c.text


def test_日付を含む表の行は_項の見出しにしない() -> None:
    # 実データ: PDF の折り返しで「添」で切れた表の行。短く、長さの条件だけでは見出しになる
    row = "(1) 有価証券報告書及びその添 事業年度 自 2024年4月1日 2025年6月25日"
    body = f"1 【表】\n(1) 経営方針\n{row}\n次の行。"
    (c,) = chunk_pages("D1", [page(body)], max_chars=1000)

    assert c.heading_path == ["1 【表】", "(1) 経営方針"]
    assert row in c.text

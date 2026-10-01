"""メモの Markdown 出力。表と出典の番号付けはコードが行う。"""

import json
import re

import pytest

from agents.pipeline import MemoResult, run_baseline
from agents.render import render_memo
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend

pytestmark = pytest.mark.anyio


def _claim(text: str, *ids: str) -> dict[str, object]:
    return {"text": text, "evidence_ids": list(ids)}


async def make_result(
    service: EdinetService,
    *,
    risky: bool = False,
    extra: dict[str, list[dict[str, object]]] | None = None,
) -> MemoResult:
    from agents.evidence import collect_metrics
    from agents.state import EvidencePool

    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)
    eq = metrics["equity_ratio.current"].id
    ps = f"E{len(pool.items) + 1}"
    sections: dict[str, list[dict[str, object]]] = {
        "overview": [_claim("機械を製造している。", ps)],
        "financial_findings": [_claim("自己資本比率は50.0%である。", eq)],
        "business_risks": [_claim("原材料価格が高騰する可能性がある。", ps)],
        "positives": [_claim("自己資本比率の水準は標準である。", eq)],
        "negatives": [_claim("原材料価格の影響を受けうる。", ps)],
        "open_items": [_claim("取引先への依存度は確認できなかった。")],
    }
    sections.update(extra or {})
    backend = ScriptedBackend([json.dumps(sections, ensure_ascii=False)])
    return await run_baseline(service, backend, "9999")


async def test_テンプレートの節が順に並ぶ(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    headings = [
        "# 与信メモ（草案）",
        "## 0. 最優先の確認事項",
        "## 1. 企業概要",
        "## 2. 財務分析",
        "### 2.1 主要指標（2期比較）",
        "### 2.2 所見",
        "### 2.3 有利子負債の構成",
        "## 3. 事業リスク",
        "## 4. 内規への照合結果",
        "## 5. 与信判断上の論点",
        "## 6. 確認が必要な事項",
        "## 出典一覧",
    ]
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)


async def test_出典一覧の資料名は_会社名と有価証券報告書の間を空ける(
    service: EdinetService,
) -> None:
    sources = render_memo(await make_result(service)).partition("## 出典一覧")[2]
    assert "サンプル工業 有価証券報告書（" in sources


async def test_第13条の検索の記録は_XBRLの財務データとは書かない(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    row = next(line for line in text.partition("## 出典一覧")[2].splitlines() if "継続企業" in line)
    assert "XBRL" not in row.split("|")[2]
    assert "語句検索" in row


async def test_主張を得られなかった節は_記載がないとは書かず_検査の記録に載せる(
    service: EdinetService,
) -> None:
    result = await make_result(service, extra={"overview": []})
    text = render_memo(result)
    overview = text.partition("## 1.")[2].partition("## 2.")[0]
    assert "主張を得られなかった" in overview
    assert "記載がないことを意味しない" in overview
    record = text.partition("## 検査の記録")[2]
    assert "主張が1件も得られなかった節" in record and "企業概要" in record


async def test_ヘッダーに_企業名_証券コード_期間_草案の注意がある(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    assert "サンプル工業" in text and "9999" in text
    assert "2026-03-31" in text and "2025-03-31" in text
    assert "人によるレビュー" in text
    assert "結論も推奨も含まない" in text


async def test_主張には出典番号がつき_出典一覧に同じ番号がある(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    body, _, sources = text.partition("## 出典一覧")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    listed = {int(n) for n in re.findall(r"^\| (\d+) \|", sources, re.M)}
    assert cited and cited == listed
    assert "機械を製造している。 [" in body


async def test_出典の番号は_本文に最初に出た順で_同じ証拠は同じ番号(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    body = text.partition("## 出典一覧")[0]
    first_seen: list[int] = []
    for n in (int(x) for x in re.findall(r"\[(\d+)\]", body)):
        if n not in first_seen:
            first_seen.append(n)
    assert first_seen == sorted(first_seen)
    assert first_seen[0] == 1


async def test_出典一覧には_本文はページと引用_数値は算式と根拠がある(
    service: EdinetService,
) -> None:
    text = render_memo(await make_result(service))
    sources = text.partition("## 出典一覧")[2]
    assert "p.2" in sources and "原材料価格が高騰する可能性があります" in sources
    assert "純資産" in sources and "総資産" in sources
    assert "PDF p.3" in sources  # 数値の入力値をPDFが読んだページ


async def test_主要指標の表は_前期と直近期と水準と根拠を示す(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    table = text.partition("### 2.1")[2].partition("### 2.2")[0]
    row = next(line for line in table.splitlines() if line.startswith("| 自己資本比率"))
    assert "50.0%" in row and "標準" in row and "純資産" in row
    icr = next(line for line in table.splitlines() if line.startswith("| インタレスト"))
    assert "算定不能" in icr


async def test_有利子負債の表は_内訳と1年以内の割合を示す(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    section = text.partition("### 2.3")[2].partition("## 3.")[0]
    assert "短期借入金" in section and "長期借入金" in section
    assert "37.5%" in section


async def test_内規照合の表は_条項_確認内容_結果_根拠(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    section = text.partition("## 4.")[2].partition("## 5.")[0]
    assert "| 第8条 | 自己資本比率 | 標準 |" in section
    assert "| 第13条 |" in section
    assert "確認できなかった" in section


async def test_継続企業の前提に記載があれば_冒頭に最優先として示す(service: EdinetService) -> None:
    from tests.edinet_fakes import PAGES

    original = list(PAGES)
    PAGES[2] = "継続企業の前提に関する重要な不確実性が存在します。"
    try:
        text = render_memo(await make_result(service))
    finally:
        PAGES[:] = original
    top = text.partition("## 0.")[2].partition("## 1.")[0]
    assert "継続企業の前提" in top and "最優先" in top
    assert re.search(r"\[\d+\]", top)


async def test_記載が見つからなければ_冒頭は断定せず_確認できなかったと書く(
    service: EdinetService,
) -> None:
    text = render_memo(await make_result(service))
    top = text.partition("## 0.")[2].partition("## 1.")[0]
    assert "確認できなかった" in top
    assert "問題はない" not in top and "問題なし" not in top


async def test_確認事項に_算定不能の指標を自動で加える(service: EdinetService) -> None:
    text = render_memo(await make_result(service))
    section = text.partition("## 6.")[2].partition("## 出典一覧")[0]
    assert "取引先への依存度は確認できなかった" in section  # LLM の項目
    assert "インタレスト・カバレッジ・レシオ" in section and "算定不能" in section  # コードの項目


async def test_検査で除外した主張は_メモの本文に入れず_検査結果の節に記録する(
    service: EdinetService,
) -> None:
    result = await make_result(service, extra={"overview": [_claim("出典のない主張。")]})
    text = render_memo(result)
    body = text.partition("## 検査の記録")[0]
    assert "出典のない主張。" not in body
    record = text.partition("## 検査の記録")[2]
    assert "出典のない主張。" in record and "出典がありません" in record


async def test_表のセルの縦線は壊さない(service: EdinetService) -> None:
    result = await make_result(service)
    result.memo.policy_rows[0] = result.memo.policy_rows[0].__class__(
        "第8条", "A|B", "標準", result.memo.policy_rows[0].evidence_ids
    )
    text = render_memo(result)
    assert "A\\|B" in text

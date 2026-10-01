"""人が主張と出典を突き合わせるための確認用の出力。"""

import json

import pytest

from agents.pipeline import run_baseline
from agents.review import render_review
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend

pytestmark = pytest.mark.anyio


async def _payload(service: EdinetService) -> dict[str, object]:
    from agents.evidence import collect_metrics
    from agents.export import to_payload
    from agents.state import EvidencePool

    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)
    eq, ps = metrics["equity_ratio.current"].id, f"E{len(pool.items) + 1}"

    def c(text: str, *ids: str) -> dict[str, object]:
        return {"text": text, "evidence_ids": list(ids)}

    reply = {
        "overview": [c("機械を製造している。", ps)],
        "financial_findings": [c("自己資本比率は50.0%である。", eq)],
        "business_risks": [c("原材料価格が高騰する可能性がある。", ps)],
        "positives": [],
        "negatives": [],
        "open_items": [c("依存度は確認できなかった。")],
    }
    result = await run_baseline(service, ScriptedBackend([json.dumps(reply)]), "9999")
    return to_payload(result, backend="scripted", model="fake")


async def test_主張ごとに_出典の引用とページを並べ_判定の欄をつける(service: EdinetService) -> None:
    text = render_review(await _payload(service))

    assert "# 主張と出典の突き合わせ" in text
    # 本文の出典: ページ・引用文
    assert "機械を製造している。" in text
    assert "p.2" in text and "原材料価格が高騰する可能性があります" in text
    # 数値の出典: 算式・PDFのページ
    assert "自己資本比率は50.0%である。" in text
    assert "純資産" in text and "PDF p.3" in text
    # 判定の欄
    assert text.count("- [ ] 支持") >= 4
    assert "- [ ] 一部だけ" in text and "- [ ] 支持しない" in text


async def test_出典のない確認事項は_出典なしと明示する(service: EdinetService) -> None:
    text = render_review(await _payload(service))
    block = text.partition("依存度は確認できなかった。")[2][:200]
    assert "出典なし" in block


async def test_節ごとに見出しを分ける(service: EdinetService) -> None:
    text = render_review(await _payload(service))
    for heading in ("## 企業概要", "## 財務の所見", "## 事業リスク", "## 確認が必要な事項"):
        assert heading in text


async def test_節の主張が無ければ_その節は作らない(service: EdinetService) -> None:
    text = render_review(await _payload(service))
    assert "## 肯定的な要素" not in text

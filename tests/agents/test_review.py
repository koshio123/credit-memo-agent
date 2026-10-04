"""人が主張と出典を突き合わせるための確認用の出力。"""

import json

import pytest

from agents.export import SavedResult, from_result
from agents.pipeline import run_baseline
from agents.review import render_review
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend

pytestmark = pytest.mark.anyio


def make_claim(text: str, *ids: str) -> dict[str, object]:
    return {"text": text, "evidence_ids": list(ids)}


async def make_saved(
    service: EdinetService, extra: dict[str, list[dict[str, object]]] | None = None
) -> SavedResult:
    from agents.evidence import collect_metrics
    from agents.state import EvidencePool

    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)
    eq, ps = metrics["equity_ratio.current"].id, f"E{len(pool.items) + 1}"
    reply: dict[str, list[dict[str, object]]] = {
        "overview": [make_claim("機械を製造している。", ps)],
        "financial_findings": [make_claim("自己資本比率は50.0%である。", eq)],
        "business_risks": [make_claim("原材料価格が高騰する可能性がある。", ps)],
        "positives": [],
        "negatives": [],
        "open_items": [make_claim("依存度は確認できなかった。")],
    }
    reply.update(extra or {})
    result = await run_baseline(service, ScriptedBackend([json.dumps(reply)]), "9999")
    return from_result(result, backend="scripted", model="fake")


async def test_主張ごとに_出典の引用とページを並べ_判定の欄をつける(service: EdinetService) -> None:
    text = render_review(await make_saved(service))

    assert "# 主張と出典の突き合わせ" in text
    assert "機械を製造している。" in text
    # 本文の出典: 書類ID・ページごとの引用
    assert "S100CUR0" in text and "p.2: " in text
    assert "原材料価格が高騰する可能性があります" in text
    # 数値の出典: 算式・PDFのページ
    assert "純資産" in text and "PDF p.3" in text
    # 判定の欄は、主張の数（本文の出典のある4件＋出典のない確認事項1件＋内規照合）ごとに1つ
    assert text.count("    - [ ] 支持する\n") == text.count("### ")
    assert text.count("### ") >= 4


async def test_出典のない確認事項は_出典なしと明示する(service: EdinetService) -> None:
    text = render_review(await make_saved(service))
    block = text.partition("依存度は確認できなかった。")[2][:200]
    assert "出典なし" in block


async def test_節ごとに見出しを分け_主張の無い節は作らない(service: EdinetService) -> None:
    text = render_review(await make_saved(service))
    for heading in ("## 企業概要", "## 財務の所見", "## 事業リスク", "## 確認が必要な事項"):
        assert heading in text
    assert "## 肯定的な要素" not in text


async def test_内規照合の各行も_根拠と一緒に確認できる(service: EdinetService) -> None:
    text = render_review(await make_saved(service))
    section = text.partition("## 内規照合")[2]
    assert "第8条" in section and "自己資本比率" in section
    assert "第13条" in section and "確認できなかった" in section


async def test_数値の警告がある主張は_確認用にも印と理由を示す(service: EdinetService) -> None:
    saved = await make_saved(
        service, {"financial_findings": [make_claim("自己資本比率は67.4%である。", "E1")]}
    )
    text = render_review(saved)
    block = text.partition("自己資本比率は67.4%である。")[2][:300]
    assert "⚠" in block and "67.4" in block


async def test_複数ページの引用は_ページごとに分けて示す(service: EdinetService) -> None:
    from retrieval.citations import SourceSpan

    saved = await make_saved(service)
    passage = next(e for e in saved.evidence.values() if e.kind == "passage")
    two = passage.model_copy(
        update={
            "spans": [
                SourceSpan(doc_id="S100CUR0", page=23, start=0, end=3, quote="あいう"),
                SourceSpan(doc_id="S100CUR0", page=24, start=0, end=3, quote="えおか"),
            ]
        }
    )
    saved.evidence[passage.id] = two
    text = render_review(saved)
    first = text.partition("p.23:")[2]
    assert first.lstrip().startswith("> あいう")  # 各ページに、そのページの引用が付く
    assert text.partition("p.24:")[2].lstrip().startswith("> えおか")


async def test_保存した結果の読み込みは_存在しない出典IDを理由つきで拒否する(
    service: EdinetService,
) -> None:
    saved = await make_saved(service)
    data = json.loads(saved.model_dump_json())
    data["memo"]["overview"][0]["evidence_ids"] = ["E999"]
    with pytest.raises(ValueError, match="E999"):
        SavedResult.model_validate(data)


async def test_Verifierの判定があれば_主張の下に示す_人の判定と比べるため(
    service: EdinetService,
) -> None:
    from agents.memo import VerificationRecord

    saved = await make_saved(service)
    text0 = saved.memo.financial_findings[0].text
    saved.memo.verifications = [
        VerificationRecord(
            section="financial_findings", claim_text=text0, verdict="partial", reason="言い過ぎ"
        )
    ]
    out = render_review(saved)
    assert "Verifier（LLM）の判定: partial（言い過ぎ）" in out

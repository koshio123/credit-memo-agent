"""検証と差し戻しの状態遷移。遷移はコードで決まり、LLM は遷移を決めない。"""

import json

import pytest

from agents.machine import Stage, next_stage, run_verification
from agents.memo import MemoDraft
from agents.state import Claim, EvidencePool, PassageEvidence
from llm.fake import ScriptedBackend
from retrieval.citations import SourceSpan

pytestmark = pytest.mark.anyio


# ---- 遷移（純粋な関数） ----


def test_検証の後に失敗が無ければ_完了() -> None:
    assert next_stage(Stage.VERIFYING, failing=0, round_no=0, max_rounds=2) is Stage.DONE


def test_検証の後に失敗があり_回数が残っていれば_書き直し() -> None:
    assert next_stage(Stage.VERIFYING, failing=2, round_no=0, max_rounds=2) is Stage.REVISING
    assert next_stage(Stage.VERIFYING, failing=2, round_no=1, max_rounds=2) is Stage.REVISING


def test_回数を使い切れば_失敗が残っていても完了() -> None:
    assert next_stage(Stage.VERIFYING, failing=2, round_no=2, max_rounds=2) is Stage.DONE


def test_回数0なら_検証だけして完了() -> None:
    assert next_stage(Stage.VERIFYING, failing=3, round_no=0, max_rounds=0) is Stage.DONE


def test_書き直しの後は_必ず検証に戻る() -> None:
    assert next_stage(Stage.REVISING, failing=0, round_no=1, max_rounds=2) is Stage.VERIFYING


def test_完了からは動かない() -> None:
    assert next_stage(Stage.DONE, failing=5, round_no=0, max_rounds=2) is Stage.DONE


# ---- 実行 ----


def _setup() -> tuple[EvidencePool, MemoDraft]:
    pool = EvidencePool()
    for text in ("原材料の価格が上昇すると影響する。", "従業員は百名である。"):
        pool.add(
            PassageEvidence(
                id="",
                sec_code="9999",
                company="x",
                heading_path=["h"],
                spans=[SourceSpan(doc_id="D1", page=1, start=0, end=len(text), quote=text)],
            )
        )
    memo = MemoDraft(
        sec_code="9999",
        company="x",
        doc_id="D1",
        period_end="2026-03-31",
        previous_period_end="2025-03-31",
    )
    memo.business_risks = [
        Claim(text="原材料価格の上昇は収益を必ず押し下げる。", evidence_ids=["E1"])
    ]
    memo.overview = [Claim(text="従業員は百名である。", evidence_ids=["E2"])]
    memo.open_items = [Claim(text="取引先の依存度を確認する。", evidence_ids=[])]
    return pool, memo


def verdicts(*items: tuple[int, str, str]) -> str:
    return json.dumps({"verdicts": [{"index": i, "verdict": v, "reason": r} for i, v, r in items]})


def revisions(*items: tuple[int, str | None]) -> str:
    return json.dumps(
        {"revisions": [{"index": i, "text": t} for i, t in items]}, ensure_ascii=False
    )


async def test_全部支持されれば_書き直さず1回の検証で終わる() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend([verdicts((0, "supported", ""), (1, "supported", ""))])
    rounds = await run_verification(backend, pool, memo, max_rounds=2)
    assert len(backend.requests) == 1
    assert [r.round_no for r in rounds] == [0]
    assert memo.rejected == []
    assert {x.verdict for x in memo.verifications} == {"supported", "not_applicable"}


async def test_支持されない主張は_書き直されて_次の検証で支持されれば残る() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend(
        [
            verdicts((0, "supported", ""), (1, "unsupported", "必ずとは書かれていない")),
            revisions((0, "原材料等の価格が上昇した場合、経営成績に影響を及ぼす可能性がある。")),
            verdicts((0, "supported", "")),
        ]
    )
    rounds = await run_verification(backend, pool, memo, max_rounds=2)

    assert memo.business_risks[0].text.startswith("原材料等の価格が上昇した場合")
    assert memo.business_risks[0].evidence_ids == ["E1"]  # 出典は書き直しで変えない
    assert memo.rejected == []
    assert [r.round_no for r in rounds] == [0, 1]
    revise_prompt = backend.requests[1].messages[0].content
    assert "必ずとは書かれていない" in revise_prompt  # 判定の理由を渡す
    assert len(backend.requests) == 3


async def test_回数を使い切っても支持されない主張は_本文から外し_理由つきで記録する() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend(
        [
            verdicts((0, "supported", ""), (1, "unsupported", "言えない")),
            revisions((0, "やはり言えない文。")),
            verdicts((0, "partial", "まだ言い過ぎ")),
        ]
    )
    await run_verification(backend, pool, memo, max_rounds=1)

    assert memo.business_risks == []
    (r,) = memo.rejected
    assert r.section == "business_risks"
    assert any(i.kind == "verifier_failed" and "まだ言い過ぎ" in i.message for i in r.issues)


async def test_書き直しで本文がnullなら_その主張を外す() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend(
        [verdicts((0, "supported", ""), (1, "unsupported", "根拠なし")), revisions((0, None))]
    )
    await run_verification(backend, pool, memo, max_rounds=2)
    assert memo.business_risks == []
    assert len(memo.rejected) == 1
    assert len(backend.requests) == 2  # 残りが無いので、再検証は呼ばない


async def test_判断できない主張は_本文に残し_印をつける() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend(
        [verdicts((0, "supported", ""), (1, "cannot_judge", "表が崩れている"))]
    )
    await run_verification(backend, pool, memo, max_rounds=2)
    assert len(memo.business_risks) == 1
    assert any(x.verdict == "cannot_judge" for x in memo.verifications)
    assert len(backend.requests) == 1  # 書き直しでは解決しないので、書き直さない


async def test_回数0なら_書き直さず_失敗を外す() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend([verdicts((0, "supported", ""), (1, "partial", "言い過ぎ"))])
    await run_verification(backend, pool, memo, max_rounds=0)
    assert memo.business_risks == [] and len(memo.rejected) == 1
    assert len(backend.requests) == 1


async def test_書き直した主張も_数値などの機械検査を通す() -> None:
    pool, memo = _setup()
    backend = ScriptedBackend(
        [
            verdicts((0, "supported", ""), (1, "partial", "x")),
            revisions((0, "原材料価格の上昇は影響を与えうるので融資可能である。")),
            verdicts((0, "supported", "")),
        ]
    )
    await run_verification(backend, pool, memo, max_rounds=2)
    assert memo.business_risks == []  # 結論の語を含むので、検査で外れる
    assert any(i.kind == "forbidden_phrase" for r in memo.rejected for i in r.issues)

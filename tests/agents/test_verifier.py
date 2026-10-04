"""Verifier: 主張が、引いた証拠に支えられているかを LLM に判定させる。LLM は偽のバックエンド。"""

import json

import pytest

from agents.state import Claim, EvidencePool, PassageEvidence
from agents.verifier import ClaimVerdict, verify_claims
from llm.fake import ScriptedBackend
from llm.types import LLMBackendError
from retrieval.citations import SourceSpan

pytestmark = pytest.mark.anyio


def _pool() -> EvidencePool:
    pool = EvidencePool()
    quote = "原材料等の価格が大幅に上昇した場合には、経営成績に影響を及ぼす可能性があります。"
    pool.add(
        PassageEvidence(
            id="",
            sec_code="9999",
            company="サンプル",
            heading_path=["3 【事業等のリスク】"],
            spans=[SourceSpan(doc_id="D1", page=23, start=0, end=len(quote), quote=quote)],
        )
    )
    return pool


def reply(*verdicts: dict[str, object]) -> str:
    return json.dumps({"verdicts": list(verdicts)}, ensure_ascii=False)


def v(index: int, verdict: str, reason: str = "理由") -> dict[str, object]:
    return {"index": index, "verdict": verdict, "reason": reason}


async def test_主張ごとの判定と理由を返す() -> None:
    backend = ScriptedBackend([reply(v(0, "supported"), v(1, "unsupported", "本文に無い"))])
    claims = [
        Claim(text="原材料価格が上がると影響する。", evidence_ids=["E1"]),
        Claim(text="x", evidence_ids=["E1"]),
    ]

    out = await verify_claims(backend, _pool(), claims)

    assert [x.verdict for x in out] == ["supported", "unsupported"]
    assert out[1].reason == "本文に無い"


async def test_依頼には_主張と証拠の全文が入り_strongで呼ぶ() -> None:
    backend = ScriptedBackend([reply(v(0, "supported"))])
    await verify_claims(backend, _pool(), [Claim(text="主張A", evidence_ids=["E1"])])
    req = backend.requests[0]
    assert req.tier == "strong" and req.temperature == 0
    body = req.messages[0].content
    assert "主張A" in body and "[E1]" in body and "経営成績に影響を及ぼす可能性があります" in body
    assert "従わない" in req.system  # 証拠中の指示に従わない旨


async def test_出典のない主張は_LLMに送らず_対象外にする() -> None:
    backend = ScriptedBackend([reply(v(0, "supported"))])
    claims = [Claim(text="確認事項", evidence_ids=[]), Claim(text="主張", evidence_ids=["E1"])]

    out = await verify_claims(backend, _pool(), claims)

    assert [x.verdict for x in out] == ["not_applicable", "supported"]
    assert "主張" in backend.requests[0].messages[0].content
    assert "確認事項" not in backend.requests[0].messages[0].content


async def test_判定が返らなかった主張は_判断できないにする_支持と見なさない() -> None:
    backend = ScriptedBackend([reply(v(0, "supported"))])  # 1 番目の主張の判定が無い
    claims = [Claim(text="a", evidence_ids=["E1"]), Claim(text="b", evidence_ids=["E1"])]

    out = await verify_claims(backend, _pool(), claims)

    assert out[1].verdict == "cannot_judge"
    assert "返さなかった" in out[1].reason


async def test_範囲外や重複した番号は_無視する_最初の判定を採る() -> None:
    backend = ScriptedBackend([reply(v(5, "supported"), v(0, "unsupported"), v(0, "supported"))])
    out = await verify_claims(backend, _pool(), [Claim(text="a", evidence_ids=["E1"])])
    assert out[0].verdict == "unsupported"


async def test_知らない判定の値は_判断できないにする() -> None:
    backend = ScriptedBackend(
        [
            reply(
                v(0, "完璧"),
            ),
            reply(v(0, "完璧")),
        ]
    )
    out = await verify_claims(backend, _pool(), [Claim(text="a", evidence_ids=["E1"])])
    assert out[0].verdict == "cannot_judge"


async def test_形式を満たせなければ_1件の場合は判断できないにして_失敗を理由に残す() -> None:
    backend = ScriptedBackend(["だめ", "だめ"])
    out = await verify_claims(backend, _pool(), [Claim(text="a", evidence_ids=["E1"])])
    assert out[0].verdict == "cannot_judge" and "形式" in out[0].reason


async def test_多い主張は_まとめて複数回に分ける() -> None:
    claims = [Claim(text=f"主張{i}", evidence_ids=["E1"]) for i in range(5)]
    backend = ScriptedBackend(
        [reply(*[v(i, "supported") for i in range(3)]), reply(*[v(i, "partial") for i in range(2)])]
    )
    out = await verify_claims(backend, _pool(), claims, batch_size=3)
    assert [x.verdict for x in out] == ["supported"] * 3 + ["partial"] * 2
    assert len(backend.requests) == 2


async def test_バックエンドの失敗は_そのまま伝える() -> None:
    with pytest.raises(LLMBackendError):
        await verify_claims(
            ScriptedBackend([LLMBackendError("落ちた")]),
            _pool(),
            [Claim(text="a", evidence_ids=["E1"])],
        )


def test_判定の型() -> None:
    assert ClaimVerdict(index=0, verdict="supported", reason="").verdict == "supported"


async def test_まとめた判定が切れたら_半分に分けて再試行する() -> None:
    from llm.types import LLMResponse

    cut = LLMResponse(text='{"verdicts": [{"ind', backend="b", model="m", truncated=True)
    claims = [Claim(text=f"主張{i}", evidence_ids=["E1"]) for i in range(4)]
    backend = ScriptedBackend(
        [
            cut,
            reply(v(0, "supported"), v(1, "partial")),
            reply(v(0, "unsupported"), v(1, "supported")),
        ]
    )

    out = await verify_claims(backend, _pool(), claims)

    assert [x.verdict for x in out] == ["supported", "partial", "unsupported", "supported"]
    assert len(backend.requests) == 3
    assert "主張2" in backend.requests[2].messages[0].content

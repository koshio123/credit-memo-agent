"""ワーカー・Planner・パイプライン。LLM は用意した応答を返す偽のバックエンドに置き換える。"""

import json

import pytest

from agents.evidence import collect_metrics
from agents.pipeline import MemoResult, run_baseline, run_multi_agent
from agents.planner import DEFAULT_OVERVIEW_QUERIES, MANDATORY_RISK_QUERIES, Plan, make_plan
from agents.state import Claim, EvidencePool
from agents.workers import Context, screen, write_claims
from edinet_mcp.service import EdinetService
from llm.fake import ScriptedBackend
from llm.types import LLMBackendError

pytestmark = pytest.mark.anyio


def reply(**sections: list[dict[str, object]]) -> str:
    return json.dumps(sections, ensure_ascii=False)


def claim(text: str, *ids: str) -> dict[str, object]:
    return {"text": text, "evidence_ids": list(ids)}


def ids(service: EdinetService) -> tuple[str, str]:
    """(自己資本比率の証拠 ID, 最初の本文の証拠 ID)。パイプラインと同じ順で集めて求める。"""
    pool = EvidencePool()
    metrics = collect_metrics(service, "9999", pool)
    return metrics["equity_ratio.current"].id, f"E{len(pool.items) + 1}"


def ctx(service: EdinetService, backend: ScriptedBackend) -> Context:
    return Context(backend=backend, service=service, pool=EvidencePool(), sec_code="9999")


# ---- 検査による選別 ----


def test_選別は_出典の無い主張_存在しない出典_提示していない出典_結論の語を除く(
    service: EdinetService,
) -> None:
    c = ctx(service, ScriptedBackend([]))
    metrics = collect_metrics(service, "9999", c.pool)
    equity, sales = metrics["equity_ratio.current"].id, metrics["net_sales.current"].id

    accepted, rejected = screen(
        [
            Claim(text="自己資本比率は50.0%である。", evidence_ids=[equity]),
            Claim(text="事業は堅調である。", evidence_ids=[]),
            Claim(text="売上高は2,000円である。", evidence_ids=[sales]),  # 提示していない証拠
            Claim(text="売上は増えた。", evidence_ids=["E999"]),
            Claim(text="自己資本比率は50.0%で融資可能である。", evidence_ids=[equity]),
        ],
        c.pool,
        {equity},
        section="financial_findings",
    )

    assert [a.text for a in accepted] == ["自己資本比率は50.0%である。"]
    assert len(rejected) == 4
    assert {r.section for r in rejected} == {"financial_findings"}
    reasons = {i.kind for r in rejected for i in r.issues}
    assert {"no_evidence", "unknown_evidence", "forbidden_phrase", "unoffered_evidence"} <= reasons


def test_数値の警告だけの主張は_残す_メモに警告として残る(service: EdinetService) -> None:
    c = ctx(service, ScriptedBackend([]))
    m = collect_metrics(service, "9999", c.pool)["equity_ratio.current"]
    accepted, flagged = screen(
        [Claim(text="自己資本比率は80.1%である。", evidence_ids=[m.id])],
        c.pool,
        {m.id},
        section="financial_findings",
    )
    assert len(accepted) == 1
    assert [i.severity for i in flagged[0].issues] == ["warning"]


# ---- 主張を書かせる ----


async def test_主張を書かせる_証拠の一覧が依頼に入り_JSONが主張になる(
    service: EdinetService,
) -> None:
    backend = ScriptedBackend([reply(claims=[claim("自己資本比率は50.0%である。", "E1")])])
    c = ctx(service, backend)
    metrics = collect_metrics(service, "9999", c.pool)
    evidence = [metrics["equity_ratio.current"]]

    claims = await write_claims(c, task="財務の所見を書く", evidence=evidence, max_claims=3)

    assert [x.text for x in claims] == ["自己資本比率は50.0%である。"]
    sent = backend.requests[0]
    body = sent.messages[0].content
    assert "[E1]" in body and "50.0%" in body and "財務の所見を書く" in body
    assert "融資" in sent.system  # 可否を述べない指示
    assert backend.requests[0].temperature == 0


async def test_書かせた主張の件数は_上限で切る(service: EdinetService) -> None:
    backend = ScriptedBackend([reply(claims=[claim(f"主張{i}", "E1") for i in range(10)])])
    claims = await write_claims(ctx(service, backend), task="t", evidence=[], max_claims=3)
    assert len(claims) == 3


# ---- Planner ----


async def test_Plannerは_問いを決める_必須の問いは必ず含める(service: EdinetService) -> None:
    backend = ScriptedBackend(
        [json.dumps({"overview_queries": ["主力製品"], "risk_queries": ["為替の影響"]})]
    )
    plan = await make_plan(
        ctx(service, backend), company="サンプル工業", industry="機械", summary=""
    )
    assert "主力製品" in plan.overview_queries
    assert "為替の影響" in plan.risk_queries
    for required in MANDATORY_RISK_QUERIES:
        assert required in plan.risk_queries


async def test_Plannerの出力が不正なら_既定の問いに戻る(service: EdinetService) -> None:
    backend = ScriptedBackend(["だめ", "まだだめ"])
    plan = await make_plan(ctx(service, backend), company="x", industry="y", summary="")
    assert plan.overview_queries == list(DEFAULT_OVERVIEW_QUERIES)
    assert plan.used_default
    assert len(backend.requests) == 2  # 作り直しは 1 回


async def test_Plannerの問いは_数と長さを制限し_重複を除く(service: EdinetService) -> None:
    many = [f"問い{i}" for i in range(20)]
    backend = ScriptedBackend(
        [json.dumps({"overview_queries": many + many, "risk_queries": ["a" * 500, ""]})]
    )
    plan = await make_plan(ctx(service, backend), company="x", industry="y", summary="")
    assert len(plan.overview_queries) <= 4
    assert len(set(plan.overview_queries)) == len(plan.overview_queries)
    assert all(0 < len(q) <= 80 for q in plan.risk_queries)


def test_Planは_問いを持つ() -> None:
    assert Plan(overview_queries=["a"], risk_queries=["b"]).used_default is False


# ---- パイプライン ----


async def test_マルチエージェントは_5回の呼び出しでメモを作る(service: EdinetService) -> None:
    eq, ps = ids(service)
    plan = json.dumps({"overview_queries": ["事業の内容"], "risk_queries": ["為替の影響"]})
    backend = ScriptedBackend(
        [
            plan,
            reply(claims=[claim("当社グループは機械を製造している。", ps)]),
            reply(claims=[claim("自己資本比率は50.0%で、水準は標準である。", eq)]),
            reply(claims=[claim("原材料価格が高騰する可能性がある。", ps)]),
            reply(
                positives=[claim("自己資本比率の水準は標準である。", eq)],
                negatives=[claim("原材料価格の高騰の影響を受けうる。", ps)],
                open_items=[claim("取引先への依存度の割合は資料から確認できなかった。")],
            ),
        ]
    )
    result = await run_multi_agent(service, backend, "9999")

    assert isinstance(result, MemoResult)
    assert result.calls == 5
    assert len(backend.requests) == 5
    memo = result.memo
    assert memo.sec_code == "9999" and memo.company == "サンプル工業"
    assert memo.financial_findings and memo.business_risks and memo.overview
    assert memo.positives and memo.negatives and memo.open_items
    assert memo.policy_rows  # 内規照合の表はコードが作る
    assert memo.rejected == []
    for section in (memo.overview, memo.financial_findings, memo.positives):
        for c in section:
            assert all(e in result.pool for e in c.evidence_ids)


async def test_ベースラインは_1回の呼び出しで全部書く(service: EdinetService) -> None:
    eq, ps = ids(service)
    backend = ScriptedBackend(
        [
            reply(
                overview=[claim("機械を製造している。", ps)],
                financial_findings=[claim("自己資本比率は50.0%である。", eq)],
                business_risks=[claim("原材料価格が高騰する可能性がある。", ps)],
                positives=[claim("自己資本比率の水準は標準である。", eq)],
                negatives=[claim("原材料価格の影響を受けうる。", ps)],
                open_items=[claim("依存度は確認できなかった。")],
            )
        ]
    )
    result = await run_baseline(service, backend, "9999")
    assert result.calls == 1
    assert result.memo.financial_findings and result.memo.open_items


async def test_検査で除外された主張は_メモに入れず_除外の記録に残す(service: EdinetService) -> None:
    eq, _ = ids(service)
    backend = ScriptedBackend(
        [
            reply(
                overview=[claim("出典のない主張。")],
                financial_findings=[claim("自己資本比率は50.0%である。", eq)],
                business_risks=[],
                positives=[],
                negatives=[],
                open_items=[],
            )
        ]
    )
    result = await run_baseline(service, backend, "9999")
    assert result.memo.overview == []
    assert [r.claim.text for r in result.memo.rejected] == ["出典のない主張。"]
    assert result.memo.rejected[0].section == "overview"


async def test_バックエンドの失敗は_そのまま伝える(service: EdinetService) -> None:
    backend = ScriptedBackend([LLMBackendError("落ちた")])
    with pytest.raises(LLMBackendError):
        await run_baseline(service, backend, "9999")


async def test_使用量を記録する(service: EdinetService) -> None:
    from llm.types import LLMResponse, Usage

    empty = reply(
        overview=[],
        financial_findings=[],
        business_risks=[],
        positives=[],
        negatives=[],
        open_items=[],
    )
    resp = LLMResponse(
        text=empty, backend="b", model="m", usage=Usage(input_tokens=100, output_tokens=20)
    )
    result = await run_baseline(service, ScriptedBackend([resp]), "9999")
    assert (result.input_tokens, result.output_tokens) == (100, 20)


async def test_全ページの本文を取れる(service: EdinetService) -> None:
    from tests.edinet_fakes import PAGES

    assert service.all_pages("9999") == PAGES

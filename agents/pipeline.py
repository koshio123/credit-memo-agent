"""メモの生成: マルチエージェントと、単一エージェントのベースライン。

どちらも同じ証拠（数値はコードの算定結果、本文は検索結果）から、同じ検査を通して作る。
違いは、LLM の使い方だけ（アブレーションで比べる）。
  - ベースライン: 固定の問いで集めた証拠を、1 回の呼び出しで全部の節に書かせる。
  - マルチエージェント: Planner が問いを決め、Worker が節ごとに書き、Drafter が論点をまとめる。
"""

from collections.abc import Coroutine, Sequence
from dataclasses import dataclass

from agents.evidence import LEVEL_RATIOS, collect_metrics, collect_passages
from agents.memo import DraftOut, Failure, Flagged, MemoDraft, MemoOut
from agents.planner import DEFAULT_OVERVIEW_QUERIES, DEFAULT_RISK_QUERIES, make_plan
from agents.policy import GoingConcernResult, build_policy_rows, check_going_concern
from agents.prompts import (
    FINANCIAL_TASK,
    OVERVIEW_TASK,
    RISK_TASK,
    SYSTEM,
    draft_prompt,
    memo_prompt,
    render_evidence,
)
from agents.state import Claim, Evidence, EvidencePool, MetricEvidence, PassageEvidence
from agents.workers import Context, screen, write_claims
from edinet_mcp.models import CompanyInfo
from edinet_mcp.service import EdinetMcpError, EdinetService
from llm.structured import StructuredOutputError, complete_structured
from llm.types import LLMBackend, LLMRequest, Message
from llm.usage import CountingBackend

# 財務の所見に渡す数値の証拠（内規照合の記録などは除く）
_FINANCIAL_PREFIXES = (
    "equity_ratio",
    "current_ratio",
    "operating_margin",
    "interest_coverage",
    "debt_repayment_years",
    "sales_growth",
    "net_sales",
    "operating_income",
    "net_income_attributable_to_owners",
    "total_assets",
    "net_assets",
    "interest_bearing_debt",
    "debt_due_within_1y_ratio",
    "consecutive_operating_loss",
    "sales_drop",
)
_PASSAGES_PER_QUERY = 3


@dataclass
class MemoResult:
    memo: MemoDraft
    pool: EvidencePool
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    mode: str
    metrics: dict[str, MetricEvidence]


def _unique(evidence: Sequence[PassageEvidence]) -> list[PassageEvidence]:
    seen: dict[str, PassageEvidence] = {}
    for item in evidence:
        seen.setdefault(item.id, item)
    return list(seen.values())


_SUMMARY_KEYS = (*LEVEL_RATIOS, "sales_growth")  # Planner に見せる財務指標


def _summary(metrics: dict[str, MetricEvidence]) -> str:
    lines = [
        f"- {m.label}: {m.display}" + (f"（{m.level}）" if m.level else "")
        for key, m in metrics.items()
        if key.endswith(".current") and key.split(".")[0] in _SUMMARY_KEYS
    ]
    return "\n".join(lines)


def _company_info(service: EdinetService, sec_code: str) -> CompanyInfo:
    for info in service.list_companies():
        if info.sec_code == sec_code.strip():
            return info
    raise EdinetMcpError(f"証券コード {sec_code!r} は対象外です")


def _new_memo(info: CompanyInfo) -> MemoDraft:
    return MemoDraft(
        sec_code=info.sec_code,
        company=info.name,
        doc_id=info.doc_id,
        period_end=info.period_end,
        previous_period_end=info.previous_period_end,
    )


def _finish_policy(
    memo: MemoDraft,
    service: EdinetService,
    pool: EvidencePool,
    metrics: dict[str, MetricEvidence],
) -> GoingConcernResult:
    going_concern = check_going_concern(memo.doc_id, service.all_pages(memo.sec_code))
    memo.policy_rows = build_policy_rows(metrics, pool, memo.doc_id, going_concern)
    memo.going_concern_signal = going_concern.status == "signal_found"
    return going_concern


def _financial_evidence(metrics: dict[str, MetricEvidence]) -> list[MetricEvidence]:
    return [m for key, m in metrics.items() if key.split(".")[0] in _FINANCIAL_PREFIXES]


def _evidence_ids(claims: Sequence[Claim]) -> set[str]:
    return {e for c in claims for e in c.evidence_ids}


async def run_baseline(service: EdinetService, backend: LLMBackend, sec_code: str) -> MemoResult:
    """単一エージェント: 固定の問いで集めた証拠を、1 回の呼び出しで全部の節に書かせる。"""
    counting = CountingBackend(backend)
    pool = EvidencePool()
    ctx = Context(counting, service, pool, sec_code)
    ctx.metrics = collect_metrics(service, sec_code, pool)
    queries = [*DEFAULT_OVERVIEW_QUERIES, *DEFAULT_RISK_QUERIES]
    found = collect_passages(service, sec_code, queries, pool, k=_PASSAGES_PER_QUERY)
    passages = _unique([p for ps in found.values() for p in ps])
    evidence: list[Evidence] = [*_financial_evidence(ctx.metrics), *passages]

    request = LLMRequest(
        system=SYSTEM,
        messages=[Message(role="user", content=memo_prompt(render_evidence(evidence)))],
        tier="standard",
        max_tokens=ctx.max_tokens * 2,
    )
    memo = _new_memo(_company_info(service, sec_code))
    try:
        out = await complete_structured(counting, request, MemoOut)
    except StructuredOutputError as e:
        memo.failures.append(Failure(section="all", message=str(e)[:300]))
        out = MemoOut()
    offered = {e.id for e in evidence}
    for section in (
        "overview",
        "financial_findings",
        "business_risks",
        "positives",
        "negatives",
        "open_items",
    ):
        accepted, flagged = screen(
            getattr(out, section)[:5],
            pool,
            offered,
            section,
            require_evidence=section != "open_items",
        )
        setattr(memo, section, accepted)
        _record(memo, flagged)
    _finish_policy(memo, service, pool, ctx.metrics)
    return _result(memo, pool, counting, "baseline", ctx.metrics)


async def run_multi_agent(service: EdinetService, backend: LLMBackend, sec_code: str) -> MemoResult:
    """Planner が問いを決め、Worker が節ごとに書き、Drafter が論点と確認事項をまとめる。"""
    counting = CountingBackend(backend)
    pool = EvidencePool()
    ctx = Context(counting, service, pool, sec_code)
    ctx.metrics = collect_metrics(service, sec_code, pool)
    info = _company_info(service, sec_code)
    memo = _new_memo(info)

    plan = await make_plan(ctx, memo.company, info.industry, _summary(ctx.metrics))

    # 概要。各ワーカーは互いに独立だが、順に呼ぶ（claude_code は利用枠を使うので並行にしない。
    # 並行にするかは、バックエンドごとに選べるようにする余地として残す）
    overview_found = collect_passages(
        service, sec_code, plan.overview_queries, pool, k=_PASSAGES_PER_QUERY
    )
    overview_evidence = _unique([p for ps in overview_found.values() for p in ps])
    claims = await _guarded(
        memo, "overview", write_claims(ctx, OVERVIEW_TASK, overview_evidence, max_claims=4)
    )
    memo.overview, flagged = screen(claims, pool, {e.id for e in overview_evidence}, "overview")
    _record(memo, flagged)

    # 財務の所見
    financial_evidence = _financial_evidence(ctx.metrics)
    claims = await _guarded(
        memo,
        "financial_findings",
        write_claims(ctx, FINANCIAL_TASK, financial_evidence, max_claims=6),
    )
    memo.financial_findings, flagged = screen(
        claims, pool, {e.id for e in financial_evidence}, "financial_findings"
    )
    _record(memo, flagged)

    # 事業リスク
    risk_found = collect_passages(service, sec_code, plan.risk_queries, pool, k=_PASSAGES_PER_QUERY)
    risk_evidence = _unique([p for ps in risk_found.values() for p in ps])
    claims = await _guarded(
        memo, "business_risks", write_claims(ctx, RISK_TASK, risk_evidence, max_claims=6)
    )
    memo.business_risks, flagged = screen(
        claims, pool, {e.id for e in risk_evidence}, "business_risks"
    )
    _record(memo, flagged)

    # 内規照合（コード）
    _finish_policy(memo, service, pool, ctx.metrics)

    # 論点と確認事項（Drafter）
    prior = [*memo.overview, *memo.financial_findings, *memo.business_risks]
    cited = _evidence_ids(prior)
    offered_evidence: list[Evidence] = [pool.get(i) for i in sorted(cited, key=_id_number)]
    prior_text = "\n".join(f"- {c.text} [{', '.join(c.evidence_ids)}]" for c in prior)
    request = LLMRequest(
        system=SYSTEM,
        messages=[
            Message(
                role="user",
                content=draft_prompt(
                    prior_text or "（主張なし）", render_evidence(offered_evidence)
                ),
            )
        ],
        tier="standard",
        max_tokens=ctx.max_tokens,
    )
    try:
        drafted = await complete_structured(counting, request, DraftOut)
    except StructuredOutputError as e:
        section = "positives/negatives/open_items"
        memo.failures.append(Failure(section=section, message=str(e)[:300]))
        drafted = DraftOut()
    for section in ("positives", "negatives", "open_items"):
        accepted, flagged = screen(
            getattr(drafted, section)[:5],
            pool,
            cited,
            section,
            require_evidence=section != "open_items",
        )
        setattr(memo, section, accepted)
        _record(memo, flagged)
    return _result(memo, pool, counting, "multi_agent", ctx.metrics)


async def _guarded(
    memo: MemoDraft, section: str, call: Coroutine[object, object, list[Claim]]
) -> list[Claim]:
    """形式を満たせない出力は、その節の失敗として記録し、他の節の生成は続ける。"""
    try:
        return await call
    except StructuredOutputError as e:
        memo.failures.append(Failure(section=section, message=str(e)[:300]))
        return []


def _id_number(evidence_id: str) -> int:
    return int(evidence_id[1:])


def _record(memo: MemoDraft, flagged: list[Flagged]) -> None:
    for item in flagged:
        if any(i.severity == "error" for i in item.issues):
            memo.rejected.append(item)
        else:
            memo.warnings.append(item)


def _result(
    memo: MemoDraft,
    pool: EvidencePool,
    counting: CountingBackend,
    mode: str,
    metrics: dict[str, MetricEvidence],
) -> MemoResult:
    return MemoResult(
        memo=memo,
        pool=pool,
        calls=counting.calls,
        cached_calls=counting.cached_calls,
        input_tokens=counting.input_tokens,
        output_tokens=counting.output_tokens,
        mode=mode,
        metrics=metrics,
    )

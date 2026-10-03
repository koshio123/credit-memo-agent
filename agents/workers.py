"""ワーカー共通の部品: 文脈、主張を書かせる呼び出し、検査による選別。"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from agents.checks import check_claims
from agents.memo import ClaimsOut, Flagged, IssueRecord
from agents.prompts import SYSTEM, claims_task, render_evidence
from agents.state import Claim, EvidencePool, MetricEvidence, PassageEvidence
from edinet_mcp.service import EdinetService
from llm.structured import complete_structured
from llm.types import LLMBackend, LLMRequest, Message, Tier


@dataclass
class Context:
    backend: LLMBackend
    service: EdinetService
    pool: EvidencePool
    sec_code: str
    max_tokens: int = 3000
    metrics: dict[str, MetricEvidence] = field(default_factory=dict[str, MetricEvidence])


def screen(
    claims: Sequence[Claim],
    pool: EvidencePool,
    offered: set[str],
    section: str,
    require_evidence: bool = True,
) -> tuple[list[Claim], list[Flagged]]:
    """検査で error のある主張を除く。warning だけの主張は残す（警告は rejected でなく警告に記録）。

    offered は、その呼び出しで LLM に提示した証拠の ID。提示していない証拠を引用した主張は除く
    （LLM が他の節の証拠を勝手に使うのを防ぐ）。

    Args:
        claims: LLM が書いた主張。
        pool: 証拠の集まり。
        offered: LLM に提示した証拠の ID。
        section: 主張の節の名前（記録用）。
        require_evidence: False なら、出典が無いことを問題にしない。

    Returns:
        (残す主張, 検査に引っかかった主張)。後者には除いたものと、警告つきで残したものの両方が入る。
    """
    accepted: list[Claim] = []
    flagged: list[Flagged] = []
    issues_by_index: dict[int, list[IssueRecord]] = {}
    for issue in check_claims(claims, pool):
        if issue.kind == "no_evidence" and not require_evidence:
            continue
        issues_by_index.setdefault(issue.claim_index, []).append(IssueRecord.of(issue))
    for index, claim in enumerate(claims):
        records = issues_by_index.setdefault(index, [])
        for evidence_id in claim.evidence_ids:
            if evidence_id in pool and evidence_id not in offered:
                records.append(
                    IssueRecord(
                        kind="unoffered_evidence",
                        severity="error",
                        message=f"この節に提示していない証拠です: {evidence_id}",
                    )
                )
    for index, claim in enumerate(claims):
        records = issues_by_index[index]
        if any(r.severity == "error" for r in records):
            flagged.append(Flagged(section=section, claim=claim, issues=records))
        else:
            accepted.append(claim)
            if records:
                flagged.append(Flagged(section=section, claim=claim, issues=records))
    return accepted, flagged


async def write_claims(
    ctx: Context,
    task: str,
    evidence: Sequence[MetricEvidence | PassageEvidence],
    max_claims: int = 5,
    tier: Tier = "standard",
) -> list[Claim]:
    """証拠だけを渡して、出典つきの主張を書かせる。件数は上限で切る。

    Args:
        ctx: ワーカー共通の文脈。
        task: 何を書かせるか。
        evidence: LLM に渡す証拠。
        max_claims: 主張の最大件数。
        tier: 使うモデルの段階。

    Returns:
        書かれた主張（最大 max_claims 件）。

    Raises:
        StructuredOutputError: 出力が形式を満たさなかったとき。
    """
    request = LLMRequest(
        system=SYSTEM,
        messages=[
            Message(
                role="user",
                content=claims_task(task, render_evidence(evidence), max_claims),
            )
        ],
        tier=tier,
        max_tokens=ctx.max_tokens,
    )
    out = await complete_structured(ctx.backend, request, ClaimsOut)
    return out.claims[:max_claims]

"""検証と差し戻しの状態機械。

  VERIFYING --(失敗なし、または回数を使い切った)--> DONE
  VERIFYING --(失敗あり、回数が残っている)--------> REVISING
  REVISING  -------------------------------------> VERIFYING

遷移はコード（`next_stage`）で決まり、LLM は遷移を決めない。失敗は「一部だけ／支持しない」の判定。
「判断できない」は書き直しでは解決しないので、失敗に数えず、本文に残して印をつける。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from agents.checks import check_claims
from agents.memo import Flagged, IssueRecord, MemoDraft, RoundRecord, VerificationRecord
from agents.prompts import SYSTEM, render_evidence
from agents.state import CLAIM_SECTIONS, Claim, EvidencePool
from agents.verifier import ClaimVerdict, Verdict, verify_claims
from llm.structured import StructuredOutputError, complete_structured
from llm.types import LLMBackend, LLMRequest, Message

FAILING: tuple[Verdict, ...] = ("partial", "unsupported")


class Stage(StrEnum):
    VERIFYING = "verifying"
    REVISING = "revising"
    DONE = "done"


def next_stage(stage: Stage, failing: int, round_no: int, max_rounds: int) -> Stage:
    """次の段階を決める。

    Args:
        stage: 今の段階。
        failing: 今の検証で「一部だけ／支持しない」だった主張の数。
        round_no: 今のラウンド番号（最初の検証は 0）。
        max_rounds: 書き直しの最大回数。

    Returns:
        次の段階。
    """
    if stage is Stage.VERIFYING:
        return Stage.REVISING if failing > 0 and round_no < max_rounds else Stage.DONE
    if stage is Stage.REVISING:
        return Stage.VERIFYING
    return Stage.DONE


class _Revision(BaseModel):
    index: int
    text: str | None = None  # None は「証拠の範囲では支持できる主張にならない」


class _ReviseOut(BaseModel):
    revisions: list[_Revision] = Field(default_factory=list[_Revision])


@dataclass
class _Item:
    section: str
    claim: Claim
    verdict: ClaimVerdict | None = None
    round_no: int = 0


def _revise_prompt(pool: EvidencePool, failing: Sequence[_Item]) -> str:
    cited: list[str] = []
    for item in failing:
        for evidence_id in item.claim.evidence_ids:
            if evidence_id in pool and evidence_id not in cited:
                cited.append(evidence_id)
    claims = "\n".join(
        f"{n}. {item.claim.text}\n   検証者の指摘: {item.verdict.reason if item.verdict else ''}"
        for n, item in enumerate(failing)
    )
    return (
        "【依頼】\n次の主張は、検証者から「証拠に十分に支えられていない」と指摘されました。"
        "指摘を踏まえ、同じ証拠だけから言える範囲に書き直してください。"
        "言い過ぎを削り、証拠にない推測を除く。数値は証拠の表記のままにする。"
        "証拠の範囲では支持できる主張にならない場合は、text を null にする。\n\n"
        f"【主張と指摘】\n{claims}\n\n"
        f"【証拠】\n{render_evidence([pool.get(i) for i in cited], quote_limit=None)}\n\n"
        '【出力の形】\n{"revisions": [{"index": 0, "text": "書き直した主張 または null"}]}'
    )


def _reject(memo: MemoDraft, item: _Item, kind: str, message: str) -> None:
    memo.rejected.append(
        Flagged(
            section=item.section,
            claim=item.claim,
            issues=[IssueRecord(kind=kind, severity="error", message=message)],
        )
    )


async def _revise(
    backend: LLMBackend, pool: EvidencePool, memo: MemoDraft, failing: list[_Item]
) -> dict[int, _Item]:
    """失敗した主張を書き直させる。

    Returns:
        書き直せた主張（機械検査も通ったもの）。元の主張の id → 新しい主張。
    """
    request = LLMRequest(
        system=SYSTEM,
        messages=[Message(role="user", content=_revise_prompt(pool, failing))],
        tier="standard",
        max_tokens=2000,
    )
    try:
        out = await complete_structured(backend, request, _ReviseOut)
    except StructuredOutputError as e:
        for item in failing:
            _reject(memo, item, "verifier_failed", f"書き直せなかった（形式）: {str(e)[:100]}")
        return {}
    by_index = {r.index: r for r in out.revisions if 0 <= r.index < len(failing)}
    revised: dict[int, _Item] = {}
    for n, item in enumerate(failing):
        reason = item.verdict.reason if item.verdict else ""
        rev = by_index.get(n)
        if rev is None or not rev.text or not rev.text.strip():
            _reject(
                memo,
                item,
                "verifier_failed",
                f"{item.verdict and item.verdict.verdict}: {reason}（書き直せなかった）",
            )
            continue
        new_claim = Claim(text=rev.text.strip(), evidence_ids=item.claim.evidence_ids)
        issues = check_claims([new_claim], pool)
        records = [IssueRecord.of(i) for i in issues]
        if any(r.severity == "error" for r in records):
            memo.rejected.append(Flagged(section=item.section, claim=new_claim, issues=records))
            continue
        if records:
            memo.warnings.append(Flagged(section=item.section, claim=new_claim, issues=records))
        revised[id(item)] = _Item(item.section, new_claim)
    return revised


def _count(verdicts: Sequence[ClaimVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


async def run_verification(
    backend: LLMBackend, pool: EvidencePool, memo: MemoDraft, max_rounds: int = 2
) -> list[RoundRecord]:
    """メモの主張を検証し、失敗した主張を最大 max_rounds 回まで書き直す。

    最終的に「一部だけ／支持しない」の主張は、メモの本文から外して除外の記録に残す。
    判定の結果と各ラウンドの記録は memo に書き込む。

    Args:
        backend: LLM のバックエンド。
        pool: 証拠の集まり。
        memo: 検証するメモ（主張の節は書き換えられる）。
        max_rounds: 書き直しの最大回数。0 なら検証だけして、失敗した主張を外す。

    Returns:
        ラウンドごとの記録。

    Raises:
        LLMBackendError: バックエンドの呼び出しが失敗したとき。
    """
    items = [_Item(key, claim) for key, _ in CLAIM_SECTIONS for claim in getattr(memo, key)]
    pending = items
    rounds: list[RoundRecord] = []
    round_no = 0
    stage = Stage.VERIFYING
    while True:
        verdicts = await verify_claims(backend, pool, [i.claim for i in pending])
        for item, verdict in zip(pending, verdicts, strict=True):
            item.verdict, item.round_no = verdict, round_no
        rounds.append(
            RoundRecord(round_no=round_no, n_verified=len(pending), counts=_count(verdicts))
        )
        failing = [i for i in pending if i.verdict and i.verdict.verdict in FAILING]
        stage = next_stage(stage, len(failing), round_no, max_rounds)
        if stage is Stage.DONE:
            break
        # 書き直す。書き直せた主張だけを、次のラウンドで検証する（他の判定はそのまま持ち越す）
        revised = await _revise(backend, pool, memo, failing)
        stage = next_stage(stage, 0, round_no, max_rounds)
        round_no += 1
        failed_ids = {id(i) for i in failing}
        # 書き直せた主張は元の位置に置き換え、書き直せなかった主張は取り除く（節の中の順序を保つ）
        items = [
            revised.get(id(i), i) for i in items if id(i) not in failed_ids or id(i) in revised
        ]
        pending = list(revised.values())
        if not pending:
            break

    for item in items:
        if item.verdict and item.verdict.verdict in FAILING:
            _reject(memo, item, "verifier_failed", f"{item.verdict.verdict}: {item.verdict.reason}")
    kept = [i for i in items if not (i.verdict and i.verdict.verdict in FAILING)]
    for key, _ in CLAIM_SECTIONS:
        setattr(memo, key, [i.claim for i in kept if i.section == key])
    memo.verifications = [
        VerificationRecord(
            section=i.section,
            claim_text=i.claim.text,
            verdict=i.verdict.verdict if i.verdict else "not_applicable",
            reason=i.verdict.reason if i.verdict else "",
            round_no=i.round_no,
        )
        for i in kept
    ]
    memo.rounds = rounds
    return rounds

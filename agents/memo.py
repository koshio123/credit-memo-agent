"""メモの下書きの型。LLM が書く部分は主張（出典つき）だけで、表はコードが作る。"""

from pydantic import BaseModel, ConfigDict, Field

from agents.checks import Issue
from agents.policy import PolicyRow
from agents.state import Claim


class IssueRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    severity: str
    message: str

    @classmethod
    def of(cls, issue: Issue) -> IssueRecord:
        return cls(kind=issue.kind, severity=issue.severity, message=issue.message)


class Flagged(BaseModel):
    """検査に引っかかった主張。error は除外され、warning はメモに残って警告として記録される。"""

    model_config = ConfigDict(frozen=True)

    section: str
    claim: Claim
    issues: list[IssueRecord]


class Failure(BaseModel):
    """LLM の出力が形式を満たせず、書けなかった節。他の節の生成は続ける。"""

    model_config = ConfigDict(frozen=True)

    section: str  # all は、1回の呼び出しで全部の節を書くベースラインの失敗
    message: str


class ClaimsOut(BaseModel):
    """LLM の出力: 主張の一覧。"""

    claims: list[Claim] = Field(default_factory=list[Claim])


class PlanOut(BaseModel):
    """Planner の出力: 有報で調べる問い。"""

    overview_queries: list[str] = Field(default_factory=list[str])
    risk_queries: list[str] = Field(default_factory=list[str])


class DraftOut(BaseModel):
    """Drafter の出力: 与信判断上の論点と、確認が必要な事項。"""

    positives: list[Claim] = Field(default_factory=list[Claim])
    negatives: list[Claim] = Field(default_factory=list[Claim])
    open_items: list[Claim] = Field(default_factory=list[Claim])


class MemoOut(DraftOut):
    """単一エージェントの出力: 全部の節。"""

    overview: list[Claim] = Field(default_factory=list[Claim])
    financial_findings: list[Claim] = Field(default_factory=list[Claim])
    business_risks: list[Claim] = Field(default_factory=list[Claim])


class MemoDraft(BaseModel):
    sec_code: str
    company: str
    doc_id: str
    period_end: str
    previous_period_end: str
    overview: list[Claim] = Field(default_factory=list[Claim])
    financial_findings: list[Claim] = Field(default_factory=list[Claim])
    business_risks: list[Claim] = Field(default_factory=list[Claim])
    positives: list[Claim] = Field(default_factory=list[Claim])
    negatives: list[Claim] = Field(default_factory=list[Claim])
    open_items: list[Claim] = Field(default_factory=list[Claim])
    policy_rows: list[PolicyRow] = Field(default_factory=list[PolicyRow])
    going_concern_signal: bool = False  # 第13条: 継続企業の前提に関する記載が見つかったか
    rejected: list[Flagged] = Field(default_factory=list[Flagged])  # 検査で除外した主張
    warnings: list[Flagged] = Field(default_factory=list[Flagged])  # 残したが警告のある主張
    failures: list[Failure] = Field(default_factory=list[Failure])  # 形式を満たせず書けなかった節

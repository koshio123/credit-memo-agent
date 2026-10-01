"""Planner: 会社の業種と財務指標の特徴から、有報で調べる問いを決める。

必須の問い（返済能力に影響する事項、取引先依存。規程 第12・14条）は、LLM の出力にかかわらず入れる。
LLM の出力が使えないときは、既定の問いで続ける（調査が止まらないように）。
"""

from dataclasses import dataclass, field

from agents.memo import PlanOut
from agents.prompts import PLAN_PROMPT, SYSTEM
from agents.workers import Context
from llm.structured import StructuredOutputError, complete_structured
from llm.types import LLMRequest, Message

DEFAULT_OVERVIEW_QUERIES: tuple[str, ...] = (
    "事業の内容 主要な製品 サービス",
    "従業員の状況 従業員数 平均年齢",
    "大株主の状況 株主構成",
)
MANDATORY_RISK_QUERIES: tuple[str, ...] = (
    "事業等のリスク 返済能力に影響する事項",
    "主要な取引先 売上高に占める割合 依存度",
)
DEFAULT_RISK_QUERIES: tuple[str, ...] = (
    *MANDATORY_RISK_QUERIES,
    "原材料価格 為替 金利の変動 リスク",
    "法規制 訴訟 偶発債務 リスク",
)
_MAX_OVERVIEW = 5
_MAX_RISK = 6
_MAX_QUERY_CHARS = 80


@dataclass(frozen=True)
class Plan:
    overview_queries: list[str]
    risk_queries: list[str]
    used_default: bool = field(default=False)


def _clean(queries: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for query in queries:
        text = " ".join(query.split())
        if 0 < len(text) <= _MAX_QUERY_CHARS and text not in seen:
            seen.append(text)
    return seen[:limit]


def default_plan() -> Plan:
    return Plan(list(DEFAULT_OVERVIEW_QUERIES), list(DEFAULT_RISK_QUERIES), used_default=True)


async def make_plan(ctx: Context, company: str, industry: str, summary: str) -> Plan:
    request = LLMRequest(
        system=SYSTEM,
        messages=[
            Message(
                role="user",
                content=PLAN_PROMPT.format(company=company, industry=industry, summary=summary),
            )
        ],
        tier="strong",
        max_tokens=600,
    )
    try:
        out = await complete_structured(ctx.backend, request, PlanOut)
    except StructuredOutputError:
        return default_plan()
    # 概要の必須の問いも、リスクと同様に、Planner の出力にかかわらず入れる（実機で、Planner が
    # 事業の内容・従業員・株主の問いを外して、概要が空になった）
    specific_overview = _clean(out.overview_queries, _MAX_OVERVIEW - len(DEFAULT_OVERVIEW_QUERIES))
    overview = _clean([*specific_overview, *DEFAULT_OVERVIEW_QUERIES], _MAX_OVERVIEW)
    specific = _clean(out.risk_queries, _MAX_RISK - len(MANDATORY_RISK_QUERIES))
    risk = _clean([*specific, *MANDATORY_RISK_QUERIES], _MAX_RISK)
    return Plan(overview, risk)

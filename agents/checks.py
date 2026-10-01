"""主張の構造の検査（決定的）。

ここで行うのは、出典の有無と存在、結論を示す語、数値の照合まで。主張が証拠の内容に
支えられているか（意味）の検証は、LLM を使う Verifier（W4）の仕事で、ここでは行わない。
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agents.state import Claim, EvidencePool

# 与信の可否・推奨を示す語（テンプレートの記載ルール 4、規程 第3条）。語の一致だけの見張りで、
# 言い換えは検出できない。「承認」のように、事実の説明にも使う語は入れない
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "融資可能",
    "融資不可",
    "融資すべき",
    "見送るべき",
    "懸念なし",
    "懸念はない",
    "問題なし",
    "問題はない",
    "推奨",
)

# 数値と、直後の単位。小さい整数（「2期連続」の 2 など）まで照合すると誤検出が多いので、
# 小数・カンマ区切り・3 桁以上・％や倍の付く数だけを照合する。年（2026年）は照合しない
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)(\s*[%％倍年])?")

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    kind: str
    severity: Severity
    claim_index: int
    message: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def find_forbidden(text: str) -> list[str]:
    """主張の中の、結論を示す語（全角半角・空白の違いは無視する）。"""
    normalized = _normalize(text)
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in normalized]


def _numbers(text: str, *, only_checked: bool) -> set[str]:
    found: set[str] = set()
    for match in _NUMBER.finditer(unicodedata.normalize("NFKC", text)):
        token = match.group(1).replace(",", "")
        unit = (match.group(2) or "").strip()
        if only_checked:
            if unit == "年":
                continue
            is_checked = "." in token or len(token) >= 3 or unit in ("%", "％", "倍")
            if not is_checked:
                continue
        found.add(token)
    return found


def check_claims(claims: Sequence[Claim], pool: EvidencePool) -> list[Issue]:
    issues: list[Issue] = []
    for index, claim in enumerate(claims):
        if not claim.evidence_ids:
            issues.append(Issue("no_evidence", "error", index, "出典がありません"))
        unknown = [e for e in claim.evidence_ids if e not in pool]
        if unknown:
            issues.append(
                Issue(
                    "unknown_evidence", "error", index, f"存在しない出典です: {', '.join(unknown)}"
                )
            )
        for phrase in find_forbidden(claim.text):
            issues.append(Issue("forbidden_phrase", "error", index, f"結論を示す語です: {phrase}"))

        cited = [pool.get(e) for e in claim.evidence_ids if e in pool]
        if not cited:
            continue
        available: set[str] = set()
        for evidence in cited:
            available |= _numbers(evidence.text, only_checked=False)
        missing = sorted(_numbers(claim.text, only_checked=True) - available)
        if missing:
            issues.append(
                Issue(
                    "number_not_in_evidence",
                    "warning",
                    index,
                    f"引用した証拠に無い数値です: {', '.join(missing)}",
                )
            )
    return issues

"""規程への照合（第8〜11条・第13条）。規則だけで行い、LLM を使わない。

- 第8〜11条は、コードが算定した数値の証拠（`agents/evidence.py`）の水準・該当を転記する。
- 第13条（継続企業の前提）は、有報の本文を語句で調べる。**「記載を確認できなかった」までしか言えず、
  「問題がない」とは断定しない**。語句の一致の見張りで、言い換えは検出できない。
  実データの20社は問題のない会社ばかりで、記載ありの検出は合成した文でしか確かめていない。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from agents.state import EvidencePool, MetricEvidence, PassageEvidence
from retrieval.citations import SourceSpan

GoingConcernStatus = Literal["signal_found", "no_signal"]

# 継続企業の前提に問題があるときにだけ出る言い回し（行の途中の改行は取り除いて照合する）。
# 監査報告書の定型文（「…重要な不確実性が認められるかどうか結論付ける」「…認められる場合は」）は除く
_SIGNALS: tuple[re.Pattern[str], ...] = (
    re.compile(r"重要事象等[^。]{0,20}(存在|認められ)"),
    re.compile(r"疑義を生じさせるような事象又は状況[^。]{0,20}(が存在|が認められ(?!る場合|るか))"),
    re.compile(r"重要な不確実性が(存在|認められ(?!る場合|るか))"),
)
# 見出し（その語だけの行）の直後に内容があれば、記載あり。「該当事項はありません」なら記載なし。
# 文中の言及（「…重要事象等は存在しておりません」）は見出しではない
_HEADING_LINES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"^[ \t]*{heading}[ \t]*$", re.M)
    for heading in (r"\(継続企業の前提に関する事項\)", r"継続企業の前提に関する重要事象等")
)
_NONE = "該当事項はありません"
# 一致の直後に続く否定（「存在しておりません」「認められない」など）は、記載ありとしない
_NEGATED_TAIL = re.compile(r"^(し(て(い|おり))?(ない|ません|ませんでした)|せず|ない|ません|ず)")
_NEXT_HEADING = re.compile(r"^\(.{1,40}\)")


@dataclass(frozen=True)
class GoingConcernResult:
    status: GoingConcernStatus
    spans: list[SourceSpan] = field(default_factory=list[SourceSpan])
    n_pages: int = 0  # 調べたページ数


def _flatten(text: str) -> tuple[str, list[int]]:
    """改行を取り除いた本文と、各文字の元の位置を返す。

    Args:
        text: ページの本文。

    Returns:
        (改行を除いた本文, 各文字の元の本文での位置)。
    """
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        if ch != "\n":
            chars.append(ch)
            origin.append(i)
    return "".join(chars), origin


def _sentence(flat: str, start: int, end: int) -> tuple[int, int]:
    """一致箇所を含む文の範囲（句点まで）。

    Args:
        flat: 改行を除いた本文。
        start: 一致の開始位置。
        end: 一致の終了位置（含まない）。

    Returns:
        (文の開始位置, 文の終了位置（含まない）)。
    """
    left = flat.rfind("。", 0, start) + 1
    right = flat.find("。", end)
    return left, (len(flat) if right < 0 else right + 1)


def _heading_spans(doc_id: str, page_number: int, text: str) -> list[SourceSpan]:
    """見出しの行の後に、「該当事項はありません」以外の内容があれば、見出しから最初の文末まで。

    Args:
        doc_id: 書類ID。
        page_number: ページ番号（1始まり）。
        text: ページの本文。

    Returns:
        記載ありと見なす箇所の出典スパン。
    """
    spans: list[SourceSpan] = []
    for pattern in _HEADING_LINES:
        for match in pattern.finditer(text):
            rest, origin = _flatten(text[match.end() :])
            body = rest[:80].lstrip()
            if not body or body.startswith(_NONE) or _NEXT_HEADING.match(body):
                continue
            sentence_end = rest.find("。")
            last = len(rest) - 1 if sentence_end < 0 else sentence_end
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            end = match.end() + origin[last] + 1
            spans.append(
                SourceSpan(
                    doc_id=doc_id,
                    page=page_number,
                    start=start,
                    end=end,
                    quote=text[start:end],
                )
            )
    return spans


def check_going_concern(doc_id: str, pages: Sequence[str]) -> GoingConcernResult:
    """継続企業の前提に関する記載を、語句で探す（第13条）。

    Args:
        doc_id: 書類ID。
        pages: ページごとの本文。

    Returns:
        記載の有無と、見つかった箇所の出典スパン。
    """
    spans: list[SourceSpan] = []
    for page_number, text in enumerate(pages, start=1):
        flat, origin = _flatten(text)
        ranges: list[tuple[int, int]] = []
        for pattern in _SIGNALS:
            for match in pattern.finditer(flat):
                if _NEGATED_TAIL.match(flat[match.end() :]):
                    continue
                ranges.append(_sentence(flat, match.start(), match.end()))
        for left, right in sorted(set(ranges)):
            start, end = origin[left], origin[right - 1] + 1
            spans.append(
                SourceSpan(
                    doc_id=doc_id,
                    page=page_number,
                    start=start,
                    end=end,
                    quote=text[start:end],
                )
            )
        spans += _heading_spans(doc_id, page_number, text)
    spans = _drop_nested(spans)
    return GoingConcernResult("signal_found" if spans else "no_signal", spans, len(pages))


def _drop_nested(spans: list[SourceSpan]) -> list[SourceSpan]:
    """他のスパンに含まれるスパンを除く（同じ文を重複して引用しない）。

    Args:
        spans: 出典スパン。

    Returns:
        重複と包含を除いたスパン（元の順）。
    """
    kept: list[SourceSpan] = []
    for span in spans:
        inside = any(
            other is not span
            and other.page == span.page
            and other.start <= span.start
            and span.end <= other.end
            and (other.start, other.end) != (span.start, span.end)
            for other in spans
        )
        if not inside and span not in kept:
            kept.append(span)
    return kept


# ---- 内規照合の表 ----


@dataclass(frozen=True)
class PolicyRow:
    clause: str
    check: str
    result: str
    evidence_ids: list[str]


_LEVEL_RATIOS = (
    "equity_ratio",
    "current_ratio",
    "operating_margin",
    "interest_coverage",
    "debt_repayment_years",
)


def build_policy_rows(
    metrics: dict[str, MetricEvidence],
    pool: EvidencePool,
    doc_id: str,
    going_concern: GoingConcernResult,
) -> list[PolicyRow]:
    """内規照合の表の行。結果は転記と規則の判定だけで、評価や結論は含まない。

    Args:
        metrics: 「項目.期」から数値の証拠への対応。
        pool: 証拠の集まり。第13条の証拠をここに追加する。
        doc_id: 書類ID。
        going_concern: 継続企業の前提に関する記載の検索結果。

    Returns:
        第8〜11条・第13条の照合の行。
    """
    rows: list[PolicyRow] = []
    scrutiny: list[MetricEvidence] = []
    for name in _LEVEL_RATIOS:
        m = metrics[f"{name}.current"]
        rows.append(PolicyRow("第8条", m.label, m.level or "算定不能", [m.id]))
        if m.level == "要精査":
            scrutiny.append(m)
    if scrutiny:
        names = "、".join(m.label for m in scrutiny)
        rows.append(
            PolicyRow(
                "第9条",
                "要精査水準の指標の原因の分析",
                f"要精査水準: {names}（原因の分析を記載する）",
                [m.id for m in scrutiny],
            )
        )
    for key, clause in (("consecutive_operating_loss", "第10条"), ("sales_drop", "第11条")):
        m = metrics[f"{key}.current"]
        rows.append(PolicyRow(clause, m.label, m.display, [m.id]))

    rows.append(_going_concern_row(pool, doc_id, going_concern))
    return rows


def _going_concern_row(pool: EvidencePool, doc_id: str, result: GoingConcernResult) -> PolicyRow:
    """第13条の行を作る。記載があれば本文の証拠を、無ければ検索の記録を、証拠に加える。

    Args:
        pool: 証拠を追加する集まり。
        doc_id: 書類ID。
        result: 継続企業の前提に関する記載の検索結果。

    Returns:
        第13条の照合の行。
    """
    check = "継続企業の前提に関する記載"
    if result.status == "signal_found":
        passage = pool.add(
            PassageEvidence(
                id="",
                sec_code="",
                company="",
                heading_path=["継続企業の前提に関する記載（語句による検索）"],
                spans=result.spans,
            )
        )
        return PolicyRow(
            "第13条", check, "記載あり（内容の確認が必要。最優先の確認事項）", [passage.id]
        )
    record = pool.add(
        MetricEvidence(
            id="",
            sec_code="",
            company="",
            doc_id=doc_id,
            label="継続企業の前提に関する記載の検索（第13条）",
            period="current",
            display="確認できなかった",
            value=Decimal(0),
            origin="search_record",
            basis=(
                f"有価証券報告書の全{result.n_pages}ページを語句で検索した結果、"
                "重要事象等・重要な疑義・重要な不確実性が存在する旨の記載は見つからなかった。"
                "語句の一致による検索であり、言い換えは検出できない"
            ),
        )
    )
    return PolicyRow("第13条", check, "該当する記載を語句の検索では確認できなかった", [record.id])

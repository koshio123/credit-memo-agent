"""L1 評価: PDF から抽出した値を、XBRL の正解データと突き合わせて、項目ごとの正答率を出す。

方式（pdfplumber、Docling など）を比べるための共通の物差し。対象は、財務諸表にそのまま載っている
11 項目（合成項目の売上債権・棚卸資産・仕入債務、有利子負債は含めない）。

レポートには数値を載せない（会社・項目・結果の種類・理由の符号だけ）。EDINET 由来の値を、
公開するリポジトリに入れないため。
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from finance.ratios import PeriodFinancials
from ingest.pdf_baseline import ExtractedValue

# 評価する項目（PeriodFinancials の項目名）-> 表示名
FIELDS: dict[str, str] = {
    "total_assets": "総資産",
    "net_assets": "純資産",
    "current_assets": "流動資産",
    "current_liabilities": "流動負債",
    "net_sales": "売上高",
    "operating_income": "営業利益",
    "interest_income": "受取利息",
    "dividend_income": "受取配当金",
    "interest_expense": "支払利息",
    "net_income_attributable_to_owners": "親会社株主に帰属する当期純利益",
    "depreciation": "減価償却費",
}


class Outcome(StrEnum):
    MATCH = "一致"
    BOTH_ABSENT = "両方なし"  # 正解も PDF も「行なし」。正解
    MISMATCH = "不一致"  # 値が違う
    MISSING = "未抽出"  # 正解はあるのに PDF から読めなかった
    SPURIOUS = "過剰"  # 正解は無い（または0）のに値を読んだ

    @property
    def correct(self) -> bool:
        """正解として数える結果か。

        Returns:
            一致、または両方なしなら True。
        """
        return self in (Outcome.MATCH, Outcome.BOTH_ABSENT)


def _absent(truth: Decimal | None) -> bool:
    """正解が「行なし」または 0（XBRL の「－」）。

    Args:
        truth: 正解の値。

    Returns:
        None または 0 なら True。
    """
    return truth is None or truth == 0


def classify(extracted: Decimal | None, truth: Decimal | None) -> Outcome:
    """抽出した値を、正解と比べて分類する。

    Args:
        extracted: PDF から抽出した値。読めなければ None。
        truth: XBRL の正解の値。

    Returns:
        一致・両方なし・不一致・未抽出・過剰のいずれか。
    """
    if extracted is None:
        return Outcome.BOTH_ABSENT if _absent(truth) else Outcome.MISSING
    if _absent(truth):
        return Outcome.MATCH if extracted == 0 else Outcome.SPURIOUS
    return Outcome.MATCH if extracted == truth else Outcome.MISMATCH


@dataclass(frozen=True)
class CompanyScore:
    sec_code: str
    name: str
    outcomes: dict[str, Outcome]
    reasons: dict[str, str]  # 読めなかった項目 -> 理由の符号


def score_company(
    sec_code: str, name: str, items: dict[str, ExtractedValue], truth: PeriodFinancials
) -> CompanyScore:
    """1社の抽出結果を、項目ごとに採点する。

    Args:
        sec_code: 証券コード。
        name: 会社名。
        items: 項目名から抽出結果への対応。
        truth: 正解の財務データ。

    Returns:
        項目ごとの結果と、読めなかった理由。
    """
    outcomes: dict[str, Outcome] = {}
    reasons: dict[str, str] = {}
    for field in FIELDS:
        item = items[field]
        outcomes[field] = classify(item.value, getattr(truth, field))
        if item.reason:
            reasons[field] = item.reason
    return CompanyScore(sec_code, name, outcomes, reasons)


@dataclass(frozen=True)
class FieldSummary:
    correct: int
    total: int
    counts: dict[Outcome, int]


@dataclass(frozen=True)
class Summary:
    per_field: dict[str, FieldSummary]
    correct: int
    total: int


def summarize(scores: list[CompanyScore]) -> Summary:
    """全社の採点を、項目ごとに集計する。

    Args:
        scores: 会社ごとの採点。

    Returns:
        項目ごとの正答数・結果の内訳と、全体の正答数。
    """
    per_field: dict[str, FieldSummary] = {}
    for field in FIELDS:
        counts = {o: 0 for o in Outcome}
        for s in scores:
            counts[s.outcomes[field]] += 1
        correct = sum(n for o, n in counts.items() if o.correct)
        per_field[field] = FieldSummary(correct, len(scores), counts)
    return Summary(
        per_field,
        correct=sum(f.correct for f in per_field.values()),
        total=sum(f.total for f in per_field.values()),
    )


def render_report(method: str, scores: list[CompanyScore]) -> str:
    """L1 評価のレポートを Markdown にする。

    Args:
        method: 評価した方式の名前。
        scores: 会社ごとの採点。

    Returns:
        項目ごとの集計と、正解にならなかったものの表を含む Markdown。
    """
    summary = summarize(scores)
    lines = [
        f"# L1 評価: {method}",
        "",
        f"対象: {len(scores)} 社（当期の有価証券報告書）× {len(FIELDS)} 項目。"
        "正解データは XBRL（財務データ CSV）から作った値。",
        "",
        f"**全体の正答: {summary.correct}/{summary.total}**（一致と「両方なし」を正解とする）",
        "",
        "| 項目 | 正答 | 一致 | 両方なし | 不一致 | 未抽出 | 過剰 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for field, label in FIELDS.items():
        f = summary.per_field[field]
        c = f.counts
        lines.append(
            f"| {label} | {f.correct}/{f.total} | {c[Outcome.MATCH]} | {c[Outcome.BOTH_ABSENT]}"
            f" | {c[Outcome.MISMATCH]} | {c[Outcome.MISSING]} | {c[Outcome.SPURIOUS]} |"
        )

    failures = [
        (s, field, s.outcomes[field])
        for s in scores
        for field in FIELDS
        if not s.outcomes[field].correct
    ]
    lines += ["", "## 正解にならなかったもの", ""]
    if not failures:
        lines.append("なし")
    else:
        lines += ["| 会社 | 項目 | 結果 | 理由 |", "| --- | --- | --- | --- |"]
        for s, field, outcome in failures:
            reason = s.reasons.get(field, "")
            lines.append(
                f"| {s.sec_code} {s.name} | {FIELDS[field]} | {outcome.value} | {reason} |"
            )
    return "\n".join(lines) + "\n"

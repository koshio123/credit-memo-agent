"""証拠の収集: サービス層の結果から、数値と本文の証拠を作り、ID を振る。

数値は XBRL とコードの算定結果（`finance/ratios.py`）から、本文は検索結果から作る。
LLM はここで作られた証拠の ID を選ぶだけで、数値も引用文も書かない。
"""

import re
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal

from agents.state import EvidencePool, MetricEvidence, PassageEvidence
from edinet_mcp.models import FinancialsResult, Period, RatiosResult
from edinet_mcp.service import EdinetService
from finance.ratios import FlagResult, PeriodFinancials, RatioResult

# 指標 -> 算式の入力項目（PeriodFinancials の項目名）。XBRL の項目名・PDF のページの絞り込み用
_DEBT_FIELDS = (
    "short_term_borrowings",
    "commercial_paper",
    "current_portion_long_term_borrowings",
    "current_portion_bonds",
    "bonds",
    "long_term_borrowings",
    "lease_obligations",
)
RATIO_INPUTS: dict[str, tuple[str, ...]] = {
    "equity_ratio": ("net_assets", "total_assets"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "operating_margin": ("operating_income", "net_sales"),
    "interest_coverage": (
        "operating_income",
        "interest_income",
        "dividend_income",
        "interest_expense",
    ),
    "debt_repayment_years": (
        *_DEBT_FIELDS,
        "net_income_attributable_to_owners",
        "depreciation",
    ),
    "sales_growth": ("net_sales",),
}
LEVEL_RATIOS = (
    "equity_ratio",
    "current_ratio",
    "operating_margin",
    "interest_coverage",
    "debt_repayment_years",
)
FIGURES: dict[str, str] = {
    "net_sales": "売上高",
    "operating_income": "営業利益",
    "net_income_attributable_to_owners": "親会社株主に帰属する当期純利益",
    "total_assets": "総資産",
    "net_assets": "純資産",
}
DEBT_LABELS: dict[str, str] = {
    "short_term_borrowings": "短期借入金",
    "commercial_paper": "コマーシャル・ペーパー",
    "current_portion_long_term_borrowings": "1年内返済予定の長期借入金",
    "current_portion_bonds": "1年内償還予定の社債",
    "bonds": "社債",
    "long_term_borrowings": "長期借入金",
    "lease_obligations": "リース債務",
}
_DUE_WITHIN_1Y = (
    "short_term_borrowings",
    "commercial_paper",
    "current_portion_long_term_borrowings",
    "current_portion_bonds",
)
_PERIOD_LABEL = {"current": "直近期", "previous": "前期"}
_VALUE_AT_END = re.compile(r"=\s*(-?[\d,]+(?:\.\d+)?)\s*(?:%|倍|年)?\s*$")


def format_yen(value: Decimal) -> str:
    """金額の表記。100万円以上は百万円（四捨五入）、それ未満は円。

    Args:
        value: 金額（円）。

    Returns:
        例: "139,657百万円"、"85,000円"。
    """
    if abs(value) >= 1_000_000:
        millions = (value / Decimal(1_000_000)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return f"{millions:,}百万円"
    return f"{value:,.0f}円"


def debt_due_within_1y(f: PeriodFinancials) -> tuple[Decimal | None, str]:
    """1年以内に返済する有利子負債の割合（%）と、算式・入力値（規程 第15条）。

    リース債務は XBRL で流動と固定が分かれていない（合算）ので、1年以内には含めない。

    Args:
        f: 1期分の連結財務データ。

    Returns:
        (割合（%）, 算式と入力値)。算定できないときは割合が None で、理由が算式の欄に入る。
    """
    total = f.interest_bearing_debt
    if total is None:
        return None, "算定不能（有利子負債の内訳がどれも無い）"
    if total <= 0:
        return None, f"算定不能（有利子負債がゼロ以下: {total:,.0f}円）"
    due = sum((getattr(f, name) or Decimal(0) for name in _DUE_WITHIN_1Y), Decimal(0))
    value = due / total * 100
    basis = (
        f"1年以内に返済する有利子負債 {due:,.0f} ÷ 有利子負債 {total:,.0f} = {value:.1f}%"
        "（リース債務は流動と固定を分けられないため、1年以内には含めない）"
    )
    return value, basis


def _ratio_display(result: RatioResult) -> str:
    """財務比率の、文章に書く表記を作る。計算根拠の末尾の数値をそのまま使う。

    Args:
        result: 財務比率の算定結果。

    Returns:
        例: "76.4%"。算定不能なら "算定不能"。
    """
    if result.value is None:
        return "算定不能"
    match = _VALUE_AT_END.search(result.basis)
    number = match.group(1) if match else f"{result.value:,.1f}"
    return f"{number}{result.unit}"


def _flag_display(flag: FlagResult) -> str:
    """留意事項の判定の表記を作る。

    Args:
        flag: 留意事項の判定。

    Returns:
        "該当"、"非該当"、"判定不能" のいずれか。
    """
    if flag.applies is None:
        return "判定不能"
    return "該当" if flag.applies else "非該当"


def _inputs(
    fields: Sequence[str], financials: FinancialsResult
) -> tuple[dict[str, str], list[int]]:
    """算式の入力項目について、XBRL の項目名と PDF のページを集める。

    Args:
        fields: 入力項目の名前（PeriodFinancials の項目名）。
        financials: 財務データの取得結果。

    Returns:
        (入力項目 -> XBRL の項目名, 入力値が載っている PDF のページ（昇順）)。
    """
    xbrl = {f: financials.provenance[f] for f in fields if f in financials.provenance}
    pages = sorted({financials.pdf_pages[f] for f in fields if f in financials.pdf_pages})
    return xbrl, pages


def collect_metrics(
    service: EdinetService, sec_code: str, pool: EvidencePool
) -> dict[str, MetricEvidence]:
    """財務比率・規程の留意事項・主要な金額の証拠を、当期と前期について作る。

    キーは「項目.期」（例: equity_ratio.current）。

    Args:
        service: EDINET のサービス層。
        sec_code: 証券コード。
        pool: 証拠を追加する集まり。

    Returns:
        「項目.期」から数値の証拠への対応。
    """
    periods: tuple[Period, Period] = ("current", "previous")
    financials = {p: service.get_financials(sec_code, p) for p in periods}
    ratios: dict[str, RatiosResult] = {p: service.get_ratios(sec_code, p) for p in periods}
    out: dict[str, MetricEvidence] = {}

    def add(key: str, evidence: MetricEvidence) -> None:
        """証拠を集まりに追加し、キーに結びつけて記録する。

        Args:
            key: 「項目.期」のキー。
            evidence: 追加する証拠。
        """
        out[key] = pool.add(evidence)

    for period in periods:
        fin = financials[period]
        report = ratios[period].ratios
        names = (*LEVEL_RATIOS, "sales_growth") if period == "current" else LEVEL_RATIOS
        for name in names:
            result: RatioResult = getattr(report, name)
            xbrl, pages = _inputs(RATIO_INPUTS[name], fin)
            add(
                f"{name}.{period}",
                MetricEvidence(
                    id="",
                    sec_code=fin.sec_code,
                    company=fin.company,
                    doc_id=fin.doc_id,
                    label=result.name,
                    period=period,
                    display=_ratio_display(result),
                    value=result.value,
                    level=result.level.value if result.level else None,
                    basis=result.basis,
                    xbrl_items=xbrl,
                    pdf_pages=pages,
                ),
            )
        for field, label in FIGURES.items():
            amount = getattr(fin.financials, field)
            if amount is None:
                continue
            xbrl, pages = _inputs((field,), fin)
            add(
                f"{field}.{period}",
                MetricEvidence(
                    id="",
                    sec_code=fin.sec_code,
                    company=fin.company,
                    doc_id=fin.doc_id,
                    label=f"{label}（{_PERIOD_LABEL[period]}）",
                    period=period,
                    display=format_yen(amount),
                    value=amount,
                    basis=f"XBRL {xbrl.get(field, field)} = {amount:,.0f}円",
                    xbrl_items=xbrl,
                    pdf_pages=pages,
                ),
            )

    cur = financials["current"]
    debt = cur.financials
    parts = [(n, getattr(debt, n)) for n in DEBT_LABELS if getattr(debt, n) is not None]
    if debt.interest_bearing_debt is not None:
        xbrl_debt, pages_debt = _inputs([n for n, _ in parts], cur)
        add(
            "interest_bearing_debt.current",
            MetricEvidence(
                id="",
                sec_code=cur.sec_code,
                company=cur.company,
                doc_id=cur.doc_id,
                label="有利子負債（直近期）",
                period="current",
                display=format_yen(debt.interest_bearing_debt),
                value=debt.interest_bearing_debt,
                basis="有利子負債 = "
                + " ＋ ".join(f"{DEBT_LABELS[n]} {v:,.0f}" for n, v in parts)
                + f" = {debt.interest_bearing_debt:,.0f}円",
                xbrl_items=xbrl_debt,
                pdf_pages=pages_debt,
            ),
        )
    for name, amount in parts:
        xbrl_one, pages_one = _inputs((name,), cur)
        add(
            f"{name}.current",
            MetricEvidence(
                id="",
                sec_code=cur.sec_code,
                company=cur.company,
                doc_id=cur.doc_id,
                label=f"{DEBT_LABELS[name]}（直近期）",
                period="current",
                display=format_yen(amount),
                value=amount,
                basis=f"XBRL {xbrl_one.get(name, name)} = {amount:,.0f}円",
                xbrl_items=xbrl_one,
                pdf_pages=pages_one,
            ),
        )
    due_value, due_basis = debt_due_within_1y(debt)
    xbrl_due, pages_due = _inputs(list(DEBT_LABELS), cur)
    add(
        "debt_due_within_1y_ratio.current",
        MetricEvidence(
            id="",
            sec_code=cur.sec_code,
            company=cur.company,
            doc_id=cur.doc_id,
            label="1年以内に返済する有利子負債の割合（第15条）",
            period="current",
            display="算定不能" if due_value is None else f"{due_value:.1f}%",
            value=due_value,
            basis=due_basis,
            xbrl_items=xbrl_due,
            pdf_pages=pages_due,
        ),
    )
    for flag_name in ("consecutive_operating_loss", "sales_drop"):
        flag: FlagResult = getattr(ratios["current"].ratios, flag_name)
        add(
            f"{flag_name}.current",
            MetricEvidence(
                id="",
                sec_code=cur.sec_code,
                company=cur.company,
                doc_id=cur.doc_id,
                label=f"{flag.name}（{flag.clause}）",
                period="current",
                display=_flag_display(flag),
                value=None,
                basis=flag.basis,
            ),
        )
    return out


def collect_passages(
    service: EdinetService,
    sec_code: str,
    queries: Sequence[str],
    pool: EvidencePool,
    k: int = 5,
) -> dict[str, list[PassageEvidence]]:
    """問いごとに検索し、本文の証拠を作る。同じ箇所が複数の問いで見つかっても、証拠は 1 つ。

    Args:
        service: EDINET のサービス層。
        sec_code: 証券コード。
        queries: 検索の問い。
        pool: 証拠を追加する集まり。
        k: 問いごとに取る件数。

    Returns:
        問いから、見つかった本文の証拠（上位から）への対応。
    """
    result: dict[str, list[PassageEvidence]] = {}
    for query in queries:
        evidences: list[PassageEvidence] = []
        for passage in service.search_filings(query, sec_code, k=k):
            evidence = pool.find_passage(passage.spans) or pool.add(
                PassageEvidence(
                    id="",
                    sec_code=passage.sec_code,
                    company=passage.company,
                    heading_path=passage.heading_path,
                    spans=passage.spans,
                )
            )
            evidences.append(evidence)
        result[query] = evidences
    return result

"""内規への照合（規程 第8〜11・13条）。規則だけで行い、LLM を使わない。"""

from decimal import Decimal

from agents.evidence import collect_metrics
from agents.policy import PolicyRow, build_policy_rows, check_going_concern
from agents.state import EvidencePool, MetricEvidence
from edinet_mcp.service import EdinetService
from retrieval.citations import verify_span

# 監査報告書の定型文。行の途中で改行される（実データの書式）。継続企業の前提に問題があるわけではない
AUDIT_BOILERPLATE = (
    "・ 経営者が継続企業を前提として連結財務諸表を作成することが適切であるかどうか、また、\n"
    "入手した監査証拠に基\n"
    "づき、継続企業の前提に重要な疑義を生じさせるような事象又は状況に関して重要な不確実性が認められるかど\n"
    "うか結論付ける。継続企業の前提に関する重要な不確実性が認められる場合は、監査報告書において連結財務諸表\n"
    "の注記事項に注意を喚起すること、又は重要な不確実性に関する連結財務諸表の注記事項が適切でない場合は、"
    "除外事項付意見を表明することが求められている。\n"
)


# ---- 継続企業の前提（第13条） ----


def test_監査報告書の定型文だけなら_記載なし() -> None:
    pages = ["表紙", AUDIT_BOILERPLATE]
    result = check_going_concern("D1", pages)
    assert result.status == "no_signal"
    assert result.spans == []


def test_注記の見出しが_該当事項なし_なら記載なし() -> None:
    pages = ["(継続企業の前提に関する事項)\n該当事項はありません。\n(連結の範囲に関する事項)\n39社"]
    assert check_going_concern("D1", pages).status == "no_signal"


def test_注記に内容があれば_記載あり_出典スパンつき() -> None:
    page = (
        "(継続企業の前提に関する事項)\n"
        "当社グループは、前連結会計年度に引き続き、営業損失を計上しており、継続企業の前提に\n"
        "重要な疑義を生じさせるような事象又は状況が存在しております。\n"
        "(連結の範囲に関する事項)\n"
    )
    result = check_going_concern("D1", ["表紙", page])
    assert result.status == "signal_found"
    assert result.spans
    assert all(verify_span(s, ["表紙", page]) for s in result.spans)
    assert result.spans[0].page == 2
    assert "重要な疑義" in "".join(s.quote for s in result.spans)


def test_重要事象等が存在する旨の記載は_行をまたいでも検出する() -> None:
    page = (
        "3 【事業等のリスク】\n継続企業の前提に関する重要事象等\n"
        "当社グループには、継続企業の前提に関する重要\n事象等が存在しています。\n"
    )
    result = check_going_concern("D1", [page])
    assert result.status == "signal_found"
    assert all(verify_span(s, [page]) for s in result.spans)


def test_重要な不確実性が存在する旨の記載を検出する() -> None:
    page = "継続企業の前提に関する重要な不確実性が存在するため、注意を喚起します。"
    assert check_going_concern("D1", [page]).status == "signal_found"


def test_重要事象等の見出しが_該当事項なし_なら記載なし() -> None:
    page = "継続企業の前提に関する重要事象等\n該当事項はありません。\n"
    assert check_going_concern("D1", [page]).status == "no_signal"


def test_複数ページの記載は_ページごとにスパンを返す() -> None:
    pages = [
        "継続企業の前提に関する重要な不確実性が存在します。",
        "別の話題",
        "継続企業の前提に重要な疑義を生じさせるような事象又は状況が存在しています。",
    ]
    result = check_going_concern("D1", pages)
    assert [s.page for s in result.spans] == [1, 3]


# ---- 内規照合の表 ----


def _metrics(service: EdinetService) -> tuple[EvidencePool, dict[str, MetricEvidence]]:
    pool = EvidencePool()
    return pool, collect_metrics(service, "9999", pool)


def test_第8条は_指標ごとに水準と根拠の証拠IDを示す(service: EdinetService) -> None:
    pool, metrics = _metrics(service)
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", ["表紙"]))

    by_check = {r.check: r for r in rows if r.clause == "第8条"}
    equity = by_check["自己資本比率"]
    assert equity.result == "標準"
    assert equity.evidence_ids == [metrics["equity_ratio.current"].id]
    # 算定不能の指標は、水準を判定せず、算定不能と示す（第8条第2項）
    assert by_check["インタレスト・カバレッジ・レシオ"].result == "算定不能"


def test_第10条と第11条は_該当か非該当かを示す(service: EdinetService) -> None:
    pool, metrics = _metrics(service)
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", ["表紙"]))
    by_clause = {r.clause: r for r in rows}
    assert by_clause["第10条"].result == "非該当"
    assert by_clause["第11条"].result == "非該当"
    assert by_clause["第10条"].evidence_ids


def test_第13条は_検索の記録を証拠にする_記載なしは語句の検索による旨を示す(
    service: EdinetService,
) -> None:
    pool, metrics = _metrics(service)
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", ["表紙", "本文"]))
    row = next(r for r in rows if r.clause == "第13条")
    assert "確認できなかった" in row.result
    assert "断定" not in row.result
    evidence = pool.get(row.evidence_ids[0])
    assert evidence.kind == "metric"
    assert "2ページ" in evidence.text  # 何ページを検索したか


def test_第13条で記載があれば_要確認として本文の証拠を示す(service: EdinetService) -> None:
    pool, metrics = _metrics(service)
    page = "継続企業の前提に関する重要な不確実性が存在します。"
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", [page]))
    row = next(r for r in rows if r.clause == "第13条")
    assert row.result.startswith("記載あり")
    assert pool.get(row.evidence_ids[0]).kind == "passage"


def test_要精査水準の指標があれば_第9条の行を足す(service: EdinetService) -> None:
    pool, metrics = _metrics(service)
    weak = metrics["current_ratio.current"].model_copy(update={"level": "要精査"})
    metrics = {**metrics, "current_ratio.current": weak}
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", ["表紙"]))
    row = next(r for r in rows if r.clause == "第9条")
    assert "流動比率" in row.check or "流動比率" in row.result
    assert isinstance(row, PolicyRow)


def test_結論の語を含まない(service: EdinetService) -> None:
    from agents.checks import find_forbidden

    pool, metrics = _metrics(service)
    rows = build_policy_rows(metrics, pool, "D1", check_going_concern("D1", ["表紙"]))
    for row in rows:
        assert find_forbidden(f"{row.check}{row.result}") == []
    assert Decimal(0) == 0

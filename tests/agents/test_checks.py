"""主張の構造の検査（決定的）。意味的に証拠が主張を支えるかは、W4 の Verifier の仕事。"""

from decimal import Decimal

from agents.checks import FORBIDDEN_PHRASES, Issue, check_claims, find_forbidden
from agents.state import Claim, EvidencePool, MetricEvidence, PassageEvidence
from retrieval.citations import SourceSpan


def _pool() -> EvidencePool:
    pool = EvidencePool()
    quote = "原材料価格が高騰した場合、売上原価が1,250百万円増加する可能性があります。"
    pool.add(
        PassageEvidence(
            id="",
            sec_code="9999",
            company="サンプル",
            heading_path=["第2 【事業の状況】", "3 【事業等のリスク】"],
            spans=[SourceSpan(doc_id="D1", page=2, start=0, end=len(quote), quote=quote)],
        )
    )
    pool.add(
        MetricEvidence(
            id="",
            sec_code="9999",
            company="サンプル",
            doc_id="D1",
            label="自己資本比率",
            period="current",
            display="76.4%",
            value=Decimal("76.4"),
            basis="純資産 500 ÷ 総資産 1,000 × 100",
            xbrl_items={"net_assets": "NetAssets"},
            pdf_pages=[60],
        )
    )
    return pool


def kinds(issues: list[Issue]) -> list[str]:
    return [i.kind for i in issues]


# ---- 証拠の ID はコードが振る ----


def test_証拠のIDはE1から順にコードが振る() -> None:
    pool = _pool()
    assert list(pool.items) == ["E1", "E2"]
    assert pool.get("E2").kind == "metric"


# ---- 出典の有無・存在 ----


def test_出典のある主張は問題なし() -> None:
    claims = [Claim(text="原材料価格の高騰は売上原価を押し上げうる。", evidence_ids=["E1"])]
    assert check_claims(claims, _pool()) == []


def test_出典が無い主張は_エラー() -> None:
    issues = check_claims([Claim(text="事業は堅調である。", evidence_ids=[])], _pool())
    assert kinds(issues) == ["no_evidence"]
    assert issues[0].severity == "error"
    assert issues[0].claim_index == 0


def test_存在しない出典IDは_エラー() -> None:
    issues = check_claims([Claim(text="売上は増えた。", evidence_ids=["E1", "E99"])], _pool())
    assert kinds(issues) == ["unknown_evidence"]
    assert "E99" in issues[0].message


# ---- 結論を示す語 ----


def test_結論を示す語を含む主張は_エラー() -> None:
    for phrase in ("融資可能", "懸念なし", "推奨"):
        issues = check_claims(
            [Claim(text=f"財務は良好で{phrase}である。", evidence_ids=["E2"])], _pool()
        )
        assert "forbidden_phrase" in kinds(issues), phrase


def test_禁止語の一覧には_テンプレートの記載ルールの語が入っている() -> None:
    for phrase in ("融資可能", "懸念なし", "推奨"):
        assert phrase in FORBIDDEN_PHRASES


def test_禁止語の検出は_全角半角や空白に左右されない() -> None:
    assert find_forbidden("融資 可能と考える")
    assert find_forbidden("ｓ" + "推奨") == ["推奨"]
    assert find_forbidden("自己資本比率は標準水準である") == []


# ---- 数値 ----


def test_主張の数値が引用した証拠にあれば_問題なし() -> None:
    claims = [Claim(text="自己資本比率は76.4%である。", evidence_ids=["E2"])]
    assert check_claims(claims, _pool()) == []


def test_カンマ区切りの数値も_同じ値として照合する() -> None:
    claims = [Claim(text="売上原価は1250百万円増える可能性がある。", evidence_ids=["E1"])]
    assert check_claims(claims, _pool()) == []


def test_証拠に無い数値は_警告() -> None:
    issues = check_claims([Claim(text="自己資本比率は80.1%である。", evidence_ids=["E2"])], _pool())
    assert kinds(issues) == ["number_not_in_evidence"]
    assert issues[0].severity == "warning"
    assert "80.1" in issues[0].message


def test_別の証拠にだけある数値は_引用していなければ警告() -> None:
    issues = check_claims([Claim(text="自己資本比率は76.4%である。", evidence_ids=["E1"])], _pool())
    assert kinds(issues) == ["number_not_in_evidence"]


def test_債務償還年数の年は_照合する_暦の年だけ除く() -> None:
    pool = EvidencePool()
    pool.add(
        MetricEvidence(
            id="",
            sec_code="9999",
            company="x",
            doc_id="D1",
            label="債務償還年数",
            period="current",
            display="5.2年",
            value=Decimal("5.2"),
            basis="要償還債務 ÷ 償還原資 = 5.2年",
        )
    )
    wrong = check_claims([Claim(text="債務償還年数は8.9年である。", evidence_ids=["E1"])], pool)
    assert kinds(wrong) == ["number_not_in_evidence"]
    assert (
        check_claims([Claim(text="債務償還年数は5.2年である。", evidence_ids=["E1"])], pool) == []
    )


def test_出典のない主張の数値も_警告する_確認事項は出典なしでも書ける() -> None:
    issues = check_claims([Claim(text="売上高は1,234百万円である。", evidence_ids=[])], _pool())
    assert sorted(kinds(issues)) == ["no_evidence", "number_without_evidence"]


def test_小さい整数や年は_数値の照合の対象にしない() -> None:
    claims = [Claim(text="2期連続で、2026年3月期に3つの事業がある。", evidence_ids=["E2"])]
    assert check_claims(claims, _pool()) == []


def test_問題が複数あれば_すべて返す() -> None:
    issues = check_claims(
        [
            Claim(text="融資可能である。", evidence_ids=[]),
            Claim(text="比率は99.9%である。", evidence_ids=["E2"]),
        ],
        _pool(),
    )
    assert sorted(kinds(issues)) == ["forbidden_phrase", "no_evidence", "number_not_in_evidence"]
    assert {i.claim_index for i in issues} == {0, 1}

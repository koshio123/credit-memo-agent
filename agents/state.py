"""メモの材料の型: 証拠（本文・数値）と、証拠に結びつく主張。

証拠の ID（E1, E2, ...）はコードが振る。LLM は ID を選ぶだけで、引用文も数値も書かない。
"""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from retrieval.citations import SourceSpan


class PassageEvidence(BaseModel):
    """有価証券報告書の本文。検索で見つかったチャンクから、コードが作る。"""

    model_config = ConfigDict(frozen=True)

    kind: Literal["passage"] = "passage"
    id: str
    sec_code: str
    company: str
    heading_path: list[str]
    spans: list[SourceSpan] = Field(min_length=1)  # ページごと（複数ページにまたがる場合は複数）

    @property
    def text(self) -> str:
        """本文の証拠の全文。

        Returns:
            各スパンの引用文を改行でつないだ文字列。
        """
        return "\n".join(span.quote for span in self.spans)


class MetricEvidence(BaseModel):
    """コードが算定した数値。算式と入力値、XBRL の項目名、PDF のページを持つ。"""

    model_config = ConfigDict(frozen=True)

    kind: Literal["metric"] = "metric"
    id: str
    sec_code: str
    company: str
    doc_id: str
    label: str  # 例: 自己資本比率
    period: Literal["current", "previous"]
    display: str  # 文章に書く表記（例: 76.4%）。LLM はこの表記をそのまま使う
    value: Decimal | None  # None は算定不能
    level: str | None = None  # 規程の水準（標準・留意・要精査）。区分のない指標は None
    basis: str  # 算式と入力値（規程 第7条）
    xbrl_items: dict[str, str] = Field(default_factory=dict[str, str])  # 入力項目 -> XBRL の項目名
    pdf_pages: list[int] = Field(default_factory=list[int])  # 入力値が載っている PDF のページ
    # xbrl: XBRL・コードの算定結果 / search_record: コードが本文を語句で検索した記録（第13条）
    origin: Literal["xbrl", "search_record"] = "xbrl"

    @property
    def text(self) -> str:
        """数値の証拠を、検索や照合に使う1つの文字列にしたもの。

        Returns:
            指標名・表記・水準・計算根拠をつないだ文字列。
        """
        return f"{self.label} {self.display} {self.level or ''} {self.basis}"


Evidence = Annotated[PassageEvidence | MetricEvidence, Field(discriminator="kind")]


class EvidencePool:
    """証拠の集まり。追加すると、E1 から順に ID が振られる。"""

    def __init__(self) -> None:
        self.items: dict[str, PassageEvidence | MetricEvidence] = {}

    def add[T: (PassageEvidence, MetricEvidence)](self, evidence: T) -> T:
        """証拠を追加し、E1 から順の ID を振る。

        Args:
            evidence: 追加する証拠。ID はここで上書きされる。

        Returns:
            ID を振った、保存された証拠。
        """
        stored = evidence.model_copy(update={"id": f"E{len(self.items) + 1}"})
        self.items[stored.id] = stored
        return stored

    def find_passage(self, spans: list[SourceSpan]) -> PassageEvidence | None:
        """同じ出典スパンの本文の証拠があれば返す（別の問いで同じ箇所が見つかったときに使い回す）。

        Args:
            spans: 探す出典スパン。

        Returns:
            同じスパンの本文の証拠。無ければ None。
        """
        for item in self.items.values():
            if isinstance(item, PassageEvidence) and item.spans == spans:
                return item
        return None

    def get(self, evidence_id: str) -> PassageEvidence | MetricEvidence:
        """ID で証拠を取る。

        Args:
            evidence_id: 証拠の ID（例: E1）。

        Returns:
            証拠。

        Raises:
            KeyError: その ID の証拠が無いとき。
        """
        return self.items[evidence_id]

    def __contains__(self, evidence_id: object) -> bool:
        """その ID の証拠があるか。

        Args:
            evidence_id: 証拠の ID。

        Returns:
            あれば True。
        """
        return evidence_id in self.items


class Claim(BaseModel):
    """メモの 1 つの主張。出典（証拠の ID）を 1 つ以上持たなければならない。"""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list[str])


# メモの、LLM が書く主張の節（キーは MemoDraft の項目名）。出力の節の並びとラベルの基準
CLAIM_SECTIONS: tuple[tuple[str, str], ...] = (
    ("overview", "企業概要"),
    ("financial_findings", "財務の所見"),
    ("business_risks", "事業リスク"),
    ("positives", "肯定的な要素"),
    ("negatives", "否定的な要素"),
    ("open_items", "確認が必要な事項"),
)

"""サービスを MCP のツールとして公開する。ツールの説明は、エージェントが読んで使い方を判断する。"""

from collections.abc import Callable
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from edinet_mcp.models import (
    CompanyInfo,
    FinancialsResult,
    PageResult,
    Passage,
    Period,
    RatiosResult,
)
from edinet_mcp.service import EdinetMcpError, EdinetService

INSTRUCTIONS = (
    "EDINET の有価証券報告書（当期）を調べるツール。与信メモの下書きの材料集めに使う。"
    "数値は get_financials / get_ratios（XBRL 由来）から取り、"
    "本文の根拠は search_filings で探して、書類 ID とページを出典として添える。"
    "融資の可否や条件は判断しない。"
)

SecCode = Annotated[str, Field(description="証券コード（4桁。例: 6744）")]


def build_server(service: EdinetService) -> MCPServer:
    # ツールは async def にして、イベントループの上で1つずつ実行する（同期の関数はスレッドで
    # 並行に走る）。サービスは1つの DB 接続・埋め込みモデル・BM25 の索引を共有しているため。
    # 1回の処理は短い（数十ミリ秒〜1秒）ので、直列でよい
    server = MCPServer("edinet-mcp", instructions=INSTRUCTIONS)

    def guarded[T](call: Callable[[], T]) -> T:
        """サービスの失敗を、エージェントに理由が見えるツールのエラーにする。"""
        try:
            return call()
        except EdinetMcpError as e:
            raise ToolError(str(e)) from e

    @server.tool()
    async def list_companies() -> list[CompanyInfo]:
        """調べられる会社の一覧（証券コード・会社名・業種・当期の書類ID）。最初に呼んで対象を確かめる。"""
        return service.list_companies()

    @server.tool()
    async def search_filings(
        query: Annotated[
            str, Field(description="知りたい内容を文章で（例: 原材料価格の高騰の影響）")
        ],
        sec_code: SecCode,
        k: Annotated[int, Field(description="返す件数（1〜20）")] = 5,
    ) -> list[Passage]:
        """会社の有価証券報告書（当期）の本文から、質問に近い箇所を探す。

        語句の一致（BM25）と意味の近さ（埋め込み）を融合した順位で返す。各結果に書類IDとページ、
        見出しの階層がつくので、メモの根拠として引用できる。内容は get_page で前後を確かめられる。
        """
        return guarded(lambda: service.search_filings(query, sec_code, k))

    @server.tool()
    async def get_financials(
        sec_code: SecCode,
        period: Annotated[
            Period, Field(description="current（当期）か previous（前期）")
        ] = "current",
    ) -> FinancialsResult:
        """連結の財務数値（円）を XBRL から取る。貸借対照表・損益計算書の主要項目と借入の内訳。

        項目が null なのは XBRL に項目が無かったことで、0 とは違う。provenance に元の項目名がある。
        """
        return guarded(lambda: service.get_financials(sec_code, period))

    @server.tool()
    async def get_ratios(sec_code: SecCode) -> RatiosResult:
        """架空の融資内規に基づく財務比率と水準（標準・留意・要精査）を計算して返す。

        自己資本比率・流動比率・営業利益率・インタレスト・カバレッジ・債務償還年数・売上高成長率と、
        営業損失の連続・売上高の減少の該当。算定できない指標は理由つきで value が null になる。
        融資の可否や条件は判断しない。value は丸めていない値。文章には basis の数字を使う。
        """
        return guarded(lambda: service.get_ratios(sec_code))

    @server.tool()
    async def get_page(
        sec_code: SecCode,
        page: Annotated[int, Field(description="ページ番号（1始まり）")],
    ) -> PageResult:
        """有価証券報告書（当期）の1ページ分の本文。検索結果の前後や、出典のページを確かめるのに使う。"""
        return guarded(lambda: service.get_page(sec_code, page))

    return server

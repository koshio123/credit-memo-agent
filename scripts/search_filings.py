"""有報の本文検索を手で試す（LLM は呼ばない。手動テスト用）。

  uv run python -m scripts.search_filings 6744 "原材料価格の高騰の影響" "継続企業の前提" -k 3

先に PostgreSQL を起動し、scripts.ingest_index で取り込んでおく。結果は、書類ID・ページ・見出し・
引用文。PDF の該当ページを開いて、引用がそこにあることを確かめる。
"""

import argparse
import logging

from edinet_mcp.service import EdinetMcpError
from edinet_mcp.wiring import build_service

logging.basicConfig(level=logging.INFO, format="%(message)s")
for _name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("search_filings")


def main() -> int:
    """有価証券報告書を検索して結果を表示する。失敗は理由だけを出して終わる。

    Returns:
        終了コード。成功は 0、失敗は 1。
    """
    try:
        return _main()
    except EdinetMcpError as e:
        log.error("失敗しました: %s", e)
        return 1


def _main() -> int:
    """引数を読み、問いごとに検索して上位の結果を表示する。

    Returns:
        終了コード。正常終了は 0。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sec_code")
    parser.add_argument("queries", nargs="+")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    service = build_service()
    for query in args.queries:
        log.info("== %s", query)
        for passage in service.search_filings(query, args.sec_code, k=args.k):
            pages = sorted({s.page for s in passage.spans})
            log.info(
                "[%d] %s p.%s | %s",
                passage.rank,
                passage.doc_id,
                "-".join(str(p) for p in (pages[0], pages[-1])) if len(pages) > 1 else pages[0],
                " > ".join(passage.heading_path[-2:]),
            )
            log.info("    %s", passage.text.replace("\n", " ")[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

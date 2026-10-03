"""edinet-mcp を stdio で起動し、実データでツールを呼んで動作を確かめる。CI では実行しない。

  uv run python -m scripts.check_edinet_mcp

先に scripts.ingest_index で取り込み、PostgreSQL を起動しておく。
"""

import asyncio
import logging
import sys

from mcp import Client, StdioServerParameters

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("check_edinet_mcp")


async def main() -> int:
    """MCP サーバーを子プロセスで起動し、各ツールを一通り呼んで確かめる。

    Returns:
        終了コード。正常終了は 0。
    """
    params = StdioServerParameters(command=sys.executable, args=["-m", "edinet_mcp"])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        log.info("ツール: %s", ", ".join(t.name for t in tools))

        companies = (await client.call_tool("list_companies", {})).structured_content
        first = companies["result"][0]  # type: ignore[index]
        code = first["sec_code"]
        log.info("対象: %s %s（%d社）", code, first["name"], len(companies["result"]))  # type: ignore[index]

        found = await client.call_tool(
            "search_filings",
            {"query": "原材料価格の高騰が収益に与える影響", "sec_code": code, "k": 3},
        )
        for p in found.structured_content["result"]:  # type: ignore[index]
            log.info(
                "  p.%s-%s %s | %s",
                p["page_start"],
                p["page_end"],
                p["heading_path"],
                p["text"][:40],
            )

        ratios = await client.call_tool("get_ratios", {"sec_code": code})
        r = ratios.structured_content["ratios"]  # type: ignore[index]
        log.info("自己資本比率: %s / %s", r["equity_ratio"]["value"], r["equity_ratio"]["level"])

        page = await client.call_tool("get_page", {"sec_code": code, "page": 1})
        log.info("1ページ目: %s…", page.structured_content["text"][:30])  # type: ignore[index]

        bad = await client.call_tool("get_page", {"sec_code": code, "page": 9999})
        log.info("範囲外のページは is_error=%s", bad.is_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

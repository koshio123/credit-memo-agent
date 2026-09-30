"""MCP サーバーを stdio で起動する:  uv run python -m edinet_mcp

標準出力は MCP のプロトコルが使うので、ログは標準エラーに出す。
"""

import logging
import sys

from edinet_mcp.server import build_server
from edinet_mcp.wiring import build_service


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    build_server(build_service()).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

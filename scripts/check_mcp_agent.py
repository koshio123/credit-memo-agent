"""Claude Agent SDK から edinet-mcp のツールを呼べることを、実機で確認する。

  uv run python -m scripts.check_mcp_agent

Claude の利用枠を少し使う（1回の問い合わせ）。CI では実行しない。
API キーでの課金を避けるため、サブスクリプションのログインで動いていなければ中止する。
先に PostgreSQL の起動と scripts.ingest_index の実行が必要。
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    ToolUseBlock,
    query,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("check_mcp_agent")

PROMPT = (
    "get_ratios ツールで証券コード 6744 の財務比率を取り、"
    "自己資本比率の値と水準を1文で答えてください。ツールの結果に無いことは書かないでください。"
)


async def main() -> int:
    options = ClaudeAgentOptions(
        system_prompt="あなたは与信メモの下書きを手伝うアシスタントです。融資の可否は判断しません。",
        model="haiku",
        tools=[],  # 組み込みツールは使わない
        mcp_servers={
            "edinet": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "edinet_mcp"],
            }
        },
        allowed_tools=["mcp__edinet__get_ratios"],
        setting_sources=[],
        max_turns=4,
        cwd=Path.cwd(),
        settings=json.dumps({"disableClaudeAiConnectors": True}),
    )
    called: list[str] = []
    answer = ""
    async for message in query(prompt=PROMPT, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            source = message.data.get("apiKeySource")
            if source != "none":
                log.error("サブスクリプション以外の認証（%r）です。中止します", source)
                return 1
            servers = {s["name"]: s["status"] for s in message.data.get("mcp_servers", [])}
            log.info("MCP サーバーの状態: %s", servers)
        elif isinstance(message, AssistantMessage):
            called += [b.name for b in message.content if isinstance(b, ToolUseBlock)]
        elif isinstance(message, ResultMessage):
            answer = message.result or ""
            log.info("費用の目安: %s", message.total_cost_usd)
    log.info("呼ばれたツール: %s", called)
    log.info("回答: %s", answer)
    return 0 if any(name.endswith("get_ratios") for name in called) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

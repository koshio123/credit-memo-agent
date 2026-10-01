"""与信メモの下書きを生成する（実際に LLM を呼ぶ。CI では実行しない）。

  uv run python -m scripts.generate_memo --sec-code 6744 --mode multi_agent
  LLM_BACKEND=claude_code uv run python -m scripts.generate_memo --sec-code 6744 --mode baseline

先に scripts.ingest_index で取り込み、PostgreSQL を起動しておく。結果は data/memos/（gitignore）に
Markdown と JSON で保存する。LLM_BACKEND=claude_code は自分の Claude のログインで動き、利用枠を使う
ので、少量・手動の実行だけにする（一度に 3 社まで）。ローカルは LOCAL_MODEL_* と LOCAL_NUM_CTX を
環境変数で指定する（既定のコンテキスト長 16384 では、証拠の一覧が収まらないことがある）。
"""

import argparse
import asyncio
import logging
from pathlib import Path

from agents.export import save_result
from agents.pipeline import MemoResult, run_baseline, run_multi_agent
from edinet_mcp.service import EdinetMcpError
from edinet_mcp.wiring import build_service
from llm.factory import create_backend
from llm.settings import LLMSettings
from llm.structured import StructuredOutputError
from llm.types import LLMBackendError

logging.basicConfig(level=logging.INFO, format="%(message)s")
for _name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("generate_memo")

MAX_CLAUDE_COMPANIES = 3


async def main() -> int:
    try:
        return await _main()
    except (EdinetMcpError, LLMBackendError, StructuredOutputError, ValueError) as e:
        # 設定・入力の誤りやバックエンドの失敗は、理由だけを出して終わる
        log.error("失敗しました: %s", e)
        return 1


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec-code", nargs="+", required=True)
    parser.add_argument("--mode", choices=["multi_agent", "baseline"], default="multi_agent")
    parser.add_argument("--out", type=Path, default=Path("data/memos"))
    args = parser.parse_args()

    settings = LLMSettings()
    if settings.llm_backend == "claude_code" and len(args.sec_code) > MAX_CLAUDE_COMPANIES:
        log.error(
            "claude_code は自分の利用枠を使うため、一度に %d 社までです", MAX_CLAUDE_COMPANIES
        )
        return 2
    backend = create_backend(settings)
    service = build_service()
    run = run_multi_agent if args.mode == "multi_agent" else run_baseline

    for sec_code in args.sec_code:
        result: MemoResult = await run(service, backend, sec_code)
        model = backend.model_for("standard")
        md_path, _ = save_result(result, args.out, settings.llm_backend, model)
        log.info(
            "%s %s: LLM %d 回（キャッシュ %d）入力 %d・出力 %d トークン / 除外 %d・警告 %d → %s",
            sec_code,
            args.mode,
            result.calls,
            result.cached_calls,
            result.input_tokens,
            result.output_tokens,
            len(result.memo.rejected),
            len(result.memo.warnings),
            md_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

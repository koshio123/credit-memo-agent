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
    """メモを生成する。LLM バックエンドの失敗は、理由だけを出して終わる。

    Returns:
        終了コード。成功は 0、失敗は 1、claude_code で会社が多すぎるときは 2。
    """
    try:
        return await _main()
    except LLMBackendError as e:
        # バックエンドの作成の失敗（API キーが環境にある等）や、利用枠・接続の失敗。
        # 会社ごとの問題ではないので、残りの会社も続けずに、理由だけを出して終わる
        log.error("失敗しました: %s", e)
        return 1


async def _main() -> int:
    """引数を読み、会社ごとにメモを生成して保存する。会社ごとの失敗は他の会社に影響させない。

    Returns:
        終了コード。成功は 0、失敗した会社があれば 1、claude_code で会社が多すぎるときは 2。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec-code", nargs="+", required=True)
    parser.add_argument("--mode", choices=["multi_agent", "baseline"], default="multi_agent")
    parser.add_argument(
        "--verify-rounds",
        type=int,
        default=None,
        help="指定すると Verifier で検証し、失敗した主張を最大この回数まで書き直す（0 は検証だけ）",
    )
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

    failed = 0
    for sec_code in args.sec_code:
        try:
            result: MemoResult = await run(service, backend, sec_code, args.verify_rounds)
        except (EdinetMcpError, StructuredOutputError) as e:
            # 会社ごとの失敗（対象外の証券コード、データ未取得など）は、他の会社に影響させない
            failed += 1
            log.error("%s: 失敗しました: %s", sec_code, e)
            continue
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

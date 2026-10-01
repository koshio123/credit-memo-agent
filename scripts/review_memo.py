"""生成したメモの JSON から、主張と出典を突き合わせるための確認用の Markdown を作る。

  uv run python -m scripts.review_memo data/memos/6744_multi_agent_claude_code.json

出力は同じ場所の <名前>.review.md。LLM は呼ばない。判定の欄に、人が印を付ける。
"""

import argparse
import json
import logging
from pathlib import Path

from agents.review import render_review

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("review_memo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.json_path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out = path.with_suffix(".review.md")
        out.write_text(render_review(payload), encoding="utf-8")
        log.info("%s → %s", path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

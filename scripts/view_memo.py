"""生成したメモの JSON から、引用ビューア（単体の HTML）を作る。LLM は呼ばない。

  uv run python -m scripts.view_memo data/memos/6744_multi_agent_claude_code.json \
      --pdf-base "file://$PWD/data/edinet/{doc_id}/{doc_id}.pdf"

出力は同じ場所の <名前>.html。書類の本体は埋め込まない。--pdf-base を渡すと、出典の PDF
のページへのリンクが付く（手元に取得した PDF を指す。ブラウザによっては #page= を使えない）。
"""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from agents.export import load_result
from agents.viewer import render_html

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("view_memo")


def main() -> int:
    """引数のJSONごとにHTMLを書く。読めないファイルがあっても残りは続ける。

    Returns:
        すべて成功なら 0、読めないファイルがあれば 1。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="+", type=Path)
    parser.add_argument("--pdf-base", default=None)
    args = parser.parse_args()
    failed = 0
    for path in args.json_path:
        try:
            out = path.with_suffix(".html")
            out.write_text(render_html(load_result(path), args.pdf_base), encoding="utf-8")
            log.info("%s → %s", path, out)
        except (OSError, ValidationError, ValueError) as e:
            failed += 1
            log.error("%s: 読めませんでした: %s", path, str(e).splitlines()[0][:200])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

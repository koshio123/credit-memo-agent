"""生成したメモの JSON から、主張と出典を突き合わせるための確認用の Markdown を作る。

  uv run python -m scripts.review_memo data/memos/6744_multi_agent_claude_code.json

出力は同じ場所の <名前>.review.md。LLM は呼ばない。判定の欄に、人が印を付ける。
読めないファイルがあっても、残りの処理は続ける（終了コードは 1）。
"""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from agents.export import load_result
from agents.review import render_review

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("review_memo")


def review(path: Path, blind: bool = False) -> Path:
    """保存した結果から、主張と出典の突き合わせの Markdown を作って保存する。

    Args:
        path: 生成結果の JSON のパス。
        blind: True なら Verifier の判定を出さない（<名前>.blind.review.md に書く）。

    Returns:
        書き出した Markdown のパス。
    """
    out = path.with_suffix(".blind.review.md" if blind else ".review.md")
    out.write_text(render_review(load_result(path), blind=blind), encoding="utf-8")
    return out


def main() -> int:
    """生成結果の JSON ごとに、確認用の Markdown を作る。

    Returns:
        終了コード。成功は 0、読めない JSON があれば 1。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="+", type=Path)
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Verifier の判定を出さない。先に人が判定してから、通常の出力と比べる",
    )
    args = parser.parse_args()
    failed = 0
    for path in args.json_path:
        try:
            log.info("%s → %s", path, review(path, args.blind))
        except ValidationError as e:
            failed += 1
            where = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:6]
            )
            log.error("%s: 読めませんでした: %s", path, where[:400])
        except (OSError, ValueError) as e:
            failed += 1
            log.error("%s: 読めませんでした: %s", path, str(e)[:200])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

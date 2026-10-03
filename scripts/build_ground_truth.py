"""XBRL の財務データから、対象企業の正解データを作る（data/ground_truth/ground_truth.json）。

  uv run python -m scripts.build_ground_truth

先に scripts.fetch_filings で書類を取得しておく。出力は data/（gitignore）に置く。
"""

import json
import logging
from pathlib import Path

from evals.companies import load_companies
from evals.ground_truth import build_company_ground_truth

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_ground_truth")

DATA_DIR = Path("data/edinet")
OUT = Path("data/ground_truth/ground_truth.json")


def main() -> int:
    """全社の正解データを作って JSON に書き出す。

    Returns:
        終了コード。正常終了は 0。
    """
    truths = [build_company_ground_truth(c, DATA_DIR) for c in load_companies()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"companies": [t.model_dump(mode="json") for t in truths]}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for t in truths:
        log.info("%s %s: 訂正再表示の疑い %d 件", t.sec_code, t.name, len(t.mismatches))
    log.info("== %s に %d 社分を書き出しました", OUT, len(truths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

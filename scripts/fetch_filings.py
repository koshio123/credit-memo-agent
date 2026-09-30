"""対象企業の有価証券報告書（直近期・前期）の PDF・XBRL・CSV を data/edinet/ に取得する。

  uv run python -m scripts.fetch_filings

取得済みのファイルは再取得しない（途中で止まっても、もう一度実行すれば続きから進む）。
書類の本体は data/（gitignore）に置き、リポジトリには含めない。
"""

import argparse
import logging
from pathlib import Path

from evals.companies import DATASETS, load_dataset
from ingest.edinet import EdinetClient, Kind, fetch_all
from ingest.settings import EdinetSettings

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fetch_filings")

DATA_DIR = Path("data/edinet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dev")
    parser.add_argument(
        "--kinds", nargs="+", choices=[k.value for k in Kind], default=[k.value for k in Kind]
    )
    args = parser.parse_args()

    key = EdinetSettings().edinet_api_key.get_secret_value()
    if not key:
        log.error("EDINET_API_KEY が .env に設定されていません")
        return 1

    results = fetch_all(
        EdinetClient(api_key=key, data_dir=DATA_DIR),
        load_dataset(args.dataset),
        kinds=[Kind(k) for k in args.kinds],
    )

    fetched = [r for r in results if r.downloaded]
    skipped = [r for r in results if not r.downloaded and not r.error]
    failed = [r for r in results if r.error]
    total_mb = sum(r.path.stat().st_size for r in results if r.path.exists()) / 1e6
    log.info(
        "== 結果: 今回取得 %d / 取得済み %d / 失敗 %d（保存先の合計 %.0f MB）",
        len(fetched),
        len(skipped),
        len(failed),
        total_mb,
    )
    for r in failed:
        log.error("失敗: %s", r.error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

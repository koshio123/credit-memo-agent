"""EDINET API キーが使えるかを確認する（CIでは実行しない）。

  uv run python -m scripts.check_edinet [YYYY-MM-DD]

.env の EDINET_API_KEY を読み、指定日（省略時は 2026-09-29）に提出された書類の一覧を取る。
キー自体と、キーを含むURLは、出力にもエラーにも出さない。
"""

import logging
import sys

import httpx

from ingest.settings import EdinetSettings

logging.basicConfig(level=logging.INFO, format="%(message)s")
# httpx / httpcore は INFO で「リクエストのURL」を出す。EDINET はキーをURLのクエリで渡すので、
# 黙らせないとキーがログに出てしまう
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("check_edinet")

DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DEFAULT_DATE = "2026-09-29"


def run(key: str, date: str, client: httpx.Client) -> int:
    try:
        # type=1 は書類の一覧（メタデータ）のみ。書類の本体は取得しない
        res = client.get(
            DOCUMENTS_URL,
            params={"date": date, "type": 1, "Subscription-Key": key},
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        # 例外のメッセージにURL（=キー）が入りうるので、型名だけを出す
        log.error("接続に失敗しました: %s", type(e).__name__)
        return 1

    log.info("HTTP %s", res.status_code)
    if res.status_code != 200:
        log.error(
            "キーが無効か、リクエストが不正です。本文: %s", res.text[:200].replace(key, "***")
        )
        return 1

    try:
        body = res.json()
    except ValueError:
        log.error("応答がJSONではありません（メンテナンス中の可能性）")
        return 1
    meta = body.get("metadata", {})
    log.info("status=%s message=%s", meta.get("status"), meta.get("message"))
    if str(meta.get("status")) != "200":
        return 1
    log.info("%s の提出書類数: %s", date, meta.get("resultset", {}).get("count"))
    return 0


def main() -> int:
    key = EdinetSettings().edinet_api_key.get_secret_value()
    if not key:
        log.error("EDINET_API_KEY が .env に設定されていません")
        return 1
    date = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATE
    with httpx.Client() as client:
        return run(key, date, client)


if __name__ == "__main__":
    raise SystemExit(main())

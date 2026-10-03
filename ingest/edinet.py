"""EDINET API v2 から、有価証券報告書の本体（PDF・XBRL・財務データCSV）を取得して保存する。

守ること:
- **キーをログにもエラーにも出さない。** EDINET はキーを URL のクエリで渡すため、httpx が INFO で
  出すリクエスト URL にキーが含まれる。httpx / httpcore のログを WARNING 以上にし、例外の
  メッセージにも URL を入れない。
- **壊れたファイルを残さない。** EDINET は失敗時にも HTTP 200 でエラーの JSON を返すことがある。
  先頭のバイト列（PDF は %PDF、zip は PK）を確認してから、一時ファイル経由で保存する。
- **取得済みは再取得しない。** 回線が遅いので、やり直しの無駄を避ける。
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import SecretStr

from evals.companies import Company

# EDINET のキーは URL のクエリに載る。httpx はリクエストの URL を INFO で出すので、黙らせる
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents"
_ATTEMPTS = 3  # 初回 + リトライ2回
_INTERVAL_SECONDS = 0.5  # リクエストの間隔。EDINET に負荷をかけない


class Kind(StrEnum):
    XBRL = "xbrl"  # 提出本文書及び監査報告書（XBRL の zip）
    PDF = "pdf"
    CSV = "csv"  # 財務データ（XBRL を CSV にしたもの、zip）


# 種別 -> (API の type パラメータ, 保存時の拡張子, 先頭のバイト列)
_SPEC: dict[Kind, tuple[int, str, bytes]] = {
    Kind.XBRL: (1, ".xbrl.zip", b"PK"),
    Kind.PDF: (2, ".pdf", b"%PDF"),
    Kind.CSV: (5, ".csv.zip", b"PK"),
}


class EdinetError(Exception):
    """取得の失敗。メッセージに URL（=キー）は含めない。"""


@dataclass(frozen=True)
class FetchResult:
    doc_id: str
    kind: Kind
    path: Path
    downloaded: bool  # True: 今回取得した / False: 取得済みだった、または失敗
    error: str | None = None


class EdinetClient:
    def __init__(
        self,
        api_key: str,
        data_dir: Path,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """EDINET のクライアントを作る。

        Args:
            api_key: EDINET の API キー。ログやエラーには出さない。
            data_dir: 取得した書類の保存先。
            http: HTTP クライアント。None なら新しく作る。
            sleep: リクエストの間隔をあける関数。テストで差し替える。
        """
        self._key = SecretStr(api_key)
        self._dir = data_dir
        self._http = http or httpx.Client(timeout=120.0)
        self._sleep = sleep

    def path_for(self, doc_id: str, kind: Kind) -> Path:
        """書類の保存先のパスを返す。

        Args:
            doc_id: 書類ID。
            kind: 書類の種別。

        Returns:
            保存先のパス（存在するとは限らない）。
        """
        return self._dir / doc_id / f"{doc_id}{_SPEC[kind][1]}"

    def download(self, doc_id: str, kind: Kind) -> FetchResult:
        """書類を取得して保存する。取得済みなら再取得しない。

        Args:
            doc_id: 書類ID。
            kind: 書類の種別。

        Returns:
            保存先と、今回取得したかどうか。

        Raises:
            EdinetError: 取得できない、または想定した形式でないとき。
        """
        path = self.path_for(doc_id, kind)
        if path.exists() and path.stat().st_size > 0:
            return FetchResult(doc_id, kind, path, downloaded=False)
        content = self._get(doc_id, kind)
        self._write_atomic(path, content)
        logger.info("取得しました: %s %s (%s bytes)", doc_id, kind.value, f"{len(content):,}")
        return FetchResult(doc_id, kind, path, downloaded=True)

    def _get(self, doc_id: str, kind: Kind) -> bytes:
        """書類のバイト列を取得する。5xx と 429 はリトライする。

        Args:
            doc_id: 書類ID。
            kind: 書類の種別。

        Returns:
            書類の中身。先頭のバイト列は種別に合うことを確認済み。

        Raises:
            EdinetError: 取得できない、または想定した形式でないとき。
        """
        type_param, _, magic = _SPEC[kind]
        label = kind.value.upper()
        last = "不明"
        for attempt in range(_ATTEMPTS):
            self._sleep(_INTERVAL_SECONDS if attempt == 0 else 2.0**attempt)
            try:
                res = self._http.get(
                    f"{BASE_URL}/{doc_id}",
                    params={"type": type_param, "Subscription-Key": self._key.get_secret_value()},
                )
            except httpx.HTTPError as e:
                # 例外のメッセージに URL（=キー）が入りうるので、型名だけを残す
                last = type(e).__name__
                continue
            if res.status_code == 200:
                if not res.content.startswith(magic):
                    raise EdinetError(
                        f"{doc_id} の{label}が想定した形式ではありません（エラー応答の可能性）"
                    )
                return res.content
            last = f"HTTP {res.status_code}"
            if res.status_code < 500 and res.status_code != 429:
                break  # 4xx（429を除く）はリトライしても直らない
        raise EdinetError(f"{doc_id} の{label}を取得できませんでした: {last}")

    @staticmethod
    def _write_atomic(path: Path, content: bytes) -> None:
        """一時ファイル経由で保存し、壊れたファイルを残さない。

        Args:
            path: 保存先。親のディレクトリが無ければ作る。
            content: 書き込む中身。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(content)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def fetch_all(
    client: EdinetClient,
    companies: Iterable[Company],
    kinds: Iterable[Kind] = (Kind.PDF, Kind.XBRL, Kind.CSV),
) -> list[FetchResult]:
    """各社の直近期・前期の書類を取得する。1件失敗しても続け、失敗は結果に含めて返す。

    Args:
        client: EDINET のクライアント。
        companies: 対象の会社。
        kinds: 取得する書類の種別。

    Returns:
        書類ごとの取得結果。失敗したものは error にメッセージが入る。
    """
    kinds = tuple(kinds)
    results: list[FetchResult] = []
    for company in companies:
        for filing in (company.filings.current, company.filings.previous):
            for kind in kinds:
                try:
                    results.append(client.download(filing.doc_id, kind))
                except EdinetError as e:
                    results.append(
                        FetchResult(
                            filing.doc_id,
                            kind,
                            client.path_for(filing.doc_id, kind),
                            downloaded=False,
                            error=str(e),
                        )
                    )
    return results

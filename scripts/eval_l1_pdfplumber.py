"""L1 評価: pdfplumber（基準線）で PDF から財務数値を読み、XBRL の正解データと突き合わせる。

  uv run python -m scripts.eval_l1_pdfplumber

PDF の本文は data/pdf_text/ にキャッシュする（2回目以降は速い）。レポートは数値を含まない形で
evals/reports/ に書き、値を含む詳細は data/l1/（gitignore）に書く。
"""

import argparse
import json
import logging
import time
from pathlib import Path

from evals.companies import DATASETS, load_dataset
from evals.ground_truth import build_company_ground_truth
from evals.l1 import CompanyScore, render_report, score_company, summarize
from ingest.pdf_baseline import ExtractedValue, cached_pages, extract_items

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval_l1")

EDINET_DIR = Path("data/edinet")
TEXT_CACHE = Path("data/pdf_text")
METHOD = "pdfplumber（基準線）"
LABELS = {
    "dev": "開発用（規則の作成に PDF を見た会社）",
    "heldout": "保留データ（PDF を見ていない会社）",
}


def pages_for(doc_id: str) -> list[str]:
    started = time.time()
    pages = cached_pages(EDINET_DIR / doc_id / f"{doc_id}.pdf", TEXT_CACHE)
    log.info("  %s: %d ページ（%.0f 秒）", doc_id, len(pages), time.time() - started)
    return pages


def _detail(items: dict[str, ExtractedValue]) -> dict[str, dict[str, object]]:
    return {
        k: {
            "value": None if v.value is None else str(v.value),
            "section": v.section,
            "page": v.page,
            "line": v.line,
            "reason": v.reason,
        }
        for k, v in items.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dev")
    args = parser.parse_args()
    detail_out = Path(f"data/l1/pdfplumber_{args.dataset}.json")
    report_out = Path(f"evals/reports/l1_pdfplumber_{args.dataset}.md")

    scores: list[CompanyScore] = []
    details: dict[str, object] = {}
    for company in load_dataset(args.dataset):
        log.info("%s %s", company.sec_code, company.name)
        items = extract_items(pages_for(company.filings.current.doc_id))
        truth = build_company_ground_truth(company, EDINET_DIR).periods[1].financials
        scores.append(score_company(company.sec_code, company.name, items, truth))
        details[company.sec_code] = _detail(items)

    detail_out.parent.mkdir(parents=True, exist_ok=True)
    detail_out.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        render_report(f"{METHOD} — {LABELS[args.dataset]}", scores), encoding="utf-8"
    )

    summary = summarize(scores)
    log.info("== 全体の正答: %d/%d（レポート: %s）", summary.correct, summary.total, report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

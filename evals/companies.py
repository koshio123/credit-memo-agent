"""検証に使う対象企業の一覧（evals/datasets/companies.json）を型つきで読む。"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DATASET_PATH = Path(__file__).parent / "datasets" / "companies.json"


class Filing(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    period_end: str
    submitted: str


class Filings(BaseModel):
    model_config = ConfigDict(frozen=True)

    current: Filing
    previous: Filing


class SelectionSnapshot(BaseModel):
    """選定時の概算（連結、直近期）。正解データは XBRL から算出し直す。"""

    model_config = ConfigDict(frozen=True)

    note: str
    equity_ratio_pct: float
    current_ratio_pct: float
    operating_margin_pct: float


class Company(BaseModel):
    model_config = ConfigDict(frozen=True)

    sec_code: str
    edinet_code: str
    name: str
    industry: str
    fiscal_year_end: str
    capital_million_yen: int
    filings: Filings
    why_selected: str
    selection_snapshot: SelectionSnapshot


class CompanyDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected_on: str
    companies: list[Company]


def load_companies(path: Path = DATASET_PATH) -> list[Company]:
    return CompanyDataset.model_validate(json.loads(path.read_text(encoding="utf-8"))).companies

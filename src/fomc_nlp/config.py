"""Project-wide constants and lightweight configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

FED_BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{FED_BASE_URL}/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_INDEX_URL = f"{FED_BASE_URL}/monetarypolicy/fomc_historical.htm"


@dataclass(frozen=True)
class PipelineConfig:
    """Default collection period."""

    start_year: int = 2000
    end_year: int = 2024

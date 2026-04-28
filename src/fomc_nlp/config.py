"""Project-wide constants and lightweight configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

FED_BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{FED_BASE_URL}/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_INDEX_URL = f"{FED_BASE_URL}/monetarypolicy/fomc_historical.htm"


@dataclass(frozen=True)
class PipelineConfig:
    """Default parameters for a reproducible M2-level analysis."""

    start_year: int = 2000
    end_year: int = 2024
    min_df: int = 1
    max_features: int = 5000


HAWKISH_TERMS = [
    "inflation",
    "price stability",
    "elevated",
    "tightening",
    "restrictive",
    "firming",
    "increase",
    "increased",
    "increases",
    "pressure",
    "pressures",
    "overheating",
    "strong labor market",
    "higher rates",
    "persistent inflation",
    "inflationary pressures",
]

DOVISH_TERMS = [
    "unemployment",
    "slowdown",
    "weakness",
    "accommodative",
    "support",
    "downside risks",
    "lower rates",
    "easing",
    "recession",
    "recovery",
    "labor market slack",
    "weak demand",
    "economic weakness",
]

UNCERTAINTY_TERMS = [
    "uncertainty",
    "uncertain",
    "risks",
    "risk",
    "downside",
    "upside",
    "volatility",
    "financial conditions",
    "stress",
    "disruptions",
    "concerns",
    "monitor",
    "likely",
    "may",
    "could",
]

KEYWORD_TERMS = [
    "inflation",
    "employment",
    "unemployment",
    "labor market",
    "risks",
    "uncertainty",
    "financial conditions",
    "growth",
    "recession",
]

DECISION_MASK_PATTERNS = [
    "decided to raise",
    "decided to lower",
    "decided to maintain",
    "decided to keep",
    "target range",
    "federal funds rate",
    "increase the target",
    "decrease the target",
    "raise the target",
    "lower the target",
    "keep the target",
    "left unchanged",
    "voted to",
]

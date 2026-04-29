"""Command-line pipeline orchestration."""

from __future__ import annotations

import argparse

import pandas as pd

from .config import (
    RAW_DATA_DIR,
    PipelineConfig,
)
from .data_collection import (
    build_raw_documents_dataset,
    collect_fomc_meeting_links,
)


def run_collection(config: PipelineConfig) -> pd.DataFrame:
    """Collect official Fed URLs and extract document text."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    links = collect_fomc_meeting_links(config.start_year, config.end_year)
    links.to_csv(RAW_DATA_DIR / "fomc_meeting_links.csv", index=False)
    raw = build_raw_documents_dataset(links)
    raw.to_csv(RAW_DATA_DIR / "fomc_documents_raw.csv", index=False)
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMC NLP pipeline")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "collect"],
        help="Pipeline steps to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
    )
    steps = set(args.steps)
    if "all" in steps:
        steps = {"collect"}

    if "collect" in steps:
        run_collection(config)


if __name__ == "__main__":
    main()

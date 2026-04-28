"""Command-line pipeline orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PipelineConfig,
)
from .data_collection import (
    build_document_dataset,
    collect_fomc_meeting_links,
    merge_statements_minutes,
)
from .features import (
    build_feature_dataset,
    build_top_semantic_shifts,
    export_keyword_frequencies,
)
from .visualization import build_all_figures


def export_raw_documents(statements: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Save extracted document text before cleaning and feature engineering."""

    statement_columns = [
        "date",
        "year",
        "meeting_id",
        "statement_url",
        "statement_text",
        "statement_download_error",
    ]
    minutes_columns = [
        "date",
        "year",
        "meeting_id",
        "minutes_url",
        "minutes_text",
        "minutes_download_error",
    ]
    raw = statements[statement_columns].merge(
        minutes[minutes_columns],
        on=["date", "year", "meeting_id"],
        how="outer",
    )
    raw = raw[
        [
            "date",
            "year",
            "meeting_id",
            "statement_url",
            "minutes_url",
            "statement_text",
            "minutes_text",
            "statement_download_error",
            "minutes_download_error",
        ]
    ].sort_values("date")
    raw.to_csv(RAW_DATA_DIR / "fomc_documents_raw.csv", index=False)
    return raw


def run_collection(config: PipelineConfig) -> pd.DataFrame:
    """Collect official Fed URLs, download text and merge corpora."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    links = collect_fomc_meeting_links(config.start_year, config.end_year)
    links.to_csv(RAW_DATA_DIR / "fomc_meeting_links.csv", index=False)

    statement_links = links.loc[links["statement_url"].notna(), ["date", "year", "meeting_id", "statement_url"]]
    minutes_links = links.loc[links["minutes_url"].notna(), ["date", "year", "meeting_id", "minutes_url"]]

    statements = build_document_dataset(statement_links, "statement")
    minutes = build_document_dataset(minutes_links, "minutes")
    export_raw_documents(statements, minutes)
    statements.to_csv(PROCESSED_DATA_DIR / "fomc_statements.csv", index=False)
    minutes.to_csv(PROCESSED_DATA_DIR / "fomc_minutes.csv", index=False)
    return merge_statements_minutes(statements, minutes)


def run_features(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build feature, keyword and top-shift tables."""

    merged_path = PROCESSED_DATA_DIR / "fomc_merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError("Missing data/processed/fomc_merged.csv. Run collection first.")
    merged = pd.read_csv(merged_path, dtype={"date": str})
    features = build_feature_dataset(
        merged,
        min_df=config.min_df,
        max_features=config.max_features,
    )
    keyword_df = export_keyword_frequencies(features)
    top_shifts = build_top_semantic_shifts(features)
    return features, keyword_df, top_shifts


def run_figures() -> None:
    """Generate all figures from processed feature tables."""

    features = pd.read_csv(PROCESSED_DATA_DIR / "fomc_features.csv", dtype={"date": str})
    keyword_path = PROCESSED_DATA_DIR / "keyword_frequencies.csv"
    top_shifts_path = PROCESSED_DATA_DIR / "top_semantic_shifts.csv"
    keyword_df = pd.read_csv(keyword_path, dtype={"date": str}) if keyword_path.exists() else pd.DataFrame()
    top_shifts = pd.read_csv(top_shifts_path, dtype={"date": str}) if top_shifts_path.exists() else pd.DataFrame()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    build_all_figures(features, keyword_df, top_shifts, FIGURES_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOMC NLP pipeline")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "collect", "features", "figures"],
        help="Pipeline steps to run.",
    )
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    steps = set(args.steps)
    if "all" in steps:
        steps = {"collect", "features", "figures"}

    if "collect" in steps:
        run_collection(config)
    if "features" in steps:
        run_features(config)
    if "figures" in steps:
        run_figures()


if __name__ == "__main__":
    main()

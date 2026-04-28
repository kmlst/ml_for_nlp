#!/usr/bin/env python3
"""Quick checks for processed FOMC datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


MERGED_COLUMNS = {
    "date",
    "year",
    "statement_url",
    "minutes_url",
    "statement_text",
    "minutes_text",
    "statement_clean_text",
    "minutes_clean_text",
    "statement_n_words",
    "minutes_n_words",
    "rate_decision",
}

FEATURE_COLUMNS = {
    "statement_tone_score",
    "minutes_tone_score",
    "statement_uncertainty_score",
    "minutes_uncertainty_score",
    "statement_semantic_shift_tfidf",
    "minutes_semantic_shift_tfidf",
    "same_meeting_distance_tfidf",
}

KNOWN_LABELS = {"hike", "cut", "hold", "unknown"}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.relative_to(ROOT)}")
    return pd.read_csv(path, dtype={"date": str})


def _check_columns(df: pd.DataFrame, expected: set[str], name: str) -> list[str]:
    missing = sorted(expected - set(df.columns))
    return [f"{name}: missing columns {missing}"] if missing else []


def _check_dates(df: pd.DataFrame, name: str) -> list[str]:
    issues = []
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().any():
        issues.append(f"{name}: invalid dates = {parsed.isna().sum()}")
    if df["date"].duplicated().any():
        issues.append(f"{name}: duplicated dates = {df['date'].duplicated().sum()}")
    if not parsed.dropna().is_monotonic_increasing:
        issues.append(f"{name}: dates are not sorted")
    return issues


def _check_merged(df: pd.DataFrame) -> list[str]:
    issues = []
    issues += _check_columns(df, MERGED_COLUMNS, "fomc_merged.csv")
    if issues:
        return issues

    issues += _check_dates(df, "fomc_merged.csv")
    labels = set(df["rate_decision"].dropna().astype(str).str.lower())
    unknown_labels = sorted(labels - KNOWN_LABELS)
    if unknown_labels:
        issues.append(f"fomc_merged.csv: unexpected labels {unknown_labels}")

    for corpus in ("statement", "minutes"):
        text_col = f"{corpus}_clean_text"
        words_col = f"{corpus}_n_words"
        empty_texts = df[text_col].fillna("").str.len().eq(0).sum()
        zero_words = df[words_col].fillna(0).eq(0).sum()
        if empty_texts:
            issues.append(f"fomc_merged.csv: empty {corpus} texts = {empty_texts}")
        if zero_words:
            issues.append(f"fomc_merged.csv: zero {corpus} word counts = {zero_words}")
    return issues


def _check_features(df: pd.DataFrame) -> list[str]:
    issues = []
    issues += _check_columns(df, FEATURE_COLUMNS | {"date"}, "fomc_features.csv")
    if issues:
        return issues
    issues += _check_dates(df, "fomc_features.csv")

    numeric_cols = sorted(FEATURE_COLUMNS)
    bad_numeric = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)
    if bad_numeric.isna().all(axis=0).any():
        empty_cols = bad_numeric.columns[bad_numeric.isna().all(axis=0)].tolist()
        issues.append(f"fomc_features.csv: fully empty feature columns {empty_cols}")
    return issues


def main() -> int:
    processed = ROOT / "data" / "processed"
    issues = []

    merged = _load_csv(processed / "fomc_merged.csv")
    issues += _check_merged(merged)

    features_path = processed / "fomc_features.csv"
    if features_path.exists():
        features = _load_csv(features_path)
        issues += _check_features(features)

    if issues:
        print("Data sanity check found issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Data sanity check passed.")
    print(f"Meetings: {len(merged)}")
    print("Rate decisions:")
    print(merged["rate_decision"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

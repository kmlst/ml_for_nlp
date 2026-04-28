"""Visualization functions for the FOMC NLP project."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fomc_nlp_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR
from .features import get_top_terms


sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _long_by_type(features: pd.DataFrame, value_suffix: str, value_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for corpus in ("statement", "minutes"):
        column = f"{corpus}_{value_suffix}"
        if column not in features:
            continue
        subset = features[["date", "year", "rate_decision", column]].copy()
        subset["document_type"] = corpus
        subset = subset.rename(columns={column: value_name})
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_documents_per_year(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save yearly document counts by corpus."""

    rows = []
    for corpus in ("statement", "minutes"):
        text_col = f"{corpus}_clean_text"
        if text_col in features:
            subset = features.loc[features[text_col].fillna("").str.len().gt(0), ["year"]].copy()
            subset["document_type"] = corpus
            rows.append(subset)
    if not rows:
        return
    counts = pd.concat(rows).groupby(["year", "document_type"]).size().reset_index(name="n_documents")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=counts, x="year", y="n_documents", hue="document_type", marker="o", ax=ax)
    ax.set_title("FOMC documents per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of documents")
    _save(fig, output_dir / "documents_per_year_by_type.png")


def plot_length_over_time_by_type(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save document length over time."""

    long_df = _long_by_type(features, "n_words", "n_words")
    if long_df.empty:
        return
    long_df["date"] = pd.to_datetime(long_df["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=long_df, x="date", y="n_words", hue="document_type", ax=ax)
    ax.set_title("Document length over time")
    ax.set_xlabel("Meeting date")
    ax.set_ylabel("Number of words")
    _save(fig, output_dir / "document_length_over_time_by_type.png")


def plot_length_distribution_by_type(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save length distributions by corpus."""

    long_df = _long_by_type(features, "n_words", "n_words")
    if long_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=long_df, x="n_words", hue="document_type", bins=30, kde=True, ax=ax)
    ax.set_title("Length distribution by document type")
    ax.set_xlabel("Number of words")
    _save(fig, output_dir / "length_distribution_by_type.png")


def plot_top_terms(features: pd.DataFrame, corpus: str, output_dir: Path = FIGURES_DIR, top_n: int = 25) -> None:
    """Save top terms for one corpus."""

    text_col = f"{corpus}_clean_text"
    if text_col not in features or features[text_col].fillna("").str.len().sum() == 0:
        return
    top_terms = get_top_terms(features[text_col].fillna(""), top_n=top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=top_terms, y="term", x="count", color="#4c78a8", ax=ax)
    ax.set_title(f"Top terms - {corpus}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    _save(fig, output_dir / f"top_terms_{corpus}s.png")


def plot_keyword_trends_by_type(keyword_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save yearly keyword trends for both corpora."""

    if keyword_df.empty:
        return
    yearly = (
        keyword_df.groupby(["year", "document_type", "keyword"], as_index=False)["frequency"]
        .mean()
        .sort_values("year")
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=yearly,
        x="year",
        y="frequency",
        hue="keyword",
        style="document_type",
        ax=ax,
    )
    ax.set_title("Keyword frequency over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean normalized frequency")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    _save(fig, output_dir / "keyword_frequency_over_time_by_type.png")


def plot_tone_scores_by_type(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save tone scores over time and by decision."""

    long_df = _long_by_type(features, "tone_score", "tone_score")
    if long_df.empty:
        return
    long_df["date"] = pd.to_datetime(long_df["date"])

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=long_df, x="date", y="tone_score", hue="document_type", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Hawkish-dovish tone score over time")
    ax.set_xlabel("Meeting date")
    ax.set_ylabel("Hawkish score - dovish score")
    _save(fig, output_dir / "tone_score_over_time_statements_minutes.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=long_df,
        x="rate_decision",
        y="tone_score",
        hue="document_type",
        errorbar="se",
        order=["cut", "hold", "hike"],
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Tone score by rate decision")
    ax.set_xlabel("Rate decision")
    ax.set_ylabel("Mean tone score")
    _save(fig, output_dir / "tone_score_by_decision_and_type.png")


def plot_uncertainty_scores_by_type(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save uncertainty scores over time."""

    long_df = _long_by_type(features, "uncertainty_score", "uncertainty_score")
    if long_df.empty:
        return
    long_df["date"] = pd.to_datetime(long_df["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=long_df, x="date", y="uncertainty_score", hue="document_type", ax=ax)
    ax.set_title("Uncertainty/risk score over time")
    ax.set_xlabel("Meeting date")
    ax.set_ylabel("Uncertainty terms per word")
    _save(fig, output_dir / "uncertainty_score_over_time_statements_minutes.png")


def plot_semantic_shift_by_type(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save TF-IDF semantic shifts over time."""

    long_df = _long_by_type(features, "semantic_shift_tfidf", "semantic_shift_tfidf")
    if long_df.empty:
        return
    long_df["date"] = pd.to_datetime(long_df["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=long_df, x="date", y="semantic_shift_tfidf", hue="document_type", ax=ax)
    ax.set_title("TF-IDF semantic shift between consecutive documents")
    ax.set_xlabel("Meeting date")
    ax.set_ylabel("1 - cosine similarity")
    _save(fig, output_dir / "semantic_shift_tfidf_statements_minutes.png")


def plot_top_semantic_shifts(
    top_shifts: pd.DataFrame,
    corpus: str,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Save top semantic-shift peaks for one corpus."""

    subset = top_shifts.loc[top_shifts["document_type"].eq(corpus)].copy()
    if subset.empty:
        return
    subset["date"] = subset["date"].astype(str)
    subset = subset.sort_values("semantic_shift_tfidf", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=subset, y="date", x="semantic_shift_tfidf", color="#f58518", ax=ax)
    ax.set_title(f"Top TF-IDF semantic shifts - {corpus}")
    ax.set_xlabel("Semantic shift")
    ax.set_ylabel("Meeting date")
    _save(fig, output_dir / f"top_semantic_shifts_{corpus}s.png")


def plot_statement_minutes_similarity(features: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Save same-meeting statement-minutes distance over time."""

    required = {"date", "same_meeting_similarity_tfidf", "same_meeting_distance_tfidf"}
    if not required.issubset(features.columns):
        return
    data = features.dropna(subset=["same_meeting_similarity_tfidf"]).copy()
    if data.empty:
        return
    data["date"] = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=data, x="date", y="same_meeting_distance_tfidf", marker="o", ax=ax)
    ax.set_title("Statement-Minutes TF-IDF distance for the same meeting")
    ax.set_xlabel("Meeting date")
    ax.set_ylabel("1 - cosine similarity")
    _save(fig, output_dir / "statement_minutes_similarity_over_time.png")


def build_all_figures(
    features: pd.DataFrame,
    keyword_df: pd.DataFrame,
    top_shifts: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Generate the full recommended figure set."""

    plot_documents_per_year(features, output_dir)
    plot_length_over_time_by_type(features, output_dir)
    plot_length_distribution_by_type(features, output_dir)
    plot_top_terms(features, "statement", output_dir)
    plot_top_terms(features, "minutes", output_dir)
    plot_keyword_trends_by_type(keyword_df, output_dir)
    plot_tone_scores_by_type(features, output_dir)
    plot_uncertainty_scores_by_type(features, output_dir)
    plot_semantic_shift_by_type(features, output_dir)
    plot_top_semantic_shifts(top_shifts, "statement", output_dir)
    plot_top_semantic_shifts(top_shifts, "minutes", output_dir)
    plot_statement_minutes_similarity(features, output_dir)

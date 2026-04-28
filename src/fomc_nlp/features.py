"""Feature engineering for FOMC statements and minutes."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DOVISH_TERMS,
    HAWKISH_TERMS,
    KEYWORD_TERMS,
    PROCESSED_DATA_DIR,
    UNCERTAINTY_TERMS,
)
from .preprocessing import clean_text, count_sentences, count_words, mask_decision_sentences


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z]){escaped}(?![a-z])", flags=re.IGNORECASE)


def count_dictionary_terms(text: str | float | None, terms: Iterable[str]) -> int:
    """Count single-word and multi-word dictionary terms in a text."""

    cleaned = clean_text(text)
    return sum(len(_term_pattern(term).findall(cleaned)) for term in terms)


def compute_dictionary_scores(
    df: pd.DataFrame,
    text_col: str,
    n_words_col: str,
    prefix: str,
    hawkish_terms: Iterable[str] = HAWKISH_TERMS,
    dovish_terms: Iterable[str] = DOVISH_TERMS,
) -> pd.DataFrame:
    """Add hawkish, dovish and net tone scores."""

    result = df.copy()
    n_words = result[n_words_col].replace(0, np.nan)
    hawkish_counts = result[text_col].map(lambda text: count_dictionary_terms(text, hawkish_terms))
    dovish_counts = result[text_col].map(lambda text: count_dictionary_terms(text, dovish_terms))
    result[f"{prefix}_hawkish_count"] = hawkish_counts
    result[f"{prefix}_dovish_count"] = dovish_counts
    result[f"{prefix}_hawkish_score"] = (hawkish_counts / n_words).fillna(0.0)
    result[f"{prefix}_dovish_score"] = (dovish_counts / n_words).fillna(0.0)
    result[f"{prefix}_tone_score"] = (
        result[f"{prefix}_hawkish_score"] - result[f"{prefix}_dovish_score"]
    )
    return result


def compute_uncertainty_scores(
    df: pd.DataFrame,
    text_col: str,
    n_words_col: str,
    prefix: str,
    uncertainty_terms: Iterable[str] = UNCERTAINTY_TERMS,
) -> pd.DataFrame:
    """Add uncertainty and risk scores."""

    result = df.copy()
    n_words = result[n_words_col].replace(0, np.nan)
    counts = result[text_col].map(lambda text: count_dictionary_terms(text, uncertainty_terms))
    result[f"{prefix}_uncertainty_count"] = counts
    result[f"{prefix}_uncertainty_score"] = (counts / n_words).fillna(0.0)
    return result


def build_tfidf_matrix(
    texts: Iterable[str],
    min_df: int = 1,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    """Build a TF-IDF matrix with English stopwords."""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    matrix = vectorizer.fit_transform([clean_text(text) for text in texts])
    return matrix, vectorizer


def compute_within_corpus_semantic_shift(
    df: pd.DataFrame,
    text_col: str,
    prefix: str,
    min_df: int = 1,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Compute 1 - cosine similarity between consecutive documents."""

    result = df.sort_values("date").reset_index(drop=True).copy()
    shift_col = f"{prefix}_semantic_shift_tfidf"
    similarity_col = f"{prefix}_successive_similarity_tfidf"
    result[shift_col] = np.nan
    result[similarity_col] = np.nan

    valid_mask = result[text_col].fillna("").str.len() > 0
    if valid_mask.sum() < 2:
        return result

    valid_index = result.index[valid_mask].to_list()
    matrix, _ = build_tfidf_matrix(result.loc[valid_index, text_col], min_df=min_df, max_features=max_features)
    for position in range(1, matrix.shape[0]):
        similarity = cosine_similarity(matrix[position], matrix[position - 1])[0, 0]
        row_index = valid_index[position]
        result.loc[row_index, similarity_col] = similarity
        result.loc[row_index, shift_col] = 1.0 - similarity
    return result


def compute_statement_minutes_similarity(
    df: pd.DataFrame,
    min_df: int = 1,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Compute TF-IDF cosine similarity between statement and minutes for each meeting."""

    result = df.copy()
    sim_col = "same_meeting_similarity_tfidf"
    dist_col = "same_meeting_distance_tfidf"
    result[sim_col] = np.nan
    result[dist_col] = np.nan

    valid = (
        result["statement_clean_text"].fillna("").str.len().gt(0)
        & result["minutes_clean_text"].fillna("").str.len().gt(0)
    )
    if not valid.any():
        return result

    valid_indices = result.index[valid].to_list()
    statement_texts = result.loc[valid_indices, "statement_clean_text"].tolist()
    minutes_texts = result.loc[valid_indices, "minutes_clean_text"].tolist()
    matrix, _ = build_tfidf_matrix(
        statement_texts + minutes_texts,
        min_df=min_df,
        max_features=max_features,
    )
    n = len(valid_indices)
    for i, row_index in enumerate(valid_indices):
        similarity = cosine_similarity(matrix[i], matrix[i + n])[0, 0]
        result.loc[row_index, sim_col] = similarity
        result.loc[row_index, dist_col] = 1.0 - similarity
    return result


def compute_keyword_frequencies(
    df: pd.DataFrame,
    corpus: str,
    keywords: Iterable[str] = KEYWORD_TERMS,
) -> pd.DataFrame:
    """Return yearly normalized keyword frequencies for one corpus."""

    text_col = f"{corpus}_clean_text"
    n_words_col = f"{corpus}_n_words"
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        n_words = row.get(n_words_col, 0) or 0
        for keyword in keywords:
            count = count_dictionary_terms(row.get(text_col, ""), [keyword])
            rows.append(
                {
                    "date": row.get("date"),
                    "year": row.get("year"),
                    "document_type": corpus,
                    "keyword": keyword,
                    "count": count,
                    "frequency": count / n_words if n_words else 0.0,
                }
            )
    return pd.DataFrame(rows)


def get_top_terms(
    texts: Iterable[str],
    top_n: int = 25,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Compute top terms by raw document-term counts."""

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1,
        max_features=10000,
    )
    matrix = vectorizer.fit_transform([clean_text(text) for text in texts])
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[top_idx], "count": counts[top_idx]})


def add_basic_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure clean, masked and length columns exist for both corpora."""

    result = df.copy()
    for corpus in ("statement", "minutes"):
        text_col = f"{corpus}_text"
        clean_col = f"{corpus}_clean_text"
        masked_col = f"{corpus}_masked_text"
        masked_clean_col = f"{corpus}_masked_clean_text"
        n_words_col = f"{corpus}_n_words"
        n_sentences_col = f"{corpus}_n_sentences"

        if text_col not in result:
            result[text_col] = ""
        result[clean_col] = result.get(clean_col, result[text_col].map(clean_text)).fillna("")
        result[masked_col] = result[text_col].map(mask_decision_sentences)
        result[masked_clean_col] = result[masked_col].map(clean_text)
        result[n_words_col] = result.get(n_words_col, result[clean_col].map(count_words)).fillna(0).astype(int)
        result[n_sentences_col] = result[text_col].map(count_sentences)
    return result


def build_feature_dataset(
    merged: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "fomc_features.csv",
    min_df: int = 1,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Create the project-level features table."""

    result = add_basic_text_columns(merged)
    for corpus in ("statement", "minutes"):
        result = compute_dictionary_scores(
            result,
            text_col=f"{corpus}_clean_text",
            n_words_col=f"{corpus}_n_words",
            prefix=corpus,
        )
        result = compute_uncertainty_scores(
            result,
            text_col=f"{corpus}_clean_text",
            n_words_col=f"{corpus}_n_words",
            prefix=corpus,
        )

    statement_shift = compute_within_corpus_semantic_shift(
        result[["date", "statement_clean_text"]].copy(),
        text_col="statement_clean_text",
        prefix="statement",
        min_df=min_df,
        max_features=max_features,
    )
    minutes_shift = compute_within_corpus_semantic_shift(
        result[["date", "minutes_clean_text"]].copy(),
        text_col="minutes_clean_text",
        prefix="minutes",
        min_df=min_df,
        max_features=max_features,
    )
    result = result.merge(
        statement_shift[["date", "statement_successive_similarity_tfidf", "statement_semantic_shift_tfidf"]],
        on="date",
        how="left",
    )
    result = result.merge(
        minutes_shift[["date", "minutes_successive_similarity_tfidf", "minutes_semantic_shift_tfidf"]],
        on="date",
        how="left",
    )
    result = compute_statement_minutes_similarity(result, min_df=min_df, max_features=max_features)
    result = result.sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def export_keyword_frequencies(
    features: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "keyword_frequencies.csv",
) -> pd.DataFrame:
    """Save normalized keyword frequencies for both corpora."""

    keyword_df = pd.concat(
        [
            compute_keyword_frequencies(features, "statement"),
            compute_keyword_frequencies(features, "minutes"),
        ],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keyword_df.to_csv(output_path, index=False)
    return keyword_df


def build_top_semantic_shifts(
    features: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "top_semantic_shifts.csv",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the largest TF-IDF semantic shifts for statements and minutes."""

    rows: list[pd.DataFrame] = []
    for corpus in ("statement", "minutes"):
        shift_col = f"{corpus}_semantic_shift_tfidf"
        tone_col = f"{corpus}_tone_score"
        uncertainty_col = f"{corpus}_uncertainty_score"
        text_col = f"{corpus}_text"
        subset = features.nlargest(top_n, shift_col)[
            ["date", "rate_decision", shift_col, tone_col, uncertainty_col, text_col]
        ].copy()
        subset = subset.rename(
            columns={
                shift_col: "semantic_shift_tfidf",
                tone_col: "tone_score",
                uncertainty_col: "uncertainty_score",
                text_col: "text",
            }
        )
        subset["document_type"] = corpus
        subset["excerpt"] = subset["text"].fillna("").str.slice(0, 500)
        rows.append(subset.drop(columns=["text"]))

    output = pd.concat(rows, ignore_index=True)
    output = output[
        [
            "document_type",
            "date",
            "rate_decision",
            "semantic_shift_tfidf",
            "tone_score",
            "uncertainty_score",
            "excerpt",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output

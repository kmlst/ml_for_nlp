"""Text cleaning and masking helpers."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

from .config import DECISION_MASK_PATTERNS


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def clean_text(text: str | float | None) -> str:
    """Normalize text while preserving economically meaningful words."""

    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text: str | float | None) -> list[str]:
    """Return lowercase alphabetic tokens."""

    return TOKEN_RE.findall(clean_text(text))


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove a compact English stopword list."""

    return [token for token in tokens if token not in STOPWORDS]


def count_words(text: str | float | None) -> int:
    """Count word-like tokens."""

    return len(tokenize_text(text))


def count_sentences(text: str | float | None) -> int:
    """Approximate sentence count for FOMC prose."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    sentences = [sentence for sentence in SENTENCE_SPLIT_RE.split(cleaned) if sentence.strip()]
    return len(sentences)


def split_sentences(text: str | float | None) -> list[str]:
    """Split text into sentence-like units."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(cleaned) if sentence.strip()]


def mask_decision_sentences(
    text: str | float | None,
    mask_token: str = "[DECISION_MASKED]",
    patterns: Iterable[str] = DECISION_MASK_PATTERNS,
) -> str:
    """Replace sentences that directly disclose the rate decision."""

    compiled = [
        re.compile(rf"\b{re.escape(pattern.lower())}\b", re.IGNORECASE)
        for pattern in patterns
    ]
    masked_sentences: list[str] = []
    for sentence in split_sentences(text):
        sentence_lower = sentence.lower()
        if any(pattern.search(sentence_lower) for pattern in compiled):
            masked_sentences.append(mask_token)
        else:
            masked_sentences.append(sentence)
    return " ".join(masked_sentences)


def strip_punctuation(text: str | float | None) -> str:
    """Remove punctuation for keyword displays and simple term counts."""

    translation = str.maketrans({char: " " for char in string.punctuation})
    return clean_text(text).translate(translation)

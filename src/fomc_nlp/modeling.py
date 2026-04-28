"""Supervised models for predicting FOMC rate decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from .config import FIGURES_DIR, PROCESSED_DATA_DIR
from .evaluation import compute_metrics, get_misclassified_examples, plot_confusion_matrix


KNOWN_LABELS = {"hike", "cut", "hold"}
MODEL_SPECS = {
    "baseline_majority": "majority",
    "logreg": "logreg",
    "svm": "svm",
}


def _corpus_plural(corpus: str) -> str:
    return "statements" if corpus == "statement" else "minutes"


def _text_column(corpus: str, text_variant: str) -> str:
    if text_variant == "masked":
        return f"{corpus}_masked_clean_text"
    return f"{corpus}_clean_text"


def _numeric_feature_columns(corpus: str) -> list[str]:
    return [
        f"{corpus}_tone_score",
        f"{corpus}_uncertainty_score",
        f"{corpus}_semantic_shift_tfidf",
        f"{corpus}_n_words",
    ]


def _valid_modeling_frame(df: pd.DataFrame, corpus: str, text_variant: str) -> pd.DataFrame:
    text_col = _text_column(corpus, text_variant)
    required = ["date", "rate_decision", text_col]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing modeling columns for {corpus}: {missing}")
    result = df.copy()
    result["rate_decision"] = result["rate_decision"].astype(str).str.lower()
    result = result.loc[result["rate_decision"].isin(KNOWN_LABELS)].copy()
    result = result.loc[result[text_col].fillna("").str.len().gt(0)].copy()
    return result.sort_values("date").reset_index(drop=True)


def _split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 6:
        raise ValueError("Not enough labeled observations to fit supervised models.")

    if split == "chronological":
        n_test = max(2, int(round(len(df) * test_size)))
        n_test = min(n_test, len(df) - 2)
        train = df.iloc[:-n_test].copy()
        test = df.iloc[-n_test:].copy()
        if train["rate_decision"].nunique() >= 2 and test["rate_decision"].nunique() >= 1:
            return train, test

    stratify = df["rate_decision"] if df["rate_decision"].value_counts().min() >= 2 else None
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )
    return (
        train.sort_values("date").reset_index(drop=True),
        test.sort_values("date").reset_index(drop=True),
    )


def _vectorize_text(train_texts, test_texts, max_features: int, min_df: int):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test, vectorizer


def _append_numeric_features(
    x_train,
    x_test,
    train: pd.DataFrame,
    test: pd.DataFrame,
    corpus: str,
):
    columns = [column for column in _numeric_feature_columns(corpus) if column in train]
    if not columns:
        return x_train, x_test
    train_numeric = train[columns].fillna(0.0).to_numpy(dtype=float)
    test_numeric = test[columns].fillna(0.0).to_numpy(dtype=float)
    return sparse.hstack([x_train, train_numeric]), sparse.hstack([x_test, test_numeric])


def build_majority_baseline():
    """Return a majority-class baseline."""

    return DummyClassifier(strategy="most_frequent")


def train_logistic_regression():
    """Return the TF-IDF + Logistic Regression classifier."""

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def train_linear_svm():
    """Return the TF-IDF + Linear SVM classifier."""

    return LinearSVC(class_weight="balanced", random_state=42)


def _make_model(model_name: str):
    if model_name == "baseline_majority":
        return build_majority_baseline()
    if model_name == "logreg":
        return train_logistic_regression()
    if model_name == "svm":
        return train_linear_svm()
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model_by_corpus(
    features: pd.DataFrame,
    corpus: str,
    text_variant: str = "complete",
    split: str = "chronological",
    include_numeric: bool = False,
    test_size: float = 0.25,
    random_state: int = 42,
    min_df: int = 1,
    max_features: int = 5000,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline, Logistic Regression and Linear SVM for one corpus."""

    df = _valid_modeling_frame(features, corpus, text_variant)
    train, test = _split_data(df, test_size=test_size, random_state=random_state, split=split)
    text_col = _text_column(corpus, text_variant)
    y_train = train["rate_decision"]
    y_test = test["rate_decision"]

    x_train, x_test, _ = _vectorize_text(
        train[text_col].fillna(""),
        test[text_col].fillna(""),
        max_features=max_features,
        min_df=min_df,
    )
    if include_numeric:
        x_train, x_test = _append_numeric_features(x_train, x_test, train, test, corpus)

    results: list[dict[str, object]] = []
    misclassified: list[pd.DataFrame] = []
    majority_class = y_train.value_counts().idxmax()

    for model_name in MODEL_SPECS:
        model = _make_model(model_name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)
        result = {
            "corpus": corpus,
            "model": model_name,
            "text_variant": text_variant,
            "split": split,
            "include_numeric": include_numeric,
            "n_train": len(train),
            "n_test": len(test),
            "majority_class_train": majority_class,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "classification_report": metrics["classification_report"],
        }
        results.append(result)

        if model_name in {"logreg", "svm"}:
            suffix = f"_{text_variant}" if text_variant != "complete" else ""
            figure_path = (
                figures_dir
                / f"confusion_matrix_{model_name}_{_corpus_plural(corpus)}{suffix}.png"
            )
            plot_confusion_matrix(
                y_test,
                y_pred,
                output_path=figure_path,
                title=f"{model_name.upper()} - {_corpus_plural(corpus)} ({text_variant})",
            )
            misclassified.append(
                get_misclassified_examples(
                    test,
                    y_test,
                    y_pred,
                    corpus=corpus,
                    text_col=text_col,
                    model_name=f"{model_name}_{text_variant}",
                )
            )

    result_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "classification_report"
            }
            | {
                "weighted_f1": result["classification_report"]
                .get("weighted avg", {})
                .get("f1-score", np.nan)
            }
            for result in results
        ]
    )
    errors = (
        pd.concat(misclassified, ignore_index=True)
        if misclassified
        else pd.DataFrame()
    )
    return result_table, errors


def compare_statement_minutes_models(
    features: pd.DataFrame,
    text_variants: tuple[str, ...] = ("complete", "masked"),
    output_results_path: Path = PROCESSED_DATA_DIR / "model_results.csv",
    output_errors_path: Path = PROCESSED_DATA_DIR / "misclassified_examples.csv",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all minimal supervised comparisons and save outputs."""

    result_tables: list[pd.DataFrame] = []
    error_tables: list[pd.DataFrame] = []
    for corpus in ("statement", "minutes"):
        for text_variant in text_variants:
            try:
                results, errors = evaluate_model_by_corpus(
                    features,
                    corpus=corpus,
                    text_variant=text_variant,
                    **kwargs,
                )
            except ValueError as exc:
                results = pd.DataFrame(
                    [
                        {
                            "corpus": corpus,
                            "model": "not_run",
                            "text_variant": text_variant,
                            "split": kwargs.get("split", "chronological"),
                            "include_numeric": kwargs.get("include_numeric", False),
                            "n_train": 0,
                            "n_test": 0,
                            "majority_class_train": "",
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "weighted_f1": np.nan,
                            "error": str(exc),
                        }
                    ]
                )
                errors = pd.DataFrame()
            result_tables.append(results)
            if not errors.empty:
                error_tables.append(errors)

    model_results = pd.concat(result_tables, ignore_index=True)
    misclassified = (
        pd.concat(error_tables, ignore_index=True)
        if error_tables
        else pd.DataFrame(
            columns=[
                "date",
                "document_type",
                "model",
                "true_label",
                "predicted_label",
                "tone_score",
                "uncertainty_score",
                "excerpt",
            ]
        )
    )
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    model_results.to_csv(output_results_path, index=False)
    misclassified.to_csv(output_errors_path, index=False)
    return model_results, misclassified


def export_modeling_datasets(
    features: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save corpus-specific modeling tables for inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for corpus in ("statement", "minutes"):
        columns = [
            "date",
            "year",
            "rate_decision",
            f"{corpus}_clean_text",
            f"{corpus}_masked_clean_text",
            f"{corpus}_tone_score",
            f"{corpus}_uncertainty_score",
            f"{corpus}_semantic_shift_tfidf",
            f"{corpus}_n_words",
        ]
        existing = [column for column in columns if column in features]
        features[existing].to_csv(output_dir / f"modeling_dataset_{_corpus_plural(corpus)}.csv", index=False)

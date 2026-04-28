"""Evaluation helpers for supervised rate-decision models."""

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


LABEL_ORDER = ["cut", "hold", "hike"]


def compute_metrics(y_true, y_pred) -> dict[str, object]:
    """Compute compact classification metrics."""

    labels = [label for label in LABEL_ORDER if label in set(y_true) | set(y_pred)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: Path,
    title: str,
    labels: list[str] | None = None,
) -> None:
    """Save a normalized confusion matrix."""

    labels = labels or [label for label in LABEL_ORDER if label in set(y_true) | set(y_pred)]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def get_misclassified_examples(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    corpus: str,
    text_col: str,
    model_name: str,
    max_examples: int = 20,
) -> pd.DataFrame:
    """Return examples where predictions differ from true labels."""

    output = df_test.copy()
    output["true_label"] = list(y_true)
    output["predicted_label"] = list(y_pred)
    output = output.loc[output["true_label"] != output["predicted_label"]].copy()
    if output.empty:
        return pd.DataFrame(
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

    output["document_type"] = corpus
    output["model"] = model_name
    output["tone_score"] = output.get(f"{corpus}_tone_score")
    output["uncertainty_score"] = output.get(f"{corpus}_uncertainty_score")
    output["excerpt"] = output[text_col].fillna("").str.slice(0, 500)
    columns = [
        "date",
        "document_type",
        "model",
        "true_label",
        "predicted_label",
        "tone_score",
        "uncertainty_score",
        "excerpt",
    ]
    return output[columns].head(max_examples).reset_index(drop=True)


def summarize_model_results(results: list[dict[str, object]]) -> pd.DataFrame:
    """Convert model run dictionaries to a clean table."""

    rows = []
    for result in results:
        report = result.get("classification_report", {})
        rows.append(
            {
                "corpus": result.get("corpus"),
                "model": result.get("model"),
                "text_variant": result.get("text_variant"),
                "split": result.get("split"),
                "n_train": result.get("n_train"),
                "n_test": result.get("n_test"),
                "majority_class_train": result.get("majority_class_train"),
                "accuracy": result.get("accuracy"),
                "macro_f1": result.get("macro_f1"),
                "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
            }
        )
    return pd.DataFrame(rows)

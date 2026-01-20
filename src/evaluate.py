from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import AUDIO_FEATURE_COLS, PATHS, RANDOM_SEED, TARGET_COL
from .data_load import load_songs

MODEL_FILES = {
    "audio_logreg": "audio_logreg.joblib",
    "audio_xgboost": "audio_xgboost.joblib",
    "lyrics_logreg": "lyrics_logreg.joblib",
    "fusion_logreg": "fusion_logreg.joblib",
}

DEFAULT_COMPARE_SET = ["audio_xgboost", "lyrics_logreg", "fusion_logreg"]

def _ensure_dirs() -> Path:
    out_dir = PATHS.reports / "figures" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _load_model(model_name: str):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES)}")
    path = PATHS.models / MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)

def _load_label_encoder():
    path = PATHS.models / "label_encoder.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Label encoder not found at: {path}")
    return joblib.load(path)

def _build_eval_data(model_name: str):
    le = _load_label_encoder()

    if model_name.startswith("audio_"):
        usecols = [TARGET_COL, *AUDIO_FEATURE_COLS]
        df = load_songs(usecols=usecols)
        X = df[AUDIO_FEATURE_COLS]
    elif model_name == "lyrics_logreg":
        usecols = [TARGET_COL, "lyrics"]
        df = load_songs(usecols=usecols)
        df["lyrics"] = df["lyrics"].fillna("").astype(str)
        X = df["lyrics"]
    elif model_name == "fusion_logreg":
        usecols = [TARGET_COL, "lyrics", *AUDIO_FEATURE_COLS]
        df = load_songs(usecols=usecols)
        df["lyrics"] = df["lyrics"].fillna("").astype(str)
        X = df[["lyrics", *AUDIO_FEATURE_COLS]]
    else:
        raise ValueError(f"Unhandled model: {model_name}")
    
    y_enc = le.transform(df[TARGET_COL])
    return X, y_enc

def _get_test_split(X, y_enc):
    _, X_test, _, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )
    return X_test, y_test

def _compute_metrics(y_test, y_pred, class_names: list[str]) -> Tuple[dict, dict]:
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": float(report["accuracy"]),
    }

    return metrics, report

def _save_normalized_confusion_matrix(out_dir: Path, y_test, y_pred, class_names, stem: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
    )
    plt.title("Normalized Confusion Matrix (row = true class)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_cm_normalized.png", dpi=240)
    plt.close(fig)

def _save_per_class_f1_bar(out_dir: Path, report: dict, stem: str) -> None:
    # extract per-class f1 and sort for readability
    labels = [k for k in report.keys() if k not in {"accuracy", "macro avg", "weighted avg"}]
    f1s = np.array([report[l]["f1-score"] for l in labels])

    order = np.argsort(f1s)  # ascending so the "hardest" classes are visible
    labels_sorted = [labels[i] for i in order]
    f1_sorted = f1s[order]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels_sorted, f1_sorted)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Per-class F1 Score (Sorted)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_f1_by_class.png", dpi=240)
    plt.close(fig)

def _save_metrics_json(out_dir: Path, stem: str, metrics: dict) -> None:
    with open(out_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def evaluate_one(model_name: str) -> None:
    out_dir = _ensure_dirs()

    model = _load_model(model_name)
    le = _load_label_encoder()
    class_names = list(le.classes_)

    X, y_enc = _build_eval_data(model_name)
    X_test, y_test = _get_test_split(X, y_enc)

    y_pred = model.predict(X_test)
    metrics, report = _compute_metrics(y_test, y_pred, class_names)

    stem = model_name

    _save_normalized_confusion_matrix(out_dir, y_test, y_pred, class_names, stem=stem)
    _save_per_class_f1_bar(out_dir, report, stem=stem)

    metrics_with_model = {"Model": model_name, **metrics}
    _save_metrics_json(out_dir, stem=stem, metrics=metrics_with_model)

    print("\nEvaluation complete:", model_name)
    print("Metrics:", metrics_with_model)
    print("Saved to:", out_dir)

def _report_to_per_class_f1(report: dict) -> Dict[str, float]:
    labels = [k for k in report.keys() if k not in {"accuracy", "macro avg", "weighted avg"}]
    return {l: float(report[l]["f1-score"]) for l in labels}

def _save_compare_macro_f1(out_dir: Path, summary_rows: List[dict]) -> None:
    names = [r["model"] for r in summary_rows]
    macro = [r["macro_f1"] for r in summary_rows]

    fig = plt.figure(figsize=(8, 4))
    plt.bar(names, macro)
    plt.ylim(0.0, 1.0)
    plt.title("Macro F1 Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_macro_f1.png", dpi=240)
    plt.close(fig)

def _save_compare_per_class_heatmap(
    out_dir: Path,
    models_in_order: List[str],
    class_names: List[str],
    per_class_f1_by_model: Dict[str, Dict[str, float]],
) -> None:
    # matrix shape: [num_classes, num_models]
    mat = np.zeros((len(class_names), len(models_in_order)), dtype=float)
    for j, m in enumerate(models_in_order):
        for i, c in enumerate(class_names):
            mat[i, j] = per_class_f1_by_model[m].get(c, 0.0)

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")

    ax.set_xticks(range(len(models_in_order)))
    ax.set_xticklabels(models_in_order, rotation=20, ha="right")

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    plt.title("Per-class F1 by Model (Heatmap)")
    plt.colorbar(im, fraction=0.03, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_dir / "compare_per_class_f1_heatmap.png", dpi=240)
    plt.close(fig)


def _save_compare_csv(out_dir: Path, summary_rows: List[dict], per_class_f1_by_model: Dict[str, Dict[str, float]]) -> None:
    # write a wide csv: model metrics + per-class f1 columns
    class_cols = sorted(next(iter(per_class_f1_by_model.values())).keys())

    out_path = out_dir / "compare_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "accuracy", "macro_f1", "weighted_f1", *[f"f1_{c}" for c in class_cols]]
        writer.writerow(header)

        for row in summary_rows:
            m = row["model"]
            per_class = per_class_f1_by_model[m]
            writer.writerow(
                [
                    m,
                    f"{row['accuracy']:.4f}",
                    f"{row['macro_f1']:.4f}",
                    f"{row['weighted_f1']:.4f}",
                    *[f"{per_class.get(c, 0.0):.4f}" for c in class_cols],
                ]
            )

    print("saved:", out_path)


def compare(models: List[str]) -> None:
    out_dir = _ensure_dirs()

    le = _load_label_encoder()
    class_names = list(le.classes_)

    summary_rows: List[dict] = []
    per_class_f1_by_model: Dict[str, Dict[str, float]] = {}

    for model_name in models:
        model = _load_model(model_name)

        X, y_enc = _build_eval_data(model_name)
        X_test, y_test = _get_test_split(X, y_enc)

        y_pred = model.predict(X_test)
        metrics, report = _compute_metrics(y_test, y_pred, class_names)

        summary = {"model": model_name, **metrics}
        summary_rows.append(summary)
        per_class_f1_by_model[model_name] = _report_to_per_class_f1(report)

    _save_compare_macro_f1(out_dir, summary_rows)
    _save_compare_per_class_heatmap(out_dir, models, class_names, per_class_f1_by_model)

    _save_compare_csv(out_dir, summary_rows, per_class_f1_by_model)

    # also save a small json summary
    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nComparison Complete")
    print("Saved to:", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate trained models and generate final figures")
    parser.add_argument("--model", choices=list(MODEL_FILES.keys()), default=None)
    parser.add_argument("--compare", action="store_true", help="compare audio/lyrics/fusion in one run")
    parser.add_argument(
        "--compare-set",
        nargs="+",
        default=None,
        help=f"models to compare (default: {DEFAULT_COMPARE_SET})",
    )

    args = parser.parse_args()

    if args.model is None and not args.compare:
        raise ValueError("provide either --model <name> or --compare")

    if args.model is not None:
        evaluate_one(args.model)

    if args.compare:
        models = args.compare_set if args.compare_set is not None else DEFAULT_COMPARE_SET
        compare(models)


if __name__ == "__main__":
    main()
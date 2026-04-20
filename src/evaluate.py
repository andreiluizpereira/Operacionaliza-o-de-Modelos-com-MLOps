from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from src.utils import ensure_dir, get_logger, load_project_config, project_root, save_json


def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run(train_result: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_project_config()
    logger = get_logger(__name__, cfg.get("logging"))

    if train_result is None:
        raise ValueError("train_result is required. Run train first.")

    best_model_name = train_result["best_model_name"]
    best_model_metadata = train_result.get("best_model_metadata", {})
    best_model = train_result["trained_models"][best_model_name]
    X_test = train_result["data"]["X_test"]
    y_test = train_result["data"]["y_test"]

    preds = best_model.predict(X_test)
    metrics = _compute_metrics(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    reports_dir = ensure_dir(project_root() / cfg.get("paths", {}).get("reports_dir", "outputs/reports"))
    report_path = reports_dir / cfg.get("modeling", {}).get("evaluation_filename", "evaluation_report.json")

    payload = {
        "best_model_name": best_model_name,
        "best_model_metadata": best_model_metadata,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "comparison": train_result["comparison"],
    }
    save_json(payload, report_path)

    logger.info("Evaluation report saved to %s", report_path)
    logger.info("Best model on holdout: %s | F1-macro=%.4f", best_model_name, metrics["f1_macro"])

    return {
        "report_path": report_path,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model_metadata": best_model_metadata,
    }

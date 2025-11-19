"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn import metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    result = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
    }
    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] > 1:
        result["roc_auc_macro"] = float(metrics.roc_auc_score(y_true, y_score, multi_class="ovr"))
        result["pr_auc_macro"] = float(metrics.average_precision_score(y_true, y_score[:, 1]))
    cm = metrics.confusion_matrix(y_true, y_pred)
    result["confusion_matrix"] = cm.tolist()
    return result

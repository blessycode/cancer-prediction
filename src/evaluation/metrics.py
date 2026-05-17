import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(metric_fn, *args, default=np.nan, **kwargs):
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        return default


def _top_k_accuracy(y_true, y_prob, k):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return float(np.mean([label in candidates for label, candidates in zip(y_true, top_k)]))


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Keep the old key for scripts that rank by f1_score.
    metrics["precision"] = metrics["precision_weighted"]
    metrics["recall"] = metrics["recall_weighted"]
    metrics["f1_score"] = metrics["f1_weighted"]

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["top_2_accuracy"] = _top_k_accuracy(y_true, y_prob, min(2, y_prob.shape[1]))
        metrics["top_3_accuracy"] = _top_k_accuracy(y_true, y_prob, min(3, y_prob.shape[1]))
        metrics["log_loss"] = float(_safe_metric(log_loss, y_true, y_prob, labels=list(range(y_prob.shape[1]))))
        metrics["roc_auc_macro"] = float(
            _safe_metric(roc_auc_score, y_true, y_prob, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_weighted"] = float(
            _safe_metric(roc_auc_score, y_true, y_prob, multi_class="ovr", average="weighted")
        )
        metrics["roc_auc"] = metrics["roc_auc_weighted"]

    target_names = class_names if class_names is not None else [str(label) for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))) if class_names is not None else labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(target_names))) if class_names is not None else labels,
    )

    return {
        "summary": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

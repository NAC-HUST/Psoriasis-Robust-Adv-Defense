from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵，行=真实类，列=预测类。"""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, object]:
    """返回 per-class 与 macro 的 precision/recall/f1。"""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    pred_sum = matrix.sum(axis=0).astype(np.float64)
    true_sum = matrix.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    return {
        "per_class": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "f1": [float(v) for v in f1],
        },
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """基于 Mann-Whitney U 的二分类 AUC（对 tie 取平均秩）。"""
    positive = y_true == 1
    n_pos = int(np.sum(positive))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(y_score.size, dtype=np.float64)

    idx = 0
    while idx < sorted_scores.size:
        end = idx
        while end + 1 < sorted_scores.size and sorted_scores[end + 1] == sorted_scores[idx]:
            end += 1
        average_rank = (idx + end) / 2.0 + 1.0
        ranks[order[idx : end + 1]] = average_rank
        idx = end + 1

    rank_sum_pos = float(np.sum(ranks[positive]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> float:
    """AUC：二分类直接算正类；多分类走 macro One-vs-Rest。

    y_score 形状：(N, num_classes) 的类别概率/分数。
    """
    y_true = y_true.astype(int)
    if y_score.ndim == 1:
        y_score = np.stack([1.0 - y_score, y_score], axis=1)

    if num_classes <= 2:
        return _binary_auc(y_true, y_score[:, 1])

    per_class_auc: list[float] = []
    for cls in range(num_classes):
        binary_true = (y_true == cls).astype(int)
        auc = _binary_auc(binary_true, y_score[:, cls])
        if not np.isnan(auc):
            per_class_auc.append(auc)
    if not per_class_auc:
        return float("nan")
    return float(np.mean(per_class_auc))


def clean_metrics(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> dict[str, object]:
    """由标签与类别分数计算干净样本指标集合。"""
    y_pred = np.argmax(y_score, axis=1)
    prf1 = precision_recall_f1(y_true, y_pred, num_classes)
    return {
        "num_samples": int(y_true.size),
        "num_classes": int(num_classes),
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": prf1["macro_precision"],
        "macro_recall": prf1["macro_recall"],
        "macro_f1": prf1["macro_f1"],
        "per_class": prf1["per_class"],
        "auc": roc_auc(y_true, y_score, num_classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }

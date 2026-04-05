from __future__ import annotations

import numpy as np


def binary_classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == targets))
    positives = predictions == 1
    true_positives = np.sum(positives & (targets == 1))
    precision = float(true_positives / max(np.sum(positives), 1))
    recall = float(true_positives / max(np.sum(targets == 1), 1))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

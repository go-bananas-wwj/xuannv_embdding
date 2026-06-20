# downstreams/tests/test_metrics.py
import numpy as np
from downstreams.metrics.segmentation import compute_segmentation_metrics


def test_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    logits = np.array([[0, 10], [10, 0]], dtype=np.float32)
    m = compute_segmentation_metrics(logits, target)
    assert m["miou"] == 1.0
    assert m["f1_0.5"] == 1.0

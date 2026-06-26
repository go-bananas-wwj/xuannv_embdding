# downstreams/tests/test_metrics.py
import numpy as np
from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.metrics.visualization import overlay_prediction, plot_pr_curve


def test_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    logits = np.array([[0, 10], [10, 0]], dtype=np.float32)
    m = compute_segmentation_metrics(logits, target)
    assert m["miou"] == 1.0
    assert m["f1_0.5"] == 1.0
    assert m["auc_roc"] == 1.0
    assert "best_threshold" in m


def test_compute_segmentation_metrics_3d() -> None:
    target = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int64,
    )
    logits = np.array(
        [
            [[0, 10], [10, 0]],
            [[10, 10], [0, 0]],
        ],
        dtype=np.float32,
    )
    m = compute_segmentation_metrics(logits, target)
    assert isinstance(m["miou"], float)
    assert isinstance(m["precision"], float)
    assert isinstance(m["recall"], float)
    assert isinstance(m["f1_0.5"], float)
    assert m["miou"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_0.5"] == 1.0


def test_ignore_index() -> None:
    target = np.array([[0, 1], [-1, 1]], dtype=np.int64)
    logits = np.array([[0, 10], [10, 10]], dtype=np.float32)
    m = compute_segmentation_metrics(logits, target, ignore_index=-1)
    # 有效像素 3 个，全部预测正确
    assert m["miou"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_all_negative() -> None:
    target = np.zeros((2, 2), dtype=np.int64)
    logits = np.full((2, 2), -10.0, dtype=np.float32)
    m = compute_segmentation_metrics(logits, target)
    assert m["miou"] == 1.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1_0.5"] == 0.0


def test_return_curve() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    logits = np.array([[0, 10], [10, 0]], dtype=np.float32)
    m = compute_segmentation_metrics(logits, target, return_curve=True)
    assert "precision_curve" in m
    assert "recall_curve" in m
    assert m["precision_curve"].size > 0
    assert m["recall_curve"].size > 0


def test_threshold_keeps_f1_05_separate() -> None:
    target = np.array([[0, 1]], dtype=np.int64)
    logits = np.array([[-0.2, 0.2]], dtype=np.float32)
    m_default = compute_segmentation_metrics(logits, target, threshold=0.5)
    m_high = compute_segmentation_metrics(logits, target, threshold=0.7)
    assert m_default["f1_0.5"] == m_high["f1_0.5"]
    assert m_high["f1_at_threshold"] < m_default["f1_at_threshold"]
    assert m_high["threshold"] == 0.7


def test_visualization_overlay() -> None:
    rgb = np.random.rand(8, 8, 3).astype(np.float32)
    pred = np.zeros((8, 8), dtype=np.uint8)
    pred[2:6, 2:6] = 1
    target = np.zeros((8, 8), dtype=np.uint8)
    target[3:7, 3:7] = 1
    overlay = overlay_prediction(rgb, pred, target)
    assert overlay.shape == rgb.shape
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0


def test_visualization_pr_curve(tmp_path) -> None:
    precision = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    recall = np.array([0.0, 0.4, 0.7, 1.0], dtype=np.float32)
    out_path = tmp_path / "pr_curve.png"
    plot_pr_curve(precision, recall, ap=0.75, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_visualization_empty_pr_curve(tmp_path) -> None:
    out_path = tmp_path / "empty_pr_curve.png"
    plot_pr_curve(np.array([]), np.array([]), ap=0.0, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

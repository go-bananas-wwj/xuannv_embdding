from __future__ import annotations

# 评估脚本测试：验证 embedding 提取脚本可正确加载配置、模型与 checkpoint 并输出 .npz。
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from torch.optim import Adam

from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import save_checkpoint


def _import_script_module(relative_path: str, module_name: str) -> types.ModuleType:
    """通过 importlib 动态加载 scripts/ 下的脚本模块。"""
    script_path = Path(__file__).parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {script_path}")
    module = types.ModuleType(spec.name)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dummy_tiff(path: Path, array: np.ndarray) -> None:
    """写入单张 GeoTIFF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(array)


def _make_dummy_dataset(tmp_path: Path, num_samples: int = 4) -> tuple[Path, Path, Path]:
    """构造临时 dataset、统计量、配置与 checkpoint，返回 (config, checkpoint, output) 路径。"""
    processed = tmp_path / "processed" / "test" / "scenes"
    processed.mkdir(parents=True)
    stats_dir = tmp_path / "statistics" / "test"
    stats_dir.mkdir(parents=True)

    height, width = 16, 16
    s2_channels = 10
    embed_dim = 8

    manifest: list[dict[str, object]] = []
    for i in range(num_samples):
        patch_id = f"2025010{i + 1:02d}_p000_r{i:03d}"
        rel_path = f"s2/s2_2025010{i + 1:02d}_p000_r{i:03d}.tif"
        array = np.random.rand(s2_channels, height, width).astype(np.float32)
        _write_dummy_tiff(processed / rel_path, array)
        manifest.append({"patch_id": patch_id, "s2": [rel_path]})

    (processed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    stats = {
        "mean": [0.0] * s2_channels,
        "std": [1.0] * s2_channels,
        "count": height * width * s2_channels,
        "band_counts": [height * width] * s2_channels,
        "num_files": num_samples,
        "source": "s2",
    }
    (stats_dir / "s2_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    config_text = f"""
experiment:
  name: eval_test
  seed: 42
data:
  root: {processed}
  region: test
  manifest_path: {processed / "manifest.json"}
  statistics_dir: {stats_dir}
  num_samples: {num_samples}
  batch_size: 2
  num_workers: 0
  sources:
    - s2
model:
  embed_dim: {embed_dim}
  sensor_channels:
    s2: {s2_channels}
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: {s2_channels}
      weight: 1.0
training:
  epochs: 1
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 0
  gradient_accumulation_steps: 1
  save_every: 1
  eval_every: 1
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    model = AEFModel(
        sensor_channels={"s2": s2_channels},
        embed_dim=embed_dim,
        target_heads={"s2_recon": ("continuous", s2_channels)},
    )
    optimizer = Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, None, 0, {})

    output_path = tmp_path / "embeddings.npz"
    return config_path, checkpoint_path, output_path


def test_extract_embeddings_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """提取脚本应在 dummy 数据上跑通并生成正确的 npz。"""
    config_path, checkpoint_path, output_path = _make_dummy_dataset(tmp_path)

    argv = [
        "extract_embeddings.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_path),
        "--device",
        "cpu",
        "--split",
        "val",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    extract_module = _import_script_module(
        "scripts/eval/extract_embeddings.py", "scripts.eval.extract_embeddings"
    )
    extract_module.main()

    assert output_path.exists()
    npz = np.load(output_path, allow_pickle=True)
    assert "embeddings" in npz
    assert "patch_ids" in npz
    assert npz["embeddings"].shape == (4, 8)
    assert npz["embeddings"].dtype == np.float32
    assert len(npz["patch_ids"]) == 4


def test_extract_embeddings_argparse() -> None:
    """评估脚本参数解析应能被测试导入并正确解析。"""
    extract_module = _import_script_module(
        "scripts/eval/extract_embeddings.py", "scripts.eval.extract_embeddings"
    )
    args = extract_module.parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pt", "--output", "out.npz"]
    )
    assert args.config == "cfg.yaml"
    assert args.checkpoint == "ckpt.pt"
    assert args.output == "out.npz"
    assert args.split == "train"
    assert args.include_maps is False


def test_knn_eval_end_to_end(tmp_path: Path) -> None:
    """KNN 评估脚本应能在随机数据上跑通并输出指标。"""
    num_train, num_test, dim = 32, 8, 16
    rng = np.random.default_rng(42)

    train_embeddings = rng.random((num_train, dim)).astype(np.float32)
    test_embeddings = rng.random((num_test, dim)).astype(np.float32)
    train_labels = rng.integers(0, 3, size=num_train).astype(np.int64)
    test_labels = rng.integers(0, 3, size=num_test).astype(np.int64)

    embedding_train_path = tmp_path / "train.npz"
    embedding_test_path = tmp_path / "test.npz"
    label_train_path = tmp_path / "labels_train.npz"
    label_test_path = tmp_path / "labels_test.npz"
    output_path = tmp_path / "knn_metrics.npz"

    np.savez(embedding_train_path, embeddings=train_embeddings)
    np.savez(embedding_test_path, embeddings=test_embeddings)
    np.savez(label_train_path, labels=train_labels)
    np.savez(label_test_path, labels=test_labels)

    knn_module = _import_script_module("scripts/eval/knn_eval.py", "scripts.eval.knn_eval")
    knn_module._main(
        [
            "--embedding-train",
            str(embedding_train_path),
            "--label-train",
            str(label_train_path),
            "--embedding-test",
            str(embedding_test_path),
            "--label-test",
            str(label_test_path),
            "--k",
            "3",
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    metrics = np.load(output_path)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "predictions" in metrics
    assert "labels" in metrics
    assert metrics["predictions"].shape == metrics["labels"].shape
    assert float(metrics["accuracy"]) >= 0.0


def test_change_detection_eval_end_to_end(tmp_path: Path) -> None:
    """变化检测评估脚本应能计算 AUC 并保存结果。"""
    num_samples, dim = 32, 16
    rng = np.random.default_rng(24)

    before = rng.random((num_samples, dim)).astype(np.float32)
    after = rng.random((num_samples, dim)).astype(np.float32)
    labels = rng.integers(0, 2, size=num_samples).astype(np.int64)
    # 确保同时包含正负样本，否则 roc_auc_score 会报错。
    labels[0] = 0
    labels[1] = 1

    before_path = tmp_path / "before.npz"
    after_path = tmp_path / "after.npz"
    label_path = tmp_path / "labels.npz"
    output_path = tmp_path / "cd_metrics.npz"

    np.savez(before_path, embeddings=before)
    np.savez(after_path, embeddings=after)
    np.savez(label_path, labels=labels)

    cd_module = _import_script_module(
        "scripts/eval/change_detection_eval.py", "scripts.eval.change_detection_eval"
    )
    cd_module._main(
        [
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--label",
            str(label_path),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    metrics = np.load(output_path)
    assert "auc" in metrics
    assert "diff" in metrics
    assert "labels" in metrics
    assert metrics["diff"].shape == metrics["labels"].shape
    auc_value = float(metrics["auc"])
    assert 0.0 <= auc_value <= 1.0

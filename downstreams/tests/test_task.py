from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask


def test_task_build() -> None:
    cfg = {"head_type": "linear", "embed_dim": 64, "num_classes": 2, "loss": "focal_dice"}
    task = ConstructionSegmentationTask(cfg)
    head = task.build_head()
    assert head is not None

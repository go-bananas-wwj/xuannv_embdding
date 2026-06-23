from downstreams.heads.base import TaskHead
from downstreams.heads.change_detection_head import ChangeDetectionHead
from downstreams.heads.classification_head import ClassificationHead
from downstreams.heads.linear_probe import LinearProbeHead
from downstreams.heads.segmentation_head import (
    FCNHead,
    UNetHead,
    UperNetHead,
    build_segmentation_head,
)

__all__ = [
    "TaskHead",
    "LinearProbeHead",
    "FCNHead",
    "UNetHead",
    "UperNetHead",
    "build_segmentation_head",
    "ClassificationHead",
    "ChangeDetectionHead",
]

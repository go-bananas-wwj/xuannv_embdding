# downstreams/tests/test_label_loaders.py
from pathlib import Path

from downstreams.data.label_loaders import parse_patch_id_from_labelme_name


def test_parse_patch_id() -> None:
    path = Path("patch_000002_20260430_rgb_uint8.json")
    assert parse_patch_id_from_labelme_name(path) == "patch_000002"

from __future__ import annotations

import numpy as np

from scripts.eval import run_registered_paper_downstream as registered


def test_target_support_hash_binds_shape_dtype_and_each_pixel() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    digest = registered.target_support_sha256(target)

    assert digest == registered.target_support_sha256(target.copy())
    assert digest != registered.target_support_sha256(np.array([[0, 1], [0, 0]], dtype=np.uint8))
    assert digest == registered.target_support_sha256(target.astype(np.float32))

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/report/build_pu_query_3polygon_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_pu_query_3polygon_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_payload() -> dict[str, object]:
    rows = []
    for task in ("building", "road", "water"):
        for index, feature in enumerate(("xuannv", "aef", "traditional")):
            rows.append(
                {
                    "task": task,
                    "task_zh": {"building": "建筑物", "road": "道路", "water": "水体"}[task],
                    "feature": feature,
                    "polygon_count": 3,
                    "support_patch_ids": ["p1", "p2", "p3"],
                    "test_patch_ids": ["t1", "t2"],
                    "test_patch_count": 2,
                    "metrics": {
                        "f1": 0.2 + index * 0.01,
                        "auc": 0.7 + index * 0.01,
                        "ap": 0.1 + index * 0.01,
                    },
                }
            )
    return {
        "protocol": {
            "polygon_count": 3,
            "fold": 0,
            "shared_supports": True,
            "test_patch_count": 2,
        },
        "rows": rows,
    }


def test_validate_payload_requires_nine_fair_rows() -> None:
    payload = sample_payload()

    MODULE.validate_payload(payload)

    payload["rows"][1]["support_patch_ids"] = ["other"]
    try:
        MODULE.validate_payload(payload)
    except ValueError as error:
        assert "shared support" in str(error)
    else:
        raise AssertionError("mismatched supports must fail validation")


def test_build_metric_outputs_creates_figure_and_markdown(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(sample_payload()), encoding="utf-8")
    assets = tmp_path / "assets"
    report = tmp_path / "report.md"

    MODULE.build_metric_outputs(results, assets, report)

    assert (assets / "pu_query_3polygon_metrics.png").exists()
    text = report.read_text(encoding="utf-8")
    assert "只标注 3 个目标多边形" in text
    assert "玄女 P10C 2026-04 月度嵌入" in text
    assert "AEF 2025 年度嵌入" in text
    assert "传统 2026-04 多源特征" in text
    assert "F1" in text and "AUC" in text and "AP" in text

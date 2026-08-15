from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
import yaml

from experiments.china_v1_fusion_smoke.archive import (
    MonthRef,
    PatchSelection,
    RasterHeader,
)
from experiments.china_v1_fusion_smoke.data import PatchYearBatch
from experiments.china_v1_fusion_smoke.registry import RegistryError
from experiments.china_v1_fusion_smoke.runner import RunnerError, main, run_stage
from experiments.china_v1_fusion_smoke.synthetic import SyntheticAnnualContext


@dataclass(frozen=True)
class RunnerFixture:
    config_path: Path
    sandbox_root: Path
    source_root: Path
    selections: tuple[PatchSelection, ...]


def _expanded(value: float, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype).expand(shape)


@pytest.fixture
def runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunnerFixture:
    """用真实 Runner 写路径和模型合同，替换重复且昂贵的 ZIP/TIFF 解码。"""
    sandbox_root = tmp_path / "sandbox"
    source_root = tmp_path / "source"
    sandbox_root.mkdir()
    source_root.mkdir()
    (sandbox_root / ".xuannv_isolated_smoke").touch()

    archives: dict[tuple[str, int, int], Path] = {}
    for sensor in ("pc-s1", "pc-s2"):
        for year in (2020, 2021):
            for month in range(1, 13):
                archive = (
                    source_root
                    / sensor
                    / str(year)
                    / f"{month:02d}"
                    / f"{sensor}_{year}_{month:02d}.zip"
                )
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(f"{sensor}-{year}-{month}".encode())
                archives[sensor, year, month] = archive

    patch_ids = tuple(f"parent_32643:{310 + index}:{3383 + index}" for index in range(4))
    selections = tuple(
        PatchSelection(
            patch_id=patch_id,
            months=tuple(
                MonthRef(
                    sensor=sensor,
                    year=year,
                    month=month,
                    archive=archives[sensor, year, month],
                    member=f"{sensor}/{year}/{month:02d}/{patch_id}.tif",
                    crc=1000 + month,
                )
                for sensor in ("pc-s1", "pc-s2")
                for year in (2020, 2021)
                for month in range(1, 13)
            ),
        )
        for patch_id in patch_ids
    )

    checked = Path("configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml")
    raw = yaml.safe_load(checked.read_text(encoding="utf-8"))
    raw["sandbox"]["root"] = str(sandbox_root)
    raw["sandbox"]["source_root"] = str(source_root)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    import experiments.china_v1_fusion_smoke.config as config_module
    import experiments.china_v1_fusion_smoke.runner as runner_module

    monkeypatch.setattr(config_module, "EXPECTED_SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr(config_module, "EXPECTED_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        runner_module,
        "select_complete_patches",
        lambda _root, count=4: selections[:count],
    )

    def fake_headers(selection: PatchSelection) -> tuple[RasterHeader, ...]:
        return tuple(
            RasterHeader(
                sensor=month.sensor,
                year=month.year,
                month=month.month,
                width=128,
                height=128,
                count=2 if month.sensor == "pc-s1" else 10,
                dtype="float32",
                crs="EPSG:32643",
                transform=(10.0, 0.0, 310000.0, 0.0, -10.0, 3384000.0, 0.0, 0.0, 1.0),
            )
            for month in selection.months
        )

    monkeypatch.setattr(runner_module, "inspect_selection_headers", fake_headers)
    batch = PatchYearBatch(
        s2=_expanded(0.25, (8, 4, 3, 10, 128, 128), dtype=torch.float32),
        s1=_expanded(-12.0, (8, 4, 3, 2, 128, 128), dtype=torch.float32),
        valid_s2=_expanded(1, (8, 4, 3, 1, 128, 128), dtype=torch.bool),
        valid_s1=_expanded(1, (8, 4, 3, 1, 128, 128), dtype=torch.bool),
        patch_ids=tuple(patch_id for patch_id in patch_ids for _ in (2020, 2021)),
        years=(2020, 2021) * 4,
    )
    monkeypatch.setattr(runner_module, "load_patch_year_batch", lambda *_args, **_kwargs: batch)

    real_metadata = {
        "aef": {
            "synthetic": True,
            "synthetic_kind": "annual_s2_fixed_projection",
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
        "highres": {
            "synthetic": True,
            "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "claimed_native_gsd_m": None,
            "model_input_gsd_m": 2,
            "contains_real_2m_information": False,
        },
    }

    def fake_context(*_args, **_kwargs) -> SyntheticAnnualContext:
        return SyntheticAnnualContext(
            aef=_expanded(0.125, (64, 128, 128), dtype=torch.float32),
            aef_valid=_expanded(1, (1, 128, 128), dtype=torch.bool),
            highres=_expanded(0.25, (3, 640, 640), dtype=torch.float32),
            highres_valid=_expanded(1, (1, 640, 640), dtype=torch.bool),
            metadata=real_metadata,
        )

    monkeypatch.setattr(runner_module, "generate_synthetic_context", fake_context)
    return RunnerFixture(config_path, sandbox_root, source_root, selections)


def _rewrite_runtime(config_path: Path, **updates: object) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["runtime"].update(updates)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_cpu_contract_runs_all_groups_inside_sandbox(runner_fixture: RunnerFixture) -> None:
    """遗漏任一分支或把合同证据写出沙箱都必须使 Runner 测试失败。"""
    result = run_stage(runner_fixture.config_path, stage="cpu-contract")

    assert set(result.groups) == {"base", "base_aef", "base_highres", "full"}
    assert result.full_zero_gate_matches_base
    assert result.sandbox_root == runner_fixture.sandbox_root
    assert result.created_paths
    assert all(path.is_relative_to(result.sandbox_root) for path in result.created_paths)
    manifest = json.loads(
        (runner_fixture.sandbox_root / "manifests" / "cpu_contract.json").read_text()
    )
    assert manifest["accuracy_conclusion_allowed"] is False
    assert manifest["groups"]["full"]["shape"] == [2, 4, 64, 16, 16]


def test_inspect_selects_exactly_four_and_does_not_write(runner_fixture: RunnerFixture) -> None:
    """inspect 只返回稳定选择及代表性头，不得产生沙箱文件。"""
    result = run_stage(runner_fixture.config_path, stage="inspect")

    assert result.patch_ids == tuple(item.patch_id for item in runner_fixture.selections)
    assert result.patch_years == 8
    assert len(result.headers) == 16
    assert result.created_paths == ()


def test_prepare_writes_eight_marked_contexts_and_unchanged_source_audit(
    runner_fixture: RunnerFixture,
) -> None:
    """prepare 必须固定四 patch/八 patch-year，且缓存与 registry 明示 synthetic。"""
    result = run_stage(runner_fixture.config_path, stage="prepare")

    assert result.patch_ids == tuple(item.patch_id for item in runner_fixture.selections)
    assert result.patch_years == 8
    assert result.source_unchanged is True
    assert result.sandbox_bytes < 5 * 1024**3
    selection = json.loads(
        (runner_fixture.sandbox_root / "manifests" / "patch_selection.json").read_text()
    )
    assert selection["patch_ids"] == list(result.patch_ids)
    assert len(selection["patches"]) == 4
    assert all(len(item["months"]) == 48 for item in selection["patches"])
    prepare_manifest = json.loads(
        (runner_fixture.sandbox_root / "manifests" / "prepare_manifest.json").read_text()
    )
    assert prepare_manifest["patch_years"] == 8
    assert prepare_manifest["source_unchanged"] is True
    for name in ("aef_registry.json", "highres_2m_registry.json"):
        registry = json.loads((runner_fixture.sandbox_root / "manifests" / name).read_text())
        assert registry["synthetic"] is True
        assert registry["formal_training_allowed"] is False
        assert registry["formal_evaluation_allowed"] is False
        assert len(registry["entries"]) == 8
    audit = json.loads((runner_fixture.sandbox_root / "path_audit.json").read_text())
    assert audit["stage"] == "prepare"
    assert all(
        (runner_fixture.sandbox_root / relative).is_relative_to(runner_fixture.sandbox_root)
        for relative in audit["created_or_modified"]
    )


def test_invalid_stage_is_rejected_before_dispatch(runner_fixture: RunnerFixture) -> None:
    """拼错的阶段不得静默落到任何可写或设备路径。"""
    with pytest.raises(ValueError, match="unknown smoke stage"):
        run_stage(runner_fixture.config_path, stage="train")

    with pytest.raises(SystemExit) as raised:
        main(["--config", str(runner_fixture.config_path), "--stage", "train"])
    assert raised.value.code == 2


def test_missing_sentinel_fails_before_any_stage_work(runner_fixture: RunnerFixture) -> None:
    """没有隔离 sentinel 时即使 inspect 不写文件也必须 fail closed。"""
    (runner_fixture.sandbox_root / ".xuannv_isolated_smoke").unlink()

    with pytest.raises(Exception, match="sentinel"):
        run_stage(runner_fixture.config_path, stage="inspect")


@pytest.mark.parametrize("updates", [{"max_patches": 5}, {"years": [2020, 2022]}])
def test_runtime_scale_cannot_exceed_four_patches_or_leave_fixed_years(
    runner_fixture: RunnerFixture,
    updates: dict[str, object],
) -> None:
    """Runner 必须重新经过 Task 1 的固定规模校验，不能信任调用方参数。"""
    _rewrite_runtime(runner_fixture.config_path, **updates)

    with pytest.raises(ValueError):
        run_stage(runner_fixture.config_path, stage="inspect")


def test_selector_returning_more_than_four_patches_is_rejected(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即使 archive 依赖违约多返回一个 patch，Runner 也不能扩大范围。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    extra = replace(runner_fixture.selections[-1], patch_id="parent_32643:999:9999")
    monkeypatch.setattr(
        runner_module,
        "select_complete_patches",
        lambda _root, count=4: (*runner_fixture.selections, extra),
    )

    with pytest.raises(RunnerError, match="exactly four"):
        run_stage(runner_fixture.config_path, stage="inspect")


def test_prepare_rejects_a_changed_source_snapshot(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare 期间任何源 ZIP size/mtime 变化都必须触发 stop rule。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    original = runner_module._snapshot_source_archives
    calls = 0

    def changed_snapshot(source_root: Path):
        nonlocal calls
        calls += 1
        snapshot = original(source_root)
        if calls == 2:
            first = next(iter(snapshot))
            snapshot[first] = {**snapshot[first], "mtime_ns": snapshot[first]["mtime_ns"] + 1}
        return snapshot

    monkeypatch.setattr(runner_module, "_snapshot_source_archives", changed_snapshot)

    with pytest.raises(RunnerError, match="source archive snapshot changed"):
        run_stage(runner_fixture.config_path, stage="prepare")
    assert not (runner_fixture.sandbox_root / "manifests" / "prepare_manifest.json").exists()


def test_prepare_rejects_missing_synthetic_flags(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旁路 metadata 少一个 synthetic 用途字段也不得写成可用 preflight。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    valid = runner_module.generate_synthetic_context(None, None, None, None, None)
    malformed_aef = dict(valid.metadata["aef"])
    malformed_aef.pop("synthetic")
    monkeypatch.setattr(
        runner_module,
        "generate_synthetic_context",
        lambda *_args, **_kwargs: replace(
            valid,
            metadata={"aef": malformed_aef, "highres": valid.metadata["highres"]},
        ),
    )

    with pytest.raises(RegistryError, match="synthetic"):
        run_stage(runner_fixture.config_path, stage="prepare")
    assert not (runner_fixture.sandbox_root / "manifests" / "prepare_manifest.json").exists()


def test_npu_smoke_requires_the_task7_launcher_environment(runner_fixture: RunnerFixture) -> None:
    """直接 Python 调用不得绕过 Task 7 对物理 NPU 2 的启动守卫。"""
    with pytest.raises(RunnerError, match="Task 7 launcher"):
        run_stage(runner_fixture.config_path, stage="npu-smoke")

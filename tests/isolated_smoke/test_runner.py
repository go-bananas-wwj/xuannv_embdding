from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
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


def test_path_audit_accumulates_prepare_and_cpu_contract_paths(
    runner_fixture: RunnerFixture,
) -> None:
    """后续阶段不得覆盖 prepare 的既有写路径 provenance。"""
    run_stage(runner_fixture.config_path, stage="prepare")
    run_stage(runner_fixture.config_path, stage="cpu-contract")

    audit = json.loads((runner_fixture.sandbox_root / "path_audit.json").read_text())
    assert audit["stages"] == ["prepare", "cpu-contract"]
    assert "synthetic/aef/patch_00_2020.pt" in audit["created_or_modified"]
    assert "manifests/cpu_contract.json" in audit["created_or_modified"]


def test_npu_path_audit_records_expected_success_before_sealing(
    runner_fixture: RunnerFixture,
) -> None:
    """finalizer 必须在 tee 结束后捕获最终日志，再声明最终 SUCCESS。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    launcher_log = runner_fixture.sandbox_root / "logs" / "npu_smoke.log"
    launcher_log.parent.mkdir()
    launcher_log.write_text("compute output\n", encoding="utf-8")
    before_compute = runner_module._snapshot_sandbox(runner_fixture.sandbox_root)
    (runner_fixture.sandbox_root / "metrics.json").write_text("{}\n", encoding="utf-8")
    runner_module._record_path_audit(
        "npu-smoke",
        runner_fixture.sandbox_root,
        before_compute,
    )
    with launcher_log.open("a", encoding="utf-8") as handle:
        handle.write("final runner JSON from tee\n")
    before_finalize = runner_module._snapshot_sandbox(runner_fixture.sandbox_root)
    success = runner_fixture.sandbox_root / "SUCCESS"
    created = runner_module._record_path_audit(
        "finalize-seal",
        runner_fixture.sandbox_root,
        before_finalize,
        final_seal_path=success,
        declared_paths=(launcher_log,),
    )

    audit = json.loads((runner_fixture.sandbox_root / "path_audit.json").read_text())
    assert not success.exists()
    assert launcher_log in created
    assert success in created
    assert "logs/npu_smoke.log" in audit["created_or_modified"]
    assert "SUCCESS" in audit["created_or_modified"]
    assert audit["stage"] == "finalize-seal"
    assert audit["stages"] == ["npu-smoke", "finalize-seal"]
    assert audit["declared_files"]["logs/npu_smoke.log"] == {
        "size": launcher_log.stat().st_size,
        "mtime_ns": launcher_log.stat().st_mtime_ns,
        "sha256": sha256(launcher_log.read_bytes()).hexdigest(),
    }
    assert audit["final_seal"] == {
        "path": "SUCCESS",
        "status": "expected_last_write",
        "exists_when_audit_written": False,
    }


def _write_preliminary_ready(runner_fixture: RunnerFixture) -> tuple[Path, Path]:
    """构造 compute 已完成、但 tee 尚未确认的真实文件状态。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    log = runner_fixture.sandbox_root / "logs" / "npu_smoke.log"
    log.parent.mkdir()
    log.write_text("complete foreground output\n", encoding="utf-8")
    audit = runner_fixture.sandbox_root / "path_audit.json"
    audit.write_text(
        json.dumps(
            {
                "stage": "npu-smoke",
                "stages": ["npu-smoke"],
                "sandbox_root": str(runner_fixture.sandbox_root),
                "created_or_modified": ["logs/npu_smoke.log", "path_audit.json"],
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ready = runner_fixture.sandbox_root / "READY_TO_SEAL"
    ready.write_text(
        json.dumps(
            {
                "status": "npu_compute_complete",
                "git_commit": "a" * 40,
                "config_sha256": runner_module._sha256_file(runner_fixture.config_path),
                "preliminary_path_audit_sha256": runner_module._sha256_file(audit),
                "source_unchanged": True,
                "synthetic": True,
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return log, ready


def test_mark_tee_complete_atomically_binds_the_finished_launcher_log(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只有 pipeline 返回后的独立动作才能把最终日志 metadata 绑定到 marker。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    log, ready = _write_preliminary_ready(runner_fixture)
    monkeypatch.setattr(runner_module, "_validate_npu_launcher", lambda _config: None)
    monkeypatch.setattr(runner_module, "_git_commit", lambda: "a" * 40)

    marker = runner_module._mark_tee_complete(runner_fixture.config_path)

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker == runner_fixture.sandbox_root / "TEE_COMPLETE"
    assert payload["status"] == "tee_pipeline_complete"
    assert payload["ready_to_seal_sha256"] == sha256(ready.read_bytes()).hexdigest()
    assert payload["launcher_log"] == {
        "path": "logs/npu_smoke.log",
        "size": log.stat().st_size,
        "mtime_ns": log.stat().st_mtime_ns,
        "sha256": sha256(log.read_bytes()).hexdigest(),
    }
    assert not marker.with_name("TEE_COMPLETE.partial").exists()


def test_mark_tee_complete_refuses_recovery_without_ready(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEE marker 不能在 READY 缺失时凭空创建。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    monkeypatch.setattr(runner_module, "_validate_npu_launcher", lambda _config: None)

    with pytest.raises(RunnerError, match="READY_TO_SEAL"):
        runner_module._mark_tee_complete(runner_fixture.config_path)


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
            archives = snapshot["archives"]
            assert isinstance(archives, dict)
            first = next(iter(archives))
            archives[first] = {
                **archives[first],
                "mtime_ns": archives[first]["mtime_ns"] + 1,
            }
        return snapshot

    monkeypatch.setattr(runner_module, "_snapshot_source_archives", changed_snapshot)

    with pytest.raises(RunnerError, match="source archive snapshot changed"):
        run_stage(runner_fixture.config_path, stage="prepare")
    assert not (runner_fixture.sandbox_root / "manifests" / "prepare_manifest.json").exists()


def test_prepare_rejects_new_direct_entry_in_canonical_archive_directory(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare 间隙新增 lock/index/temp/completion entry 必须触发源目录 stop rule。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    original_loader = runner_module.load_patch_year_batch

    def load_then_create_forbidden_entry(*args, **kwargs):
        batch = original_loader(*args, **kwargs)
        parent = runner_fixture.source_root / "pc-s2/2020/01"
        (parent / ".forbidden.lock").write_text("must be detected", encoding="utf-8")
        return batch

    monkeypatch.setattr(runner_module, "load_patch_year_batch", load_then_create_forbidden_entry)

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


def _set_task7_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("XUANNV_SMOKE_DEVICE", "npu:0")
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_missing_fuser_falls_back_to_an_idle_proc_device_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有 fuser 时，完整可读且无 owner 的 /proc 扫描必须明确返回 idle。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)

    assert runner_module._physical_device_is_idle(Path("/dev/null"), proc_root=proc_root)


def test_missing_fuser_proc_scan_detects_owner_by_character_device_rdev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fallback 必须按字符设备 st_rdev 识别 owner，而不是比较 fd symlink 字符串。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    fd_root = proc_root / "123" / "fd"
    fd_root.mkdir(parents=True)
    fd_root.joinpath("7").symlink_to("/dev/null")
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)

    assert not runner_module._physical_device_is_idle(Path("/dev/null"), proc_root=proc_root)


def test_missing_fuser_proc_scan_fails_closed_on_unreadable_process_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """任何无法读取的进程 fd 目录都必须作为 unknown fail closed，而非误报 idle。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    fd_root = proc_root / "123" / "fd"
    fd_root.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def deny_target_fd_directory(path: Path):
        if path == fd_root:
            raise PermissionError("fixture denies fd scan")
        return original_iterdir(path)

    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "iterdir", deny_target_fd_directory)

    with pytest.raises(RunnerError, match="cannot independently scan process file descriptors"):
        runner_module._physical_device_is_idle(Path("/dev/null"), proc_root=proc_root)


def test_npu_smoke_rejects_manually_set_environment_without_launcher_ancestor(
    runner_fixture: RunnerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """普通 Python 即使伪造四个环境变量，也不能冒充固定 Task 7 launcher。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    _set_task7_environment(monkeypatch)
    monkeypatch.setattr(runner_module, "_is_expected_smoke_interpreter", lambda _config: True)
    monkeypatch.setattr(runner_module, "_physical_npu2_exists", lambda: True)
    monkeypatch.setattr(runner_module, "_physical_npu2_is_idle", lambda: True)
    monkeypatch.setattr(runner_module, "_has_task7_launcher_ancestor", lambda: False)

    with pytest.raises(RunnerError, match="verified Task 7 launcher ancestor"):
        run_stage(runner_fixture.config_path, stage="npu-smoke")


def test_launcher_ancestor_verification_reads_fixed_script_from_proc(
    tmp_path: Path,
) -> None:
    """祖先进程必须是 bash 执行固定 Task 7 脚本，不能只检查任意父 PID 存在。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(
        b"/usr/bin/bash\0" + str(runner_module.TASK7_LAUNCHER).encode() + b"\0"
    )
    process.joinpath("status").write_text("Name:\tbash\nPPid:\t1\n", encoding="utf-8")
    process.joinpath("cwd").symlink_to(runner_module.EXPECTED_WORKTREE, target_is_directory=True)

    assert runner_module._has_task7_launcher_ancestor(start_pid=123, proc_root=proc_root)


def test_launcher_ancestor_accepts_fixed_relative_script_operand(tmp_path: Path) -> None:
    """合法 Task 7 argv 必须把相对 launcher 路径放在 bash 的实际脚本位置。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(
        b"/usr/bin/bash\0scripts/smoke/run_china_v1_isolated_fusion_smoke.sh\0"
    )
    process.joinpath("status").write_text("Name:\tbash\nPPid:\t1\n", encoding="utf-8")
    process.joinpath("cwd").symlink_to(runner_module.EXPECTED_WORKTREE, target_is_directory=True)

    assert runner_module._has_task7_launcher_ancestor(start_pid=123, proc_root=proc_root)


@pytest.mark.parametrize(
    "command",
    [
        (b"/usr/bin/bash\0-c\0exit 0\0" b"scripts/smoke/run_china_v1_isolated_fusion_smoke.sh\0"),
        (b"/usr/bin/bash\0-s\0" b"scripts/smoke/run_china_v1_isolated_fusion_smoke.sh\0"),
        (
            b"/usr/bin/bash\0scripts/smoke/not-the-launcher.sh\0"
            b"scripts/smoke/run_china_v1_isolated_fusion_smoke.sh\0"
        ),
    ],
    ids=("bash-c-dollar-zero", "bash-s-argument", "trailing-decoy"),
)
def test_launcher_ancestor_rejects_non_script_operand_decoys(
    tmp_path: Path, command: bytes
) -> None:
    """固定路径只作为 -c/-s 参数或后置参数时不能冒充实际执行脚本。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(command)
    process.joinpath("status").write_text("Name:\tbash\nPPid:\t1\n", encoding="utf-8")
    process.joinpath("cwd").symlink_to(runner_module.EXPECTED_WORKTREE, target_is_directory=True)

    assert not runner_module._has_task7_launcher_ancestor(start_pid=123, proc_root=proc_root)


@pytest.mark.parametrize(
    ("device_exists", "device_idle", "message"),
    [(False, True, "physical /dev/davinci2 is absent"), (True, False, "NPU 2 is busy")],
    ids=("missing-device", "busy-device"),
)
def test_npu_smoke_independently_checks_physical_device_and_idle_state(
    runner_fixture: RunnerFixture,
    monkeypatch: pytest.MonkeyPatch,
    device_exists: bool,
    device_idle: bool,
    message: str,
) -> None:
    """launcher shell 的检查不能替代 Runner 自己的设备存在性与 fuser 空闲检查。"""
    import experiments.china_v1_fusion_smoke.runner as runner_module

    _set_task7_environment(monkeypatch)
    monkeypatch.setattr(runner_module, "_is_expected_smoke_interpreter", lambda _config: True)
    monkeypatch.setattr(runner_module, "_has_task7_launcher_ancestor", lambda: True)
    monkeypatch.setattr(runner_module, "_physical_npu2_exists", lambda: device_exists)
    monkeypatch.setattr(runner_module, "_physical_npu2_is_idle", lambda: device_idle)

    with pytest.raises(RunnerError, match=message):
        run_stage(runner_fixture.config_path, stage="npu-smoke")

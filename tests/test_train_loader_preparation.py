from __future__ import annotations

from scripts.train import train as train_module


class _RawLoader:
    sampler = object()

    def __iter__(self):
        yield {"raw": 1}
        yield {"raw": 2}

    def __len__(self) -> int:
        return 2


def test_prepared_loader_runs_batch_transform_during_iteration(monkeypatch) -> None:
    calls = []

    def fake_prepare(batch, target_heads, **kwargs):
        calls.append((batch, target_heads, kwargs))
        return {"prepared": batch["raw"]}

    monkeypatch.setattr(train_module, "prepare_batch", fake_prepare)
    raw_loader = _RawLoader()
    loader = train_module._PreparedLoader(
        raw_loader,
        {"target": {"channels": 1}},
        {"s1": 0.2},
        {"enabled": True},
        False,
    )

    assert list(loader) == [{"prepared": 1}, {"prepared": 2}]
    assert len(loader) == 2
    assert loader.sampler is raw_loader.sampler
    assert len(calls) == 2
    assert calls[0][2] == {
        "source_dropout_probs": {"s1": 0.2},
        "input_masking": {"enabled": True},
        "keep_highres_inputs": False,
    }

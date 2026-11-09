from pathlib import Path

import huggingface_hub
import pytest

from app.retrieval import MODEL_FILES, MODEL_REVISION, model_snapshot


@pytest.mark.parametrize("complete", [True, False])
def test_loads_complete_cache_or_fetches_missing_files(tmp_path, monkeypatch, complete):
    for name in MODEL_FILES:
        if complete or name != "1_Pooling/config.json":
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    calls = []

    def download(model, **options):
        assert options["revision"] == MODEL_REVISION
        calls.append(options)
        if not options.get("local_files_only"):
            assert set(options["allow_patterns"]) == set(MODEL_FILES)
            pooling = tmp_path / "1_Pooling/config.json"
            pooling.parent.mkdir(exist_ok=True)
            pooling.touch()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert model_snapshot() == Path(tmp_path)
    assert len(calls) == (1 if complete else 2)
    assert calls[0]["local_files_only"] is True

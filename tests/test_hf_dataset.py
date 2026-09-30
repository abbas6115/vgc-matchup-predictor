import pytest
from pathlib import Path
from src.data.download_hf import download_hf_dataset
from src.utils.config import load_config

def test_download_hf_dataset_is_not_empty():
    dataset = download_hf_dataset()

    
    # Check that the dataset is not None
    assert dataset is not None, "The downloaded dataset should not be None."

def test_download_hf_dataset_config(monkeypatch):

    # Mock Data
    config_data = {
        "repo_id": "expected/repo-name",
        "repo_file_a": "dataset-A.json", 
        "repo_file_b": "dataset-B.json", 
        "raw_dir": "data/raw/" 
    }

    monkeypatch.setattr("src.utils.config.load_config", lambda: config_data)

    # Track the calls

    hf_download_calls = []

    def stub_hf_hub_download(**kwargs):
        hf_download_calls.append(kwargs)
        return Path(kwargs.get("local_dir")) / kwargs.get("filename")

    monkeypatch.setattr("src.data.download_hf.hf_hub_download", stub_hf_hub_download)


    dataset = download_hf_dataset()

    # Verify hf_hub_download was called twice
    assert len(hf_download_calls) == 2

    expected_files = ["logs_gen9championsvgc2026regma.json", "logs_gen9championsvgc2026regmb.json"]

    # Verify config
    for i, call in enumerate(hf_download_calls):
        assert call.get("repo_id") == "cameronangliss/vgc-battle-logs"
        assert call.get("filename") == expected_files[i]
        assert call.get("local_dir") == "data/raw/"

    # Verify the final returned path
    for path_str in dataset:
        path = Path(path_str)
        assert path.suffix == ".json"
        assert path.parent == Path("data/raw/")

    

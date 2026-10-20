from pathlib import Path

from vgc_matchup_predictor.data.download_hf import download_hf_dataset


# Mock config and network access so this test is repeatable and offline.
def test_download_hf_dataset_uses_config_and_returns_paths(monkeypatch, tmp_path):
    config_data = {
        "dataset": {
            "hf_repo": "expected/repo-name",
            "m_a": "dataset-A.json",
            "m_b": "dataset-B.json",
            "raw_path": str(tmp_path / "raw"),
        }
    }
    monkeypatch.setattr(
        "vgc_matchup_predictor.data.download_hf.load_config",
        lambda config_file: config_data,
    )

    download_calls = []

    def stub_hf_hub_download(**kwargs):
        download_calls.append(kwargs)
        return Path(kwargs["local_dir"]) / kwargs["filename"]

    monkeypatch.setattr(
        "vgc_matchup_predictor.data.download_hf.hf_hub_download",
        stub_hf_hub_download,
    )

    dataset = download_hf_dataset()

    expected_files = ["dataset-A.json", "dataset-B.json"]
    assert [call["filename"] for call in download_calls] == expected_files
    assert all(call["repo_id"] == "expected/repo-name" for call in download_calls)
    assert all(call["repo_type"] == "dataset" for call in download_calls)
    assert all(call["local_dir"] == str(tmp_path / "raw") for call in download_calls)
    assert dataset == [tmp_path / "raw" / filename for filename in expected_files]

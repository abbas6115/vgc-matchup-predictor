import json
from pathlib import Path
from unittest.mock import mock_open, patch

import vgc_matchup_predictor.data.process_json as process_json
from vgc_matchup_predictor.data.process_json import (
    create_target_file,
    process_data,
    read_json,
    start_process_data,
    write_to_json,
)

# Use pathlib inputs and current snake_case helpers to match the implementation.

def test_create_target_file_with_nested_input_path():
    assert create_target_file(Path("data/raw/file1.json"), Path("data/processed")) == (
        Path("data/processed/file1.json")
    )


def test_create_target_file_with_filename_only():
    assert create_target_file(Path("file1.json"), Path("output")) == Path(
        "output/file1.json"
    )


def test_read_json_success():
    sample_data = {"key": "value"}

    with patch.object(Path, "open", mock_open(read_data=json.dumps(sample_data))):
        assert read_json(Path("dummy_path.json")) == sample_data


def test_read_json_file_not_found(capsys):
    with patch.object(Path, "open", side_effect=FileNotFoundError):
        assert read_json(Path("nonexistent.json")) is None

    assert "Error: Could not find file" in capsys.readouterr().out


def test_read_json_io_error(capsys):
    with patch.object(Path, "open", side_effect=IOError("Disk read error")):
        assert read_json(Path("faulty.json")) is None

    assert "I/O error: Disk read error" in capsys.readouterr().out


def test_write_to_json_success():
    mocked_open = mock_open()
    with patch.object(Path, "open", mocked_open):
        assert write_to_json({"battle_id": "123"}, Path("output.json")) is True

    mocked_open.assert_called_once_with("w", encoding="utf-8")
    written_content = "".join(
        call.args[0] for call in mocked_open().write.call_args_list
    )
    assert '"battle_id": "123"' in written_content


def test_write_to_json_file_not_found(capsys):
    with patch.object(Path, "open", side_effect=FileNotFoundError):
        write_to_json({"key": "value"}, Path("invalid_dir/output.json"))

    assert "Error: Cannot find file" in capsys.readouterr().out


def test_write_to_json_io_error(capsys):
    with patch.object(
        Path, "open", side_effect=IOError("Permission denied")
    ):
        write_to_json({"key": "value"}, Path("readonly.json"))

    assert "I/O error: Permission denied" in capsys.readouterr().out


@patch("vgc_matchup_predictor.data.process_json.write_to_json")
@patch("vgc_matchup_predictor.data.process_json.get_winner")
@patch("vgc_matchup_predictor.data.process_json.get_back_pokemon")
@patch("vgc_matchup_predictor.data.process_json.parse_log")
@patch("vgc_matchup_predictor.data.process_json.read_json")
def test_process_data_success(
    mock_read, mock_parse, mock_back_pokemon, mock_winner, mock_write, tmp_path
):
    # Mock each processing step so the test only checks orchestration and output.
    mock_read.return_value = {"battle_101": ["metadata", "raw_log_content"]}
    mock_parse.return_value = {
        "p1a": {
            "name": "Pikachu",
            "item": "Light Ball",
            "ability": "Static",
            "moves": ["Thunderbolt", "Quick Attack"],
            "nature": "Timid",
        }
    }
    mock_back_pokemon.return_value = [
        ["Venusaur", "Blastoise"],
        ["Tyranitar", "Salamence"],
    ]
    mock_winner.return_value = "p1"
    input_file = tmp_path / "raw" / "file1.json"
    target_dir = tmp_path / "processed"

    process_data([input_file], target_dir)

    mock_read.assert_called_once_with(input_file)
    mock_parse.assert_called_once_with("raw_log_content")
    mock_back_pokemon.assert_called_once_with("raw_log_content", ["Pikachu"])
    mock_winner.assert_called_once_with("raw_log_content")
    mock_write.assert_called_once_with(
        [
            {
                "battle_id": "battle_101",
                "p1a": "Pikachu",
                "p1a item": "Light Ball",
                "p1a ability": "Static",
                "p1a moves": ["Thunderbolt", "Quick Attack"],
                "p1a nature": "Timid",
                "p1_back": ["Venusaur", "Blastoise"],
                "p2_back": ["Tyranitar", "Salamence"],
                "winner": "p1",
            }
        ],
        target_dir / "file1.json",
    )


@patch("vgc_matchup_predictor.data.process_json.write_to_json")
@patch("vgc_matchup_predictor.data.process_json.read_json")
def test_process_data_skips_missing_file(mock_read, mock_write, tmp_path):
    mock_read.return_value = None

    process_data([tmp_path / "missing_file.json"], tmp_path / "processed")

    mock_write.assert_not_called()


@patch("vgc_matchup_predictor.data.process_json.process_data")
@patch("vgc_matchup_predictor.data.process_json.load_config")
def test_start_process_data_uses_config(mock_load_config, mock_process_data):
    # Derive expected absolute paths from the package location used by production code.
    mock_load_config.return_value = {
        "dataset": {
            "raw_path": "data/raw/",
            "m_a": "matchup_a.json",
            "m_b": "matchup_b.json",
            "processed_path": "data/processed/",
        }
    }

    start_process_data()

    project_root = Path(process_json.__file__).resolve().parents[3]
    mock_process_data.assert_called_once_with(
        [
            project_root / "data/raw/matchup_a.json",
            project_root / "data/raw/matchup_b.json",
        ],
        project_root / "data/processed/",
    )

import json
import pytest
from unittest.mock import patch, mock_open

from src.data.process_json import (
    start_process_data,
    process_data,
    read_json,
    createTargetFile,
    write_to_json,
)

# Tests for CreateTargetFile

def test_create_target_file_unix_path():
    file_path = "data/raw/file1.json"
    target_path = "data/processed/"
    expected = "data/processed/file1.json"

    assert createTargetFile(file_path, target_path) == expected

def test_create_target_file_filename_only():
    file_path = "file1.json"
    target_path = "output/"
    expected = "output/file1.json"

    assert createTargetFile(file_path, target_path) == expected

# Tests for read_json

# Checks if read_json can succeed
def test_read_json_success():
    sample_data = {"key": "value"}
    json_str = json.dumps(sample_data)

    with patch("builtins.open", mock_open(read_data=json_str)):
        result = read_json("dummy_path.json")
        assert result == sample_data

# Checks if a file not found error is printed
def test_read_json_file_not_found(capsys):
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = read_json("non_existen.json")
        assert result is None

        captured = capsys.readouterr()
        assert "Error: Cannot find file" in captured.out

# Checks if an error is printed for I/O
def test_read_json_io_error(capsys):
    with patch("builtins.open", side_effect=IOError("Disk read error")):
        result = read_json("faulty.json")
        assert result is None

        captured = capsys.readouterr()
        assert "I/O error: Disk read error" in captured.out


# Tests for write_to_json

def test_write_to_json_success():
    data = {"battle_id": "123"}
    m_open = mock_open()

    with patch("builtins.open", m_open):
        write_to_json(data, "output.json")

        m_open.assert_called_once_with("output.json", "w")
        handle = m_open()

        # Verify write operations occured
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert '"battle_id": "123"' in written_content

def test_write_to_json_file_not_found(capsys):
    data = {"key": "value"}
    with patch("builtins.open", side_effect=FileNotFoundError):
        write_to_json(data, "invalid_dir/output.json")

        capatured = capsys.readouterr()
        assert "Error: Cannot find file" in capatured.out

def test_write_to_json_io_error(capsys):
    data = {"key": "value"}
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        write_to_json(data, "readonly.json")

        captured = capsys.readouterr()
        assert "I/O error: Permission denied" in captured.out


# Test for process_data

@patch("src.data.process_json.write_to_json")
@patch("src.data.process_json.getWinner")
@patch("src.data.process_json.parse_log")
@patch("src.data.process_json.read_json")

def test_process_data_success(mock_read, mock_parse, mock_winner, mock_write):

    #setup mock
    mock_read.return_value = {
        "battle_101": ["metadata", "raw_log_content"]
    }

    mock_parse.return_value = {
        "p1a": {
            "name": "Pikachu",
            "item": "Light Ball",
            "ability": "Static",
            "moves": ["Thunderbolt", "Quick Attack"],
            "nature": "Timid",
        }
    }

    mock_winner.return_value = "p1"

    files = ["raw/file1.json"]
    target_dir = "processed/"

    process_data(files, target_dir)

    # Assert
    mock_read.assert_called_once_with("raw/file1.json")
    mock_parse.assert_called_once_with("raw_log_content")
    mock_winner.assert_called_once_with("raw_log_content")

    excepted_output = [
        {
        "battle_id": "battle_101",
        "p1a": "Pikachu",
        "p1a item": "Light Ball",
        "p1a ability": "Static",
        "p1a moves": ["Thunderbolt", "Quick Attack"],
        "p1a nature": "Timid",
        "winner": "p1"
        }
    ]
    mock_write.assert_called_once_with(excepted_output, "processed/file1.json")


@patch("src.data.process_json.write_to_json")
@patch("src.data.process_json.read_json")

def test_process_data_missing_file_skips(mock_read, mock_write, capsys):

    # Simulate missing file returning None
    mock_read.return_value = None

    files = ["missing_file.json"]
    process_data(files, "target/")

    # Check if warning is printed and write was skipped
    captured = capsys.readouterr()
    assert "Could not find file: missing_file.json" in captured.out
    mock_write.assert_not_called()


    # Tests for start_process_data

    @patch("src.data.process_json.process_data")
    @patch("src.data.process_json.load_config")

    def test_start_process_data(mock_load_config, mock_process_data):

        # Setup Mock
        mock_load_config.return_value = {
            "dataset": {
                "raw_path": "data/raw/",
                "m_a": "matchup_a.json",
                "m_b": "matchup_b.json",
                "processed_path": "data/processed/",

            }
        }

        start_process_data()

        expected_files = [
            "data/raw/matchup_a.json",
            "data/raw/matchup_b.json",
        ]
        expected_target = "data/processed/"

        mock_process_data.assert_called_once_with(expected_files, expected_target)
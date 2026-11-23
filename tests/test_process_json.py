import json
from pathlib import Path
from unittest.mock import mock_open, patch

import vgc_matchup_predictor.data.process_json as process_json
from vgc_matchup_predictor.data.process_json import (
    calc_attack_ratio,
    calc_type_matchup,
    create_target_file,
    lookup,
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
        },
        "p1b": {
            "name": "Pikachu",
            "item": "Light Ball",
            "ability": "Static",
            "moves": ["Thunderbolt", "Quick Attack"],
            "nature": "Timid",
        },
        "p2a": {
            "name": "Pikachu",
            "item": "Light Ball",
            "ability": "Static",
            "moves": ["Thunderbolt", "Quick Attack"],
            "nature": "Timid",
        },
        "p2b": {
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

    mock_stats = {
        "pikachu": {"type_1": "Electric", "type_2": "", "speed": 90, "attack": 55, "special_attack": 50, "defense": 40, "special_defense": 50},
    }

    mock_chart = {
        "Electric": {"Electric": 0.5}
    }
    input_file = tmp_path / "raw" / "file1.json"
    target_dir = tmp_path / "processed"

    with(
        patch.object(process_json, "lookup_object", mock_stats),
        patch.object(process_json, "type_chart_object", mock_chart)
    ):
        process_data([input_file], target_dir)

    mock_write.assert_called_once()
    written_records = mock_write.call_args[0][0]
    assert written_records[0]["battle_id"] == "battle_101"
    assert written_records[0]["p1a_vs_p2a"] == -1
    assert written_records[0]["p1a_vs_p2a_ratio"] == 1.375
    assert written_records[0]["winner"] == "p1"



@patch("vgc_matchup_predictor.data.process_json.write_to_json")
@patch("vgc_matchup_predictor.data.process_json.read_json")
def test_process_data_skips_missing_file(mock_read, mock_write, tmp_path):
    mock_read.return_value = None

    process_data([tmp_path / "missing_file.json"], tmp_path / "processed")

    mock_write.assert_not_called()


@patch("vgc_matchup_predictor.data.process_json.read_json")
@patch("vgc_matchup_predictor.data.process_json.process_data")
@patch("vgc_matchup_predictor.data.process_json.load_config")
def test_start_process_data_uses_config(mock_load_config, mock_process_data, mock_read_json):
    # Derive expected absolute paths from the package location used by production code.
    mock_load_config.return_value = {
        "dataset": {
            "raw_path": "data/raw/",
            "lookup_path": "data/lookup/",
            "stats": "pokemon_stats.json",
            "type_chart": "type_chart.json",
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

def test_lookup_success():
    mock_stats = {
        "pikachu": {"type_1": "Electric", "type_2": "", "speed": 90},
    }
    with patch.object(process_json, "lookup_object", mock_stats):
        assert lookup("Pikachu", "type_1") == "Electric"
        assert lookup("Pikachu", "speed") == 90


def test_lookup_special_pokemon_normalizes_form():
    mock_stats = {
        "vivillon": {"type_1": "Bug", "type_2": "Flying"},
        "alcremie": {"type_1": "Fairy", "type_2": ""},
    }
    with patch.object(process_json, "lookup_object", mock_stats):
        assert lookup("Vivillon", "type_1") == "Bug"
        assert lookup("Alcremie", "type_1") == "Fairy"


def test_lookup_missing_object(capsys):
    with patch.object(process_json, "lookup_object", {}):
        assert lookup("Pikachu", "speed") is None

    assert "Could not find look up object" in capsys.readouterr().out


def test_calc_type_matchup_dual_type_advantage():
    mock_stats = {
        "garchomp": {"type_1": "Dragon", "type_2": "Ground"},
        "heatran": {"type_1": "Fire", "type_2": "Steel"},
    }

    mock_chart = {
        "Dragon": {"Fire": 1.0, "Steel": 0.5},
        "Ground": {"Fire": 2.0, "Steel": 2.0},
    }
    with(
        patch.object(process_json, "lookup_object", mock_stats),
        patch.object(process_json, "type_chart_object", mock_chart)
    ):
        assert calc_type_matchup("Garchomp", "Heatran") == 2.0



def test_calc_type_matchup_immunity():
    mock_stats = {
        "garchomp": {"type_1": "Dragon", "type_2": "Ground"},
        "togekiss": {"type_1": "Fairy", "type_2": "Flying"},
    }

    mock_chart = {
        "Dragon": {"Fairy": 0.0, "Flying": 1.0},
        "Ground": {"Fairy": 1.0, "Flying": 0.0},
    }
    with(
        patch.object(process_json, "lookup_object", mock_stats),
        patch.object(process_json, "type_chart_object", mock_chart)
    ):
        assert calc_type_matchup("Garchomp", "Togekiss") == -4.0


def test_calc_type_matchup_missing_types(capsys):
    with patch.object(process_json, "lookup_object", {}):
        assert calc_type_matchup("Unknown1", "Unknown2") == 1.0

    assert "could not find types" in capsys.readouterr().out


def test_calc_attack_ratio():
    mock_stats = {
        "tyranitar": {"attack": 134, "special_attack": 95},
        "blissey": {"defense": 10, "special_defense": 135},
    }
    with patch.object(process_json, "lookup_object", mock_stats):
        assert calc_attack_ratio("Tyranitar", "Blissey") == 13.4


def test_calc_attack_ratio_using_special_attack():
    mock_stats = {
        "fluttermane": {"attack": 55, "special_attack": 135},
        "incineroar": {"defense": 90, "special_defense": 90},
        "mew": {"attack": 100, "special_attack": 100},
    }
    with patch.object(process_json, "lookup_object", mock_stats):
        assert calc_attack_ratio("FlutterMane", "Incineroar") == 1.5
        assert calc_attack_ratio("Mew", "Incineroar") == 1.1111


def test_calc_attack_ratio_missing_stats(capsys):
    with patch.object(process_json, "lookup_object", {}):
        assert calc_attack_ratio("Unknown1", "Unknown2") == 1.0

    assert "could not get attacking stats" in capsys.readouterr().out
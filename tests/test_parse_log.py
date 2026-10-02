import pytest
from unittest.mock import patch

from src.data.parse_log import (
    parse_log,
    extractStartingInformation,
    getLeadPokemon,
    getPokemonInfo,
    findSubstring,
)


# Sample log 
SAMPLE_LOG = (
    "|poke|p1|Pikachu, M|\n"
    "|showteam|p1|Pikachu||Light Ball|Static|Thunderbolt,Quick Attack|Timid|\n"
    "|switch|p1a: Pikachu|Pikachu, M|100/100\n"
    "|switch|p1b: Charizard|Charizard, M|100/100\n"
    "|switch|p2a: Inteleon|Inteleon, M|100/100\n"
    "|switch|p2b: Grimmsnarl|Grimmsnarl, M|100/100\n"\
    "Pikachu||Light Ball|Static|Thunderbolt,Quick Attack|Timid\n"
    "Charizard||Life Orb|Solar Power|Heat Wave,Air Slash|Modest|\n"
    "Inteleon||Choice Specs|Torrent|Snipe Shot,Ice Beam|Timid|\n"
    "Grimmsnarl||Light Clay|Prankster|Reflect,Light Screen|Careful|\n"
    "|turn|1\n"
    "|move|p1a: Pikachu|Thunderbolt|p2a: Inteleon\n"
)

# Tests for findSubstring

def test_find_substring_match():
    log = "|switch|p1a: Pikachu|Pikachu, M|100/100"
    regex = r"p1a:[^|]*\|([^,]+)"

    result = findSubstring(regex, log)
    assert result == "Pikachu"

def test_find_substring_no_match(capsys):
    log = "|switch|p1a: Pikachu|Pikachu, M|100/100"
    regex = r"nonexistent_pattern_([^,]+)"

    result = findSubstring(regex, log)
    assert result is None

    captured = capsys.readouterr()
    assert "Error Searching for target" in captured.out


# Tests for extractStartingInformation

def test_extract_starting_information():
    result = extractStartingInformation(SAMPLE_LOG)

    assert result.startswith("|showteam|")
    assert "|turn|1" not in result
    assert "Pikachu" in result


# Tests for getLeadPokemon

def test_get_lead_pokemon():
    short_log = (
        "|switch|p1a: Pikachu|Pikachu, M|100/100\n"
        "|switch|p1b: Charizard|Charizard, M|100/100\n"
        "|switch|p2a: Inteleon|Inteleon, M|100/100\n"
        "|switch|p2b: Grimmsnarl|Grimmsnarl, M|100/100\n"
    )

    result = getLeadPokemon(short_log)

    expected = [
        "p1a: Pikachu",
        "p1b: Charizard",
        "p2a: Inteleon",
        "p2b: Grimmsnarl",
    ]

    assert result == expected


# Tests for getPokemonInfo

def test_get_pokemon_info():
    pokemon_list = [
        "p1a: Pikachu",
        "p2a: Inteleon",
    ]

    log_data = (
        "Pikachu||Light Ball|Static|Thunderbolt,Quick Attack|Timid|\n"
        "Inteleon||Choice Specs|Torrent|Snipe Shot,Ice Beam|Timid|\n"
    )

    result = getPokemonInfo(pokemon_list, log_data)

    expected = {
        "p1a": {
            "name": "Pikachu",
            "item": "Light Ball",
            "ability": "Static",
            "moves": ["Thunderbolt", "Quick Attack"],
            "nature": "Timid",
        },

        "p2a": {
            "name": "Inteleon",
            "item": "Choice Specs",
            "ability": "Torrent",
            "moves": ["Snipe Shot", "Ice Beam"],
            "nature": "Timid",
        },
    }

    assert result == expected


# Test for parse_log 

def test_parse_log_integration():
    result = parse_log(SAMPLE_LOG)

    assert "p1a" in result
    assert "p1b" in result
    assert "p2a" in result
    assert "p2b" in result

    # Check p1 details
    assert result["p1a"]["name"] == "Pikachu"
    assert result["p1a"]["item"] == "Light Ball"
    assert result["p1a"]["ability"] == "Static"
    assert result["p1a"]["moves"] == ["Thunderbolt", "Quick Attack"]
    assert result["p1a"]["nature"] == "Timid"





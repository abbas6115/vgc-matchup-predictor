"""
module to parse the battle log from data source
"""

import re


def parse_log(log:str) -> dict:
    """
    takes the battle log and parses it to get the lead pokemon information 
    returns a dictionary of dictionaries containing the pokemon information
    in style of "position" : {name: , item: , ability: , moves: , nature: }
    
    """


    short_log = extract_starting_information(log)
    lead_pokemon = get_lead_pokemon(short_log)
    battle_lead_info = get_pokemon_info(lead_pokemon, short_log)
    
    return battle_lead_info

def extract_starting_information(log:str) -> str:
    """
    takes the log and strips the lines between "|showteam|" (inclusive) and "|turn|1" exclusive
    to shorten it
    """
    return log[log.find('|showteam|'):log.find('|turn|1')]


def get_lead_pokemon(log:str) -> list:
    """
    takes the log and returns a list of the pokemon each player leads
    """
    
    return [
        "p1a: " + find_substring(r"p1a:[^|]*\|([^,]+)", log),
        "p1b: " + find_substring(r"p1b:[^|]*\|([^,]+)", log),
        "p2a: " + find_substring(r"p2a:[^|]*\|([^,]+)", log),
        "p2b: " + find_substring(r"p2b:[^|]*\|([^,]+)", log),
        ]

# takes a list of pokemon and returns a list of dictionaries containing the pokemon information
# style of {player: , name: , item: , ability: , moves: , nature: }
def get_pokemon_info(pokemon_list:list, log:str) -> dict:

    """
    takes a list of pokemon and the log 
    returns a dictionary of dictionaries containing the pokemon information
    in style of "position" : {name: , item: , ability: , moves: , nature: }
    """

    pokemon_info_list = {}
    for pokemon in pokemon_list:
        pokemon_info = {}
        line = find_substring(r"({}\|\|[^\]]*)".format(pokemon[5:]), log)

        pokemon_info["name"] = pokemon[5:]

        # remove pokemon name from line
        line = line[(len(pokemon_info["name"]+"||")):]

        pokemon_info["item"] = line[:line.find('|')]
        line = line[line.find('|')+1:]

        
        pokemon_info["ability"] = line[:line.find('|')]
        line = line[line.find('|')+1:]


        pokemon_info["moves"] = line[:line.find('|')].split(",")
        line = line[line.find('|')+1:]

        pokemon_info["nature"] = line[:line.find('|')]

        line = line[line.find('|')+1:]
        
        pokemon_info_list[pokemon[:3]] = pokemon_info
    
    return pokemon_info_list

def get_winner(log:str) -> str:
    """
    Takes the log and returns the winner of the battle
    either p1 or p2
    """
    winner = find_substring(r"\|win\|((?:(?!\\n).)*)", log)
    if not winner:
        return ""

    player = find_substring(fr"\|player\|([^|]+)\|{re.escape(winner)}\|", log)
    return player

def find_substring(regex, log) -> str: 
    """
    helper function to search log with regex
    takes a regex and log
    returns a string of characters that match, returns empty if characters dont match
    """

    pattern = re.compile(regex)
    search = pattern.search(log)
    if search:
        return search.group(1) 
    else:
        print(f"Error Searching for target within log with pattern {regex}")
        return ""
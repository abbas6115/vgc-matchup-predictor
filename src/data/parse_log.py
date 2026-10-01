"""
module to parse the battle log from json
takes 
"""

import re


def parse_log(log:str):
    """
    takes the battle log and parses it to get the lead pokemon information 
    returns a dictionary of pokemon
    """

    short_log = extractStartingInformation(log)
    lead_pokemon = getLeadPokemon(short_log)
    pokemon_lead_info = getPokemonInfo(lead_pokemon, short_log)
    return pokemon_lead_info

def extractStartingInformation(log:str) -> str:
    """
    takes the log and strips the lines between "|showteam|" (inclusive) and "|turn|1" exclusive
    to shorten it
    """
    return log[log.find('|showteam|'):log.find('|turn|1')]


def getLeadPokemon(log:str) -> list:
    """
    takes the log and returns a list of the pokemon each player leads
    """
    
    return [
        findSubstring(r"(p1a:[^|]*)", log),
        findSubstring(r"(p1b:[^|]*)", log),
        findSubstring(r"(p2a:[^|]*)", log),
        findSubstring(r"(p2b:[^|]*)", log),
        ]

# takes a list of pokemon and returns a list of dictionaries containing the pokemon information
# style of {player: , name: , item: , ability: , moves: , nature: }
def getPokemonInfo(pokemon_list:list, log:str) -> dict:

    """
    takes a list of pokemon and the log 
    returns a list of dictionaries containing the pokemon information
    in style of {player: , name: , item: , ability: , moves: , nature: }
    """

    pokemon_info_list = []
    for pokemon in pokemon_list:
        pokemon_info = {}
        line = findSubstring(r"({}\|\|[^\]]*)".format(pokemon[5:]), log)
        
        pokemon_info["player"] = pokemon[:2]

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
        
        pokemon_info_list.append(pokemon_info)
    
    return pokemon_info_list

        

def findSubstring(regex, log) -> str: 
    """
    helper function to search log with regex
    takes a regex and log
    returns a string of characters that match, returns empty if characters dont match
    """

    pattern = re.compile(regex)
    search = pattern.search(log)
    result = search.group(1) 
    if search:
        return result
    else:
        return ""


    print(parse_log())
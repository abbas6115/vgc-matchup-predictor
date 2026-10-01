import json
from parse_log import *
import re

def process_data(files:list[str]) -> None:
    for file in files:
        data = read_json(file)

        for key in data:
                battle_info = {}
                
                battle_info['battle_id'] = key
                
                battle_log = data[key][1]
                parsed_battle_log = parse_log(battle_log)
            
                for position in parsed_battle_log:
                    battle_info[position] = parsed_battle_log[position]["name"]
                    battle_info[position+" item"] = parsed_battle_log[position]["item"]
                    battle_info[position+" ability"] = parsed_battle_log[position]["ability"]
                    battle_info[position+" moves"] = parsed_battle_log[position]["moves"]
                    battle_info[position+" nature"] = parsed_battle_log[position]["nature"]
                
                



def read_json(file:str) -> dict:
    """
    Helper function to read json files 
    takes a file as a string
    returns a dict
    """
    try:
        with open(file, 'r') as f:
            data = json.load(f)

    except FileNotFoundError:
        print("Error: Cannot find file")

    except IOError as e:
        print(f"I/O error: {e}")

    return data





process_data(["data\\raw\\logs_gen9championsvgc2026regma.json"])
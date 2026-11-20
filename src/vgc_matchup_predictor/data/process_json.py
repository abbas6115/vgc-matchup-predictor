import json
from math import log
from pathlib import Path
from vgc_matchup_predictor.data.parse_log import parse_log, get_winner,get_back_pokemon
from vgc_matchup_predictor.utils.config import load_config

# object for lookup pokemon stat or type
lookup_object = {}

# object that reads the type matchup .json to calc type advantage
type_chart_object = {}

def start_process_data()-> None:
    """
    Function just to declare file path and target paths and call process data
    """
    global lookup_object
    global type_chart_object

    config = load_config('data_config.yaml')
    project_root = Path(__file__).resolve().parents[3]
    RAW_FILE_PATH = project_root / config['dataset']['raw_path']

    lookup_file= project_root / config["dataset"]["lookup_path"] / config["dataset"]["stats"]
    lookup_object = read_json(lookup_file)
    
    type_chart_file = project_root / config["dataset"]["lookup_path"] / config["dataset"]["type_chart"]
    type_chart_object = read_json(type_chart_file)
    

    raw_files = [
            RAW_FILE_PATH / config['dataset']['m_a'],
            RAW_FILE_PATH / config['dataset']['m_b'],
        ]

    target_path = project_root / config['dataset']['processed_path']
    print("Processing files")    
    process_data(raw_files,target_path)
    print("Completed Processing files")


def process_data(files:list[str], target:str) -> None:
    """
    takes a list of json files to process and creates a processed path
    """

    # ensure target directories exist
    target.mkdir(parents=True, exist_ok=True)

    # iterate through files
    for file in files:
        data = read_json(file)

        # skip to next file if it cant be found
        if not data:
            continue

        # create target file path to add  data to
        target_file = create_target_file(file,target)

        battle_info_list = []

        # Search through each battle and process the data
        for key in data:
                battle_info = {}
                
                battle_info['battle_id'] = key
                
                battle_log = data[key][1]
                parsed_battle_log = parse_log(battle_log)
                
                lead_pokemon= []

                # add each lead pokemon information
                for position in parsed_battle_log:
                    lead_pokemon.append(parsed_battle_log[position]["name"])
                    battle_info[position] = parsed_battle_log[position]["name"]
                    battle_info[position+" speed"] = lookup(parsed_battle_log[position]["name"],"speed")
                    battle_info[position+" type1"] = lookup(parsed_battle_log[position]["name"],"type_1")
                    battle_info[position+" type2"] = lookup(parsed_battle_log[position]["name"],"type_2")
                    battle_info[position+" item"] = parsed_battle_log[position]["item"]
                    battle_info[position+" ability"] = parsed_battle_log[position]["ability"]
                    battle_info[position+" moves"] = parsed_battle_log[position]["moves"]
                    battle_info[position+" nature"] = parsed_battle_log[position]["nature"]

                # add individual type matchups
                battle_info["p1a_vs_p2a"] = calc_type_matchup(battle_info["p1a"],battle_info["p2a"])
                battle_info["p1a_vs_p2b"] = calc_type_matchup(battle_info["p1a"],battle_info["p2b"])
                battle_info["p1b_vs_p2a"] = calc_type_matchup(battle_info["p1b"],battle_info["p2a"])
                battle_info["p1b_vs_p2b"] = calc_type_matchup(battle_info["p1b"],battle_info["p2b"])

                # add back pokemon
                back_pokemon = get_back_pokemon(battle_log,lead_pokemon)
                battle_info["p1_back"] = back_pokemon[0]
                battle_info["p2_back"] = back_pokemon[1]

                # add winner of the battle
                battle_info["winner"] = get_winner(battle_log)
                battle_info_list.append(battle_info)
        
        if not write_to_json(battle_info_list,target_file):
            print(f"Could not write to target path {target}")
            

def read_json(file:Path) -> dict:
    """
    Helper function to read json files 
    takes a file as a string
    returns the json object as a dict
    """
    try:
        print(f'\nOpening: "{file}" to read')
        with file.open("r", encoding="utf-8") as f:
            print(f'Reading: "{file}"')
            data = json.load(f)

        print(f'Successfully read "{file}"')
        return data

    except FileNotFoundError:
        print(f'Error: Could not find file: "{file}"')

    except IOError as e:
        print(f"I/O error: {e}")

    return None

def create_target_file(file, path):
    """
    takes the file and the target path and create a new file in path to write to
    """
    return path / file.name

def lookup(pokemon: str, attribute: str) -> str:
    """
    Looks up the pokemon within the pokemon_stats.json and gets the attribute specified.
    """
    specialPokemon = ["Vivillon", "Alcremie", "Florges", "Furfrou","Sinistcha", "Polteageist"]

    for p in specialPokemon:
        if p in pokemon:
            pokemon = p
            break

    pokemon_object = lookup_object.get(pokemon.lower())
    if pokemon_object is None:
        print(f"Could not find Pokémon: {pokemon}")
        return ""

    if attribute not in pokemon_object:
        print(f"Could not find attribute '{attribute}' for {pokemon}")
        return ""

    return pokemon_object[attribute]

config = load_config('data_config.yaml')
project_root = Path(__file__).resolve().parents[3]
RAW_FILE_PATH = project_root / config['dataset']['raw_path']

lookup_file= project_root / config["dataset"]["lookup_path"] / config["dataset"]["stats"]
lookup_object = read_json(lookup_file)

type_chart_file = project_root / config["dataset"]["lookup_path"] / config["dataset"]["type_chart"]
type_chart_object = read_json(type_chart_file)
    
def calc_type_matchup(pokemon1:str, pokemon2:str, attacking:bool=True):
    """
    Calculates the type advatange of both attaking types and picks the higher one as the score.
    it takes the attacking type against both defending types then multiplies its matchup to get the resultant
    """
    attacking_mon = ""
    if(attacking):
        attacking_mon = pokemon1
        defending_mon = pokemon2
    else:
        attacking_mon = pokemon2
        defending_mon = pokemon1


    atk_type1 = lookup(attacking_mon,"type_1")
    atk_type2 = lookup(attacking_mon,"type_2")
    def_type1 = lookup(defending_mon,"type_1")
    def_type2 = lookup(defending_mon,"type_2")

    if not atk_type1 or not def_type1:
        print("could not find types")
        return 1.0

    atk_type1_val = 0.0
    atk_type2_val = 0.0
    score = 0

    atk_type1_val = type_chart_object[atk_type1][def_type1]

    if def_type2:
        atk_type1_val *= type_chart_object[atk_type1][def_type2]

    # return if attacker only has one type
    if atk_type2:
        atk_type2_val = type_chart_object[atk_type2][def_type1]
        
        if def_type2:
            atk_type2_val *= type_chart_object[atk_type2][def_type2]

    final_matchup = max(atk_type1_val, atk_type2_val)

    # prevent log(0)
    if (final_matchup == 0):
        score = -4.0
    else: score = log(final_matchup,2)
    
    score = int(score)
    return score if attacking else -1 * score


def write_to_json(data:dict,target: str) -> bool: 
    """
    Function to the proceesed file to a json
    return a bool if it fails
    """

    try:
        print(f'\nOpening: "{target}" to write')
        with target.open("w", encoding="utf-8") as file:
            print(f'Writing: "{target}"')
            json.dump(data,file,indent=2)

            print(f'Successfully wrote to: "{target}"')
        return True
    
    except FileNotFoundError:
        print("Error: Cannot find file path to write to")


    except IOError as e:
        print(f"I/O error: {e}")



if __name__ == '__main__':
    start_process_data()
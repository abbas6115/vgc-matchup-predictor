import json
from src.data.parse_log import parse_log
from src.utils.config import load_config

def start_process_data()-> None:
    """
    Function just to declare file path and target paths and call process data
    """
    config = load_config()

    RAW_FILE_PATH = config['dataset']['raw_path']

    raw_files = [
            RAW_FILE_PATH + config['dataset']['m_a'],
            RAW_FILE_PATH + config['dataset']['m_b'],
        ]

    target_path = config['dataset']['processed_path']

    process_data(raw_files,target_path)


def process_data(files:list[str], target:str) -> None:
    """
    takes a list of json files to process and creates a processed path
    """

    # iterate through files
    for file in files:
        data = read_json(file)

        # skip steps if file cant be found
        if not data:
            print("Could not find file: "+file)
            continue

        # create target file path to add  data to

        target_file = createTargetFile(file,target)

        battle_info_list = []

        # Search through each battle and process the data
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

                battle_info_list.append(battle_info)

        
        write_to_json(battle_info_list,target_file)

def read_json(file:str) -> dict:
    """
    Helper function to read json files 
    takes a file as a string
    returns the json object as a dict
    """
    try:
        with open(file, 'r') as f:
            data = json.load(f)

        return data

    except FileNotFoundError:
        print("Error: Cannot find file")

    except IOError as e:
        print(f"I/O error: {e}")

    return None

def createTargetFile(file, path):
    """
    takes the file and the target path and create a new file in path to write to
    """
    return path+file.split('/')[-1] 

def write_to_json(data:dict,target: str) -> None:
    
    try:
        with open(target, 'w') as file:
            json.dump(data,file,indent=2)
    
    except FileNotFoundError:
        print("Error: Cannot find file")

    except IOError as e:
        print(f"I/O error: {e}")

if __name__ == '__main__':
    start_process_data()

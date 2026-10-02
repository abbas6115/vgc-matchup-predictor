from pathlib import Path
import yaml

# Safely finds and loads the data_config.yaml file in configs/
def load_config(config_path: str = 'configs/data_config.yaml') -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found at {path.resolve()}')

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
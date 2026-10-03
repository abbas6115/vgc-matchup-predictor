from pathlib import Path
import yaml

# Safely finds and loads the data_config.yaml file in configs/
def load_config() -> dict:
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "configs" / "data_config.yaml"
    
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found at {path.resolve()}')

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
from pathlib import Path
import yaml

# Safely finds and loads the data_config.yaml file in configs/
def load_config(config_file) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "configs" / config_file
    
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found at {path.resolve()}')

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# loads and merges all YAML files in configs/ into one dictionary
def load_all_configs() -> dict:
    config_path = Path('configs')
    merged_config = {}

    for yaml_file in config_path.glob('*.yaml'):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content:
                merged_config.update(content)

    return merged_config
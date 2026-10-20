from pathlib import Path
import yaml

# Determine the directory relative to this config.py file
# Adjust parent count or anchor directly to project structure
def _get_configs_dir() -> Path:

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        configs_dir = parent / "configs"
        if configs_dir.is_dir():
            return configs_dir
            
    # Fallback: assume working directory's config folder
    return Path.cwd() / "configs"

# Safely finds and loads the data_config.yaml file in configs/
def load_config(config_file: str) -> dict:
    path = _get_configs_dir() / config_file
    
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found at {path.resolve()}')

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# loads and merges all YAML files in configs/ into one dictionary
def load_all_configs() -> dict:
    config_path = _get_configs_dir()
    merged_config = {}

    for yaml_file in config_path.glob('*.yaml'):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content:
                merged_config.update(content)

    return merged_config
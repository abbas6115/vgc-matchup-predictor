from pathlib import Path
from huggingface_hub import hf_hub_download
from vgc_matchup_predictor.utils.config import load_config

# Wrapper method that imports the JSON files for regulations M-A and M-B battles (Bo1) using
# Hugging Face's hf_hub_download() method, then returns the absolute file paths for both tables as a list
def download_hf_dataset():
    config = load_config()
    REPO_ID = config['dataset']['hf_repo']
    FILENAME_MA = config['dataset']['m_a']
    FILENAME_MB = config['dataset']['m_b']

    # Resolve data/raw relative to the project root
    project_root = Path(__file__).resolve().parents[3]
    local_dir = project_root / config["dataset"]["raw_path"]
    local_dir.mkdir(parents=True, exist_ok=True) # make sure directory exists

    file_path_logs_MA = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME_MA,
        local_dir=str(local_dir),
        repo_type='dataset'
    )
    file_path_logs_MB = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME_MB,
        local_dir=str(local_dir),
        repo_type='dataset'
    )
    return [ file_path_logs_MA, file_path_logs_MB ]

if __name__ == '__main__':
    download_hf_dataset()
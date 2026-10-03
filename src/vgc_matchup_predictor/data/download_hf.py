from huggingface_hub import hf_hub_download
from src.vgc_matchup_predictor.utils.config import load_config

# Wrapper method that imports the JSON files for regulations M-A and M-B battles (Bo1) using
# Hugging Face's hf_hub_download() method, then returns the absolute file paths for both tables as a list
def download_hf_dataset():
    config = load_config()
    REPO_ID = config['dataset']['hf_repo']
    FILENAME_MA = config['dataset']['m_a']
    FILENAME_MB = config['dataset']['m_b']
    LOCAL_DIR = config['dataset']['raw_path']

    file_path_logs_MA = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME_MA,
        local_dir=LOCAL_DIR,
        repo_type='dataset'
    )
    file_path_logs_MB = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME_MB,
        local_dir=LOCAL_DIR,
        repo_type='dataset'
    )
    return [ file_path_logs_MA, file_path_logs_MB ]

if __name__ == '__main__':
    download_hf_dataset()
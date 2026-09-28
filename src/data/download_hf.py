from huggingface_hub import hf_hub_download

FILENAME = 'cameronangliss/vgc-battle-logs'
LOCAL_DIR = '././data/raw/'
REPO_TYPE = 'dataset'

# Wrapper method that imports the JSON files for regulations M-A and M-B battles (Bo1) using
# Hugging Face's hf_hub_download() method, then returns the absolute file paths for both tables as a list
def download_hf_dataset():
    file_path_logs_MA = hf_hub_download(
        repo_id=FILENAME,
        filename='logs_gen9championsvgc2026regma.json',
        local_dir=LOCAL_DIR,
        repo_type=REPO_TYPE
    )
    file_path_logs_MB = hf_hub_download(
        repo_id=FILENAME,
        filename='logs_gen9championsvgc2026regmb.json',
        local_dir=LOCAL_DIR,
        repo_type=REPO_TYPE
    )
    return [ file_path_logs_MA, file_path_logs_MB ]
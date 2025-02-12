import os

from huggingface_hub import snapshot_download

from src.config import DEFAULT_SETTINGS, MODEL_PATHS

SAVE_DIR = DEFAULT_SETTINGS["model_save_dir"]


def download_model(model_name: str, model_save_path: str) -> None:
    """
    Downloads a given LLM model from Hugging Face and stores it in the specified directory.

    Args:
        model_name (str): The name of the model to download from Hugging Face.
        model_save_path (str): The directory where the model should be stored.
    """
    if not os.path.exists(model_save_path):
        print(f"Downloading {model_name} from Hugging Face...")
        snapshot_download(repo_id=model_name, local_dir=model_save_path)
        print(f"Model saved at: {model_save_path}")
    else:
        print(f"Model already exists at: {model_save_path}, skipping download.")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    download_model(DEFAULT_SETTINGS["llm"], MODEL_PATHS["llm"])
    download_model(DEFAULT_SETTINGS["judge_model"], MODEL_PATHS["judge"])


if __name__ == "__main__":
    main()

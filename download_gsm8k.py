import re

import pandas as pd
from datasets import load_dataset


def download_gsm8k(split: str, config: str = "main") -> pd.DataFrame:
    """Downloads the GSM8K dataset for the specified split.

    Args:
        split (str): The dataset split to download ('train' or 'test').
        config (str): The dataset configuration ('main' or 'socratic').

    Returns:
        pd.DataFrame: The dataset as a Pandas DataFrame.
    """
    dataset = load_dataset("openai/gsm8k", config, split=split)
    return dataset.to_pandas()


def preprocess_gsm8k(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the GSM8K dataset by extracting numeric answers and removing invalid rows.

    This function renames the 'answer' column to 'raw_answer', trims whitespace, and extracts
    the numerical answer from the solution string. It also removes rows where the extracted
    answer is missing.

    Args:
        df (pd.DataFrame): The raw GSM8K dataset containing questions and answers.

    Returns:
        pd.DataFrame: The preprocessed dataset with a cleaned 'answer' column.
    """

    def extract_number(text):
        """Extracts the final numeric answer from a solution string."""
        match = re.search(r"####\s*(\d+)", str(text))
        return int(match.group(1)) if match else None

    df = df.rename(columns={"answer": "raw_answer"})
    df["raw_answer"] = df["raw_answer"].str.strip()
    df["answer"] = df["raw_answer"].apply(extract_number)

    # remove rows with no correctly formatted answer
    df = df.dropna(subset=["answer"])

    return df


def save_to_parquet(df: pd.DataFrame, split: str) -> None:
    """Saves the preprocessed dataset to a Parquet file.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        split (str): The dataset split name ('train' or 'test').

    Returns:
        None: Saves the DataFrame to a file.
    """
    output_path = f"./data/gsm8k_{split}.parquet"

    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"{split} split saved to: {output_path}")


def main():
    for split in ["train", "test"]:
        print(f"Downloading {split} ...")
        df = download_gsm8k(split)
        df = preprocess_gsm8k(df)
        save_to_parquet(df, split)


if __name__ == "__main__":
    main()

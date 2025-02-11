from pathlib import Path
from typing import Dict

import pandas as pd


class DataHandler:
    """A class for handling dataset loading and validation.
    A class for handling dataset loading and validation.

    Attributes:
        dataset_name (str): The name of the dataset to be handled.
        dataset (Dict[str, pd.DataFrame]): A dictionary containing different splits of the dataset.

    Methods:
        load_dataset() -> Dict[str, pd.DataFrame]: Returns the loaded dataset.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

        if self.dataset_name == "gsm8k":
            self._check_gsm8k()
            self.dataset = self._load_gsm8k()

        else:
            raise NotImplementedError(f"Dataset {dataset_name} not found")

    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the loaded dataset. It has a dictionary containing different splits of the dataset.
        Each split has at least the columns "question" and "answer".

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing different splits of the dataset.
        """

        assert self.dataset is not None, "Dataset not loaded"

        return self.dataset

    def _load_gsm8k(self) -> Dict[str, pd.DataFrame]:
        """
        Loads the GSM8K dataset from Parquet files.

        Reads the train and test splits, along with sampled subsets for testing and validation.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing:
                - "train": Full training dataset
                - "test": Full test dataset
                - "test_100": A sample of 100 rows from the training dataset
                - "validation_32": A sample of 32 rows from the training dataset
        """
        train_split = pd.read_parquet("./data/gsm8k_train.parquet")
        test_split = pd.read_parquet("./data/gsm8k_test.parquet")

        test_100 = train_split.sample(n=100, random_state=42)
        validation_32 = train_split.sample(n=32, random_state=42)

        return {
            "train": train_split,
            "test": test_split,
            "test_100": test_100,
            "validation_32": validation_32,
        }

    def _check_gsm8k(self) -> None:
        """
        Checks if the required GSM8K dataset files exist.

        Raises:
            ValueError: If the required dataset files are missing.
        """
        for split in ["train", "test"]:
            path = Path(f"./data/gsm8k_{split}.parquet")
            if not path.exists():
                raise ValueError("Please download the GSM8K dataset first")

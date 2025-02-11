from pathlib import Path

import pandas as pd
import pytest

from src.DataHandler import DataHandler


@pytest.fixture
def mock_gsm8k_files(tmp_path):
    """
    Creates temporary GSM8K dataset files for testing.
    """
    data = {"question": ["Q1", "Q2", "Q3"], "answer": [42, 13, 7]}
    df = pd.DataFrame(data)

    # Create temporary directory and files
    train_path = tmp_path / "gsm8k_train.parquet"
    test_path = tmp_path / "gsm8k_test.parquet"

    df.to_parquet(train_path, engine="pyarrow")
    df.to_parquet(test_path, engine="pyarrow")

    return tmp_path


def test_data_handler_initialization(mock_gsm8k_files, monkeypatch):
    """
    Test if DataHandler initializes properly when the dataset files exist.
    """
    # Mock the data directory
    monkeypatch.setattr(Path, "exists", lambda _: True)

    handler = DataHandler("gsm8k")
    assert isinstance(handler, DataHandler)
    assert isinstance(handler.dataset, dict)


def test_load_dataset(mock_gsm8k_files, monkeypatch):
    """
    Test if load_dataset() returns a dictionary containing Pandas DataFrames.
    """
    monkeypatch.setattr(Path, "exists", lambda _: True)

    handler = DataHandler("gsm8k")
    dataset = handler.load_dataset()

    assert isinstance(dataset, dict)
    assert "train" in dataset
    assert "test" in dataset
    assert "test_100" in dataset
    assert "validation_32" in dataset
    assert isinstance(dataset["train"], pd.DataFrame)
    assert isinstance(dataset["test"], pd.DataFrame)


def test_check_gsm8k_missing_files(monkeypatch):
    """
    Test if _check_gsm8k() raises ValueError when files are missing.
    """
    monkeypatch.setattr(Path, "exists", lambda _: False)  # Simulate missing files

    with pytest.raises(ValueError, match="Please download the GSM8K dataset first"):
        DataHandler("gsm8k")


def test_invalid_dataset():
    """
    Test if NotImplementedError is raised for an unsupported dataset.
    """
    with pytest.raises(NotImplementedError, match="Dataset unknown_dataset not found"):
        DataHandler("unknown_dataset")

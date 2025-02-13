import argparse
import os
import time
from typing import List

# turn off logging to terminal for vLLM
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
import pandas as pd
import torch.distributed as dist
import vllm

from src import DataHandler
from src.config import DEFAULT_SETTINGS, MODEL_PATHS
from src.llm_inference import inference_vllm


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for running the script.

    Returns:
        argparse.Namespace: A namespace containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of seeds")
    parser.add_argument(
        "--temps", nargs="+", type=float, required=True, help="List of temperatures"
    )
    parser.add_argument(
        "--dataset-split", type=str, default="test_100", help="Dataset split to use"
    )

    args = parser.parse_args()

    assert len(args.seeds) == len(args.temps), "Number of seeds and temperatures must be the same"

    valid_splits = ["train", "test", "test_100", "validation_32"]
    if args.dataset_split not in valid_splits:
        raise ValueError(f"Dataset split must be one of {valid_splits}")

    return args


def load_dataset(split: str) -> pd.DataFrame:
    """
    Loads a dataset split from the DataHandler.

    Args:
        split (str): The dataset split to load (e.g., "train", "test", "test_100", "validation_32").

    Returns:
        pd.DataFrame: A DataFrame containing the dataset for the specified split.
    """
    data_handler = DataHandler("gsm8k")
    return data_handler.load_dataset()[split]


def save_results(
    results: List[List[str]], ground_truths: List[str], questions: List[str], split: str
) -> None:
    """Saves the model's inference results along with questions and ground truths as Parquet files.

    Args:
        results (List[List[str]]): A list of lists containing model-generated responses.
                                   Each inner list corresponds to a different inference run.
        ground_truths (List[str]): A list of ground-truth answers corresponding to the questions.
        questions (List[str]): A list of questions from the dataset.
        split (str): The dataset split (e.g., "train", "test", "test_100", "validation_32").
    """
    save_dir = f"./data/gsm8k_{split}/"
    os.makedirs(save_dir, exist_ok=True)

    for idx, run in enumerate(results):
        df = pd.DataFrame(
            {
                "question": questions,
                "model_response": run,
                "ground_truth": ground_truths,
            }
        )
        df.to_parquet(f"{save_dir}run_{idx}.parquet", engine="pyarrow", index=False)


def main():
    args = parse_arguments()
    seeds, temperatures, split = args.seeds, args.temps, args.dataset_split
    number_of_runs = len(seeds)

    df = load_dataset(split)
    questions = df["question"].tolist()
    ground_truths = df["answer"].tolist()

    llm = vllm.LLM(MODEL_PATHS["llm"], max_model_len=DEFAULT_SETTINGS["llm_max_length"])

    # run the model for N times
    runs = []
    for i in range(number_of_runs):
        seed = seeds[i]
        temp = temperatures[i]
        runs.append(inference_vllm(llm, questions, seed, temp, MODEL_PATHS["llm"]))

    save_results(runs, ground_truths, questions, split)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Script execution time: {elapsed_time:.1f} seconds")
    if dist.is_initialized():
        dist.destroy_process_group()

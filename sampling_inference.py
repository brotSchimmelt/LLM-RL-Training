import argparse
import os
import time
from typing import Any, Dict, List

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import pandas as pd
import vllm

from src import DataHandler, GSM8KAnswerChecker, GSM8KAnswerCheckerResult
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


def calculate_metrics(results: List[List[GSM8KAnswerCheckerResult]]) -> Dict[str, Any]:
    """Calculates performance metrics for multiple runs of model answers.

    Args:
        results (List[List[GSM8KAnswerCheckerResult]]): List of results from multiple runs,
            where each run contains a list of answer checker results.

    Returns:
        Dict[str, Any]: Dictionary containing the following metrics:
            - pass@k: Fraction of questions answered correctly in at least one run
            - pass@1: Fraction of questions answered correctly in the first run
            - majority@k: Fraction of questions where majority of runs were correct
            - avg_len_correct: Average length of correct answers
            - avg_len_incorrect: Average length of incorrect answers
            - num_judge_calls: Total number of LLM judge calls
            - num_failed_judge_calls: Number of failed LLM judge calls
            - failed_thinking_tokens: Total tokens in answers without thinking tags
    """
    num_correct, num_incorrect = 0, 0
    total_len_correct, total_len_incorrect = 0, 0
    num_judge_calls, num_failed_judge_calls = 0, 0
    failed_thinking_tokens = 0
    overview_correct = []
    for run in results:
        local_correct = 0
        for result in run:
            if result.check_method == "LLM judge":
                num_judge_calls += 1
                if result.explanation == "LLM judge answer was invalid":
                    num_failed_judge_calls += 1

            if result.is_correct:
                total_len_correct += len(result.model_answer)
                num_correct += 1
                local_correct += 1
            else:
                total_len_incorrect += len(result.model_answer)
                num_incorrect += 1
                if result.explanation == "Answer contains no thinking tags":
                    failed_thinking_tokens += len(result.model_answer)
        overview_correct.append(local_correct)

    # calc pass@1
    num_correct_pass_1 = sum(overview_correct[0])

    # calc pass@k and majority@k
    majority_k_threshold = int(len(results) / 2) + 1  # strict majority
    num_correct_majority_k, num_correct_pass_k = 0, 0
    for idx in range(len(results[0])):
        local_correct = 0
        for run in results:
            if run[idx].is_correct:
                local_correct += 1
        if local_correct >= majority_k_threshold:
            num_correct_majority_k += 1
        if local_correct > 0:
            num_correct_pass_k += 1

    return {
        "pass@k": num_correct_pass_k / len(results[0]),
        "pass@1": num_correct_pass_1 / len(results[0]),
        "majority@k": num_correct_majority_k / len(results[0]),
        "avg_len_correct": total_len_correct / num_correct,
        "avg_len_incorrect": total_len_incorrect / num_incorrect,
        "num_judge_calls": num_judge_calls,
        "num_failed_judge_calls": num_failed_judge_calls,
        "failed_thinking_tokens": failed_thinking_tokens,
    }


def main():
    args = parse_arguments()
    seeds, temperatures, split = args.seeds, args.temps, args.dataset_split
    number_of_runs = len(seeds)

    df = load_dataset(split)
    questions = df["question"].tolist()
    ground_truths = df["answer"].tolist()

    llm = vllm.LLM(MODEL_PATHS["llm"], max_model_len=DEFAULT_SETTINGS["llm_max_length"])
    answer_checker = GSM8KAnswerChecker(model_name=DEFAULT_SETTINGS["llm"])

    # run the model for N times
    runs = []
    for i in range(number_of_runs):
        seed = seeds[i]
        temp = temperatures[i]
        runs.append(inference_vllm(llm, questions, seed, temp, MODEL_PATHS["llm"]))

    # judge the answers
    results = []
    for responses in runs:
        results.append(
            answer_checker.check_answer(
                question=questions, model_answer=responses, ground_truth=ground_truths
            )
        )

    # calculate metrics
    metrics = calculate_metrics(results)
    for name, metric in metrics.items():
        print(f"{name}: {metric:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Script execution time: {elapsed_time:.1f} seconds")

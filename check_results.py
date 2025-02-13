from typing import Any, Dict, List

from src.gsm8k_answer_checker import GSM8KAnswerCheckerResult


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
    # # judge the answers
    # results = []
    # for responses in runs:
    #     results.append(
    #         answer_checker.check_answer(
    #             question=questions, model_answer=responses, ground_truth=ground_truths
    #         )
    #     )

    # # calculate metrics
    # metrics = calculate_metrics(results)
    # for name, metric in metrics.items():
    #     print(f"{name}: {metric:.4f}")
    pass


if __name__ == "__main__":
    main()

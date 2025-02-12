from dataclasses import dataclass
from typing import List, Union

import numpy as np

from . import utils
from .llm_inference import inference_llm_judge_vllm


@dataclass
class GSM8KAnswerCheckerResult:
    """
    Represents the result of an answer validation check.

    Attributes:
        is_correct (bool): Whether the answer is correct.
        explanation (str): Explanation of the correctness of the answer.
        extracted_answer (str): The extracted numeric answer from the model's response.
        ground_truth (str): The correct answer provided as ground truth.
        check_method (str): The method used for validation (e.g., "Exact match" or "LLM judge").
    """

    is_correct: bool
    explanation: str
    extracted_answer: str
    ground_truth: str
    check_method: str


class GSM8KAnswerChecker:
    """
    A class to validate answers generated for GSM8K problems.

    This class checks whether a model-generated answer is correct using
    exact match comparison and an optional LLM-based judge.

    Attributes:
        model_name (str): The name of the model being evaluated.
        ignore_thinking_tags (bool): Whether to ignore `<think>` tags in the model's output.
    """

    def __init__(
        self,
        model_name: str,
        ignore_thinking_tags: bool = False,
    ) -> None:
        self.model_name = model_name
        self.ignore_thinking_tags = ignore_thinking_tags

        if self.ignore_thinking_tags:
            print("Ignoring thinking tags for answer validation")

    def check_answer(
        self,
        question: Union[str, List[str]],
        model_answer: Union[str, List[str]],
        ground_truth: Union[str, List[str]],
    ) -> List[GSM8KAnswerCheckerResult]:
        """
        Checks if the model's answer is correct by comparing it to the ground truth.

        Supports both single-question validation and batch processing.

        Args:
            question (Union[str, List[str]]): The question(s) being answered.
            model_answer (Union[str, List[str]]): The model-generated answer(s).
            ground_truth (Union[str, List[str]]): The correct answer(s) for comparison.

        Returns:
            List[GSM8KAnswerCheckerResult]:
                The result of the validation, either for a single question
                or a list of validation results for multiple questions.
        """
        assert type(question) is type(model_answer), "Question and answer must be the same type"
        assert type(question) is type(ground_truth), "Question and truth must be the same type"

        if isinstance(question, list):
            return self._check_answer_batch(question, model_answer, ground_truth)
        else:
            return self._check_answer_batch(list(question), list(model_answer), list(ground_truth))

    def _check_answer_batch(
        self, questions: List[str], model_answers: List[str], ground_truths: List[str]
    ) -> List[GSM8KAnswerCheckerResult]:
        """
        Checks answers for a batch of questions.

        This method applies exact match comparison and optionally uses
        an LLM judge if the extracted answer is malformed.

        Args:
            questions (List[str]): The list of questions being answered.
            model_answers (List[str]): The corresponding model-generated answers.
            ground_truths (List[str]): The correct answers for validation.

        Returns:
            List[GSM8KAnswerCheckerResult]: A list of validation results
                for each question-answer pair.
        """
        assert len(questions) == len(model_answers) == len(ground_truths), (
            "List of questions, answers, and ground truths must be the same length"
        )

        has_valid_think_tags = [
            utils.contains_thinking_sections(model_answer) for model_answer in model_answers
        ]

        # filter out all answers that don't have valid think tags
        results = np.full(len(model_answers), None)
        for idx, has_valid_think_tag in enumerate(has_valid_think_tags):
            if not has_valid_think_tag and not self.ignore_thinking_tags:
                results[idx] = GSM8KAnswerCheckerResult(
                    is_correct=False,
                    explanation="Answer contains no thinking tags",
                    extracted_answer=None,
                    ground_truth=ground_truths[idx],
                    check_method="Exact match",
                )

        # clean the answers
        model_answers = [
            utils.remove_thinking_sections(model_answer).strip() for model_answer in model_answers
        ]

        extracted_answers = [
            utils.extract_number_gsm8k(model_answer) for model_answer in model_answers
        ]

        malformed_answers = dict()
        for idx, result in enumerate(results):
            if result:
                continue

            if extracted_answers[idx]:
                is_correct = extracted_answers[idx] == ground_truths[idx]
                explanation = "Correct result" if is_correct else "Incorrect result"

                results[idx] = GSM8KAnswerCheckerResult(
                    is_correct=is_correct,
                    explanation=explanation,
                    extracted_answer=extracted_answers[idx],
                    ground_truth=ground_truths[idx],
                    check_method="Exact match",
                )

            else:
                malformed_answers[idx] = model_answers[idx]

        if malformed_answers:
            judge_output = inference_llm_judge_vllm(
                questions=[questions[idx] for idx in malformed_answers.keys()],
                answers=list(malformed_answers.values()),
                ground_truths=[ground_truths[idx] for idx in malformed_answers.keys()],
            )

            for idx in malformed_answers.keys():
                explanation = (
                    "LLM judge says 'correct'"
                    if judge_output[idx]
                    else "LLM judge says 'incorrect'"
                )

                results[idx] = GSM8KAnswerCheckerResult(
                    is_correct=judge_output[idx],
                    explanation=explanation,
                    extracted_answer=None,
                    ground_truth=ground_truths[idx],
                    check_method="LLM judge",
                )

        return results

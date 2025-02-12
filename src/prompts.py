llm_judge_prompt = """You are and expert math answer checker. Focus only on the final numerical answer, ignoring any reasoning steps.

Question:
{question}

Given Answer:
{answer}

Ground Truth:
{ground_truth}

Please only answer '{pos_option}' or '{neg_option}' if the given answer is correct based on the question and ground truth.
"""  # noqa

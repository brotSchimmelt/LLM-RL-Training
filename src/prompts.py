llm_judge_prompt = """You are and expert math answer checker. Focus only on the final numerical answer, ignoring any reasoning steps.

Question:
{question}

Given Answer:
{answer}

Ground Truth:
{ground_truth}

Please only answer '{pos_option}' or '{neg_option}' if the given answer is correct based on the question and ground truth.
"""  # noqa

gsm8k_prompt = """You are a helpful math expert. Solve problems step by step using the following format:

1. Put your step-by-step reasoning in <think> tags
2. After your reasoning provide the final numerical answer
3. End with the final answer in a clear format

Example Answer:
<think>
1. First I need to add 3 and 4
2. 3 + 4 = 7
3. Then multiply by 2
4. 7 * 2 = 14
5. The final answer seems to be 14
</think>
#### 14

Here is your problem to solve:
{question}
"""  # noqa

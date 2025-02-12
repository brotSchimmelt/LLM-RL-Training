import pytest

from src.llm_inference import apply_prompt_format


def test_apply_prompt_format_llama():
    prompts = ["How are you?"]
    model_name = "llama3"
    expected_output = [
        "<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  # noqa
    ]
    assert apply_prompt_format(prompts, model_name) == expected_output


def test_apply_prompt_format_qwen():
    prompts = ["Tell me a joke"]
    model_name = "qwen-7b"
    expected_output = ["<|im_start|>user\nTell me a joke<|im_end|>\n<|im_start|>assistant"]
    assert apply_prompt_format(prompts, model_name) == expected_output


def test_apply_prompt_format_invalid_model():
    prompts = ["Test"]
    model_name = "unknown_model"
    with pytest.raises(ValueError, match="No prompt template found in config.py for unknown_model"):
        apply_prompt_format(prompts, model_name)

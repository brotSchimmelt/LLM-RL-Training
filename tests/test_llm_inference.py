import pytest

from src.llm_inference import apply_prompt_format


def test_apply_prompt_format_llama():
    prompts = ["Hello, world!", "How are you?"]
    model_name = "llama3"
    expected_output = [
        "<|startoftext|>Hello, world!<|endoftext|>",
        "<|startoftext|>How are you?<|endoftext|>",
    ]
    assert apply_prompt_format(prompts, model_name) == expected_output


def test_apply_prompt_format_qwen():
    prompts = ["Tell me a joke", "What's the weather?"]
    model_name = "qwen-7b"
    expected_output = ["[CHATML]Tell me a joke[/CHATML]", "[CHATML]What's the weather?[/CHATML]"]
    assert apply_prompt_format(prompts, model_name) == expected_output


def test_apply_prompt_format_invalid_model():
    prompts = ["Test"]
    model_name = "unknown_model"
    with pytest.raises(ValueError, match="No prompt template found in config.py for unknown_model"):
        apply_prompt_format(prompts, model_name)

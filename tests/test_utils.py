import pytest

from src.utils import contains_thinking_sections, extract_number_gsm8k, remove_thinking_sections


def test_extract_number_gsm8k():
    assert extract_number_gsm8k("#### 42") == 42
    assert extract_number_gsm8k("Some text before #### 100 and after") == 100
    assert extract_number_gsm8k("No number here") is None
    assert extract_number_gsm8k("#### abc") is None
    assert extract_number_gsm8k("#### 007") == 7  # Leading zeros should be handled correctly
    assert extract_number_gsm8k("") is None


def test_remove_thinking_sections():
    assert remove_thinking_sections("Hello <think>this is hidden</think> world!") == "Hello  world!"
    assert remove_thinking_sections("<think>Just thinking</think>") == ""
    assert remove_thinking_sections("No thinking tags here") == "No thinking tags here"
    assert remove_thinking_sections("<think>Multiple</think> <think>thoughts</think>") == ""
    assert remove_thinking_sections("") == ""


def test_contains_thinking_sections():
    assert contains_thinking_sections("Hello <think>this is hidden</think> world!") is True
    assert contains_thinking_sections("<think>Just thinking</think>") is True
    assert contains_thinking_sections("No thinking tags here") is False
    assert contains_thinking_sections("<think>Multiple</think> <think>thoughts</think>") is True
    assert contains_thinking_sections("") is False


if __name__ == "__main__":
    pytest.main()

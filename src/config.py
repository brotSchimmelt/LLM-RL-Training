DEFAULT_SETTINGS = {
    "judge_model": "Qwen/Qwen2.5-7B-Instruct",
    "judge_max_model_length": 8_192,
    "llm": "unsloth/Llama-3.2-1B",
    "llm_max_length": 4_096,
    "model_save_dir": "./models/",
    "max_tokens": 8_192,
}

MODEL_PATHS = {
    "llm": f"{DEFAULT_SETTINGS['model_save_dir']}{DEFAULT_SETTINGS['llm'].replace('/', '_')}",
    "judge": f"{DEFAULT_SETTINGS['model_save_dir']}{DEFAULT_SETTINGS['judge_model'].replace('/', '_')}",  # noqa
}


PROMPT_FORMATS = {
    "llama3": {
        "prompt_template": "<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
        "system_prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
        "eos_token": "<|eot_id|>",
    },
    "chatml": {
        "prompt_template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
        "system_prompt_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",  # noqa: E501
        "eos_token": "<|im_end|>",
    },
}

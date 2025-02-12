DEFAULT_SETTINGS = {
    # "judge_model": "Qwen/Qwen2.5-7B-Instruct",
    "judge_model": "unsloth/Llama-3.2-1B",
    "judge_max_model_length": 100,
    "llm": "unsloth/Llama-3.2-1B",
    "model_save_dir": "./models/",
}

MODEL_PATHS = {
    "llm": f"{DEFAULT_SETTINGS['model_save_dir']}{DEFAULT_SETTINGS['llm'].replace('/', '_')}",
    "judge": f"{DEFAULT_SETTINGS['model_save_dir']}{DEFAULT_SETTINGS['judge_model'].replace('/', '_')}",  # noqa
}

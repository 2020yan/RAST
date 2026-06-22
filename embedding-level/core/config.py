__cfg = {
    'llama2-7b': {
        'model_nickname': 'llama2-7b',
        'model_name': 'meta-llama/Llama-2-7b-chat-hf', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama3-8b': {
        'model_nickname': 'llama3-8b',
        'model_name': '/home/ymy/data/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct', 
        'n_layer': 32, 
        'n_dimension': 4096
    },
    'qwen2.5-7b': {
        'model_nickname': 'qwen2.5-7b',
        'model_name': '/home/ymy/data/qwen25_model/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28',
        'n_layer': 28,
        'n_dimension': 3584
    },
}

class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)

def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])
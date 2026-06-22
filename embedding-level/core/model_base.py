from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import cfg, get_cfg
import torch

class ModelBase:
    def __init__(self, model_nickname: str):
        self.llm_cfg = get_cfg(model_nickname)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.llm_cfg.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.model_name)

    def apply_sft_template(self, instruction, system_message=None):
        if system_message is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
        return messages
    
    def apply_inst_template(self, text):
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        return messages
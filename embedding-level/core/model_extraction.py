from core.model_base import ModelBase
from core.embedding_manager import EmbeddingManager
import torch


class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

    def extract_embds(self, inputs: list[str], system_message: str=None, message: str=None) -> EmbeddingManager:
        embds_manager = EmbeddingManager(self.llm_cfg, message)
        embds_manager.layers = [
            torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
        ]

        for i, txt in enumerate(inputs):
            # 根据模型类型选择合适的模板应用方式
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                # 使用tokenizer内置的chat template
                messages = self.apply_sft_template(instruction=txt, system_message=system_message)
                encoded = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_tensors="pt",
                    return_dict=True
                )
            else:
                # 对于没有chat_template的模型（如Qwen 2.5），直接使用文本
                # Qwen模型通常使用特殊的格式
                if 'qwen' in self.llm_cfg.model_nickname.lower():
                    # Qwen模型使用 <|im_start|> 和 <|im_end|> 格式
                    if system_message:
                        formatted_text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{txt}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        formatted_text = f"<|im_start|>user\n{txt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    # 其他模型直接使用原始文本
                    formatted_text = txt
                
                encoded = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )

            # 从 BatchEncoding 中提取 input_ids
            if hasattr(encoded, 'input_ids'):
                input_ids = encoded.input_ids.to(self.device)
            else:
                # 如果返回的是 tensor，直接使用
                input_ids = encoded.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            for j in range(self.llm_cfg.n_layer):
                embds_manager.layers[j][i, :] = hidden_states[j][:, -1, :].detach().cpu()

        return embds_manager
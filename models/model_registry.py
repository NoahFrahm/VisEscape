from functools import lru_cache
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor
)


@lru_cache(maxsize=None)
def get_qwen_2_5_VL_7B():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct" # needs local storage due to large size
    print("INFO: Loading 'Qwen2_5_VL 7B' model...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        force_download=True
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    
    print("INFO: Loaded 'Qwen2_5_VL 7B' model successfully.")
    
    return model, processor

@lru_cache(maxsize=None)
def get_qwen_2_5_VL_32B():
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct" # needs local storage due to large size
    print("INFO: Loading 'Qwen2_5_VL 32B' model...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        force_download=True
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    
    print("INFO: Loaded 'Qwen2_5_VL 32B' model successfully.")
    
    return model, processor

@lru_cache(maxsize=None)
def get_qwen_3_VL_30B():
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct" # needs local storage due to large size
    print("INFO: Loading 'Qwen3_VL 30B' model...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        force_download=True
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    
    print("INFO: Loaded 'Qwen3_VL 30B' model successfully.")
    
    return model, processor
    

import base64
import io
import os

import time
from openai import OpenAI
from PIL import Image


def ensure_csv_ext(path: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() != ".csv":
        return root + ".csv"
    return path

def _encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_inference_vision_caption(
    image_path: str,
    prompt: str,
    model_name: str,
    base_url: str = None,
) -> str:
    """
    Unified vision caption inference function that works with both OpenAI and vLLM.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the vision model
        model_name: Full model name (e.g., 'gpt-4o-mini' or 'OpenGVLab/InternVL2_5-38B')
        base_url: Base URL for vLLM server. If None, uses OpenAI API.
    
    Returns:
        Generated caption string
    """
    base64_image = _encode_image(image_path)
    
    # Configure client based on whether base_url is provided
    if base_url:
        # vLLM mode
        client = OpenAI(api_key="EMPTY", base_url=base_url)
    else:
        # OpenAI mode
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Retry logic (for OpenAI rate limits)
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            if base_url:
                # vLLM: fail immediately
                return f"Failed to generate caption: {str(e)}"
            else:
                # OpenAI: retry after delay
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue


# Backward compatibility aliases
def run_inference_vision_caption_vllm(
    image_path: str,
    prompt: str,
    model_name: str,
    base_url: str,
) -> str:
    """Deprecated: Use run_inference_vision_caption instead"""
    return run_inference_vision_caption(image_path, prompt, model_name, base_url)


def run_inference_vision_caption_openai(
    image_path: str,
    prompt: str,
    model_name: str,
) -> str:
    """Deprecated: Use run_inference_vision_caption instead"""
    return run_inference_vision_caption(image_path, prompt, model_name, base_url=None)
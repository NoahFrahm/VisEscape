import torch
import time
from typing import Tuple
from qwen_vl_utils import process_vision_info
from models.model_registry import get_qwen_2_5_VL_7B, get_qwen_2_5_VL_32B
import requests
from typing import Optional


def qwen_infer(model, processor, prompt: list, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 4096) -> list:
        """
        prompt: [
            {
                "role": "user",
                "content": [
                    {"type":"image","image":"file://..."},
                    {"type":"text","text":"Describe this image."}
                ]
            }
        ]
        """

        text = processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(prompt)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device) # this is problematic for distributed inference

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        # get logits during inference for prediction set building from scores
        # generated_outputs = model.generate(**inputs,output_scores=True,output_logits=True,return_dict_in_generate=True,output_hidden_states=True,output_attentions=True, max_new_tokens=128)

        # Trim off prompt tokens before decoding
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

def _single_token_id(tokenizer, s: str):
    """Return an ID that is a single token for s, trying 's' then ' s'."""
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    ids_ws = tokenizer(" " + s, add_special_tokens=False)["input_ids"]
    if len(ids_ws) == 1:
        return ids_ws[0]
    raise ValueError(f"Label '{s}' is not a single token; try 'A.' style labels instead.")

def qwen_infer_mc(
    model,
    processor,
    prompt: list,
    labels=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"),
    calibration_prompt = None,
):
    """
    One forward pass MC scoring:
      - assumes `prompt` already asks for a single-letter answer from `labels`
      - returns {"pred", "probs", "labels"} where probs align with labels
    If `calibration_prompt` is provided (a content-free message list), applies
    contextual calibration by subtracting its next-token logits before softmax.
    """

    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, add_vision_id=True #NOTE vision id for referencing images?
    )  # next token is first assistant token

    # Prepare multimodal tensors via the processor
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Forward pass to read next-token logits (no generation)
    with torch.no_grad():
        out = model(**inputs)
        next_logits = out.logits[:, -1, :].float()  # [batch=1, vocab]

    # Map label strings -> single token IDs
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor  # VL processors hold .tokenizer
    label_ids = [_single_token_id(tok, s) for s in labels]
    label_ids_t = torch.tensor(label_ids, device=next_logits.device)
    # break

    # Optional contextual calibration: subtract bias logits from a content-free prompt
    # e.g., make this the same exact format as prompt but without any images
    restricted = next_logits[:, label_ids_t]
    if calibration_prompt is not None:
        null_text = processor.apply_chat_template(
            calibration_prompt, tokenize=False, add_generation_prompt=True
        )
        null_inputs = processor(
            text=[null_text], padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            null_out = model(**null_inputs)
            null_next_logits = null_out.logits[:, -1, :].float()

        # print("restricted pre calibration probs:", restricted.squeeze(0).tolist())
        # print("calibration probs:", null_next_logits[:, label_ids_t].squeeze(0).tolist())
        restricted = next_logits[:, label_ids_t] - null_next_logits[:, label_ids_t]
        # print("restricted post calibration:", restricted.squeeze(0).tolist())

    # Softmax over the restricted label set
    probs = torch.softmax(restricted, dim=-1).squeeze(0).tolist()
    pred = labels[int(torch.tensor(probs).argmax().item())]

    return {"pred": pred, "probs": probs, "labels": list(labels)}

def qwen_format_content_local(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{c[1]}"
                }
            )
    return formated_content


def call_qwen_api(sys_prompt, contents, mc=False, use_32B=False) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    calibration_message_text = None
    message_text = None

    formatted_content = qwen_format_content_local(contents[0] if mc else contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formatted_content},
    ]

    if mc:
        formatted_calibration_content = qwen_format_content_local(contents[1])
        calibration_message_text = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_calibration_content},
        ]        

    if use_32B:
        model, processor = get_qwen_2_5_VL_32B()
    else:
        model, processor = get_qwen_2_5_VL_7B()

    while retry_count < max_tries:
        try:
            if mc:
                prompt_answer = qwen_infer_mc(
                    model,
                    processor,
                    prompt=message_text,
                    calibration_prompt = calibration_message_text,
                )
                return prompt_answer
            else:
                prompt_answer = qwen_infer(
                    model=model,
                    processor=processor,
                    prompt=message_text
                )
                return prompt_answer[0]
        except Exception as e:
            print("Error: ", e)
            retry_count += 1
            continue

    return None



def call_qwen_api_server(sys_prompt, contents, port=12182, host_name='localhost', mc=False) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    calibration_message_text = None
    message_text = None

    formatted_content = qwen_format_content_local(contents[0] if mc else contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formatted_content},
    ]

    if mc:
        formatted_calibration_content = qwen_format_content_local(contents[1])
        calibration_message_text = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_calibration_content},
        ]

    while retry_count < max_tries:
        try:
            if mc:
                server_endpoint_link = f"http://{host_name}:{port}/Qwen_VL/infer_mc"
                resp = requests.post(
                            server_endpoint_link,
                            json={"prompt": message_text, "calibration_prompt": calibration_message_text},
                            timeout=300
                        )
                return resp.json()
            else:
                server_endpoint_link = f"http://{host_name}:{port}/Qwen_VL/infer"
                resp = requests.post(
                            server_endpoint_link,
                            json={"prompt": message_text},
                            timeout=300
                        )
                return resp.json()['text']
        except Exception as e:
            print("Error: ", e)
            retry_count += 1
            continue

    return None


def qwen_api_server_check(port, host_name):
    max_tries = 5
    retry_count = 0
    while retry_count < max_tries:
        try:
            resp = requests.get(
                        f"http://{host_name}:{port}/health",
                        timeout=300
                    )
            return resp.ok:
        except Exception as e:
            retry_count += 1
            continue
    return False
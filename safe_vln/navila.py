"""External NaViLA loading and prompt preparation for Safe-VLN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def build_navigation_prompt(instruction: str, num_video_frames: int = 8, conv_mode: str = "llama_3") -> str:
    from llava.conversation import conv_templates

    image_token = "<image>\n"
    question = (
        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {image_token * (num_video_frames - 1)}, and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" Analyze this series of images to decide your next action, '
        "which could be turning left or right by a specific degree, moving forward a certain distance, or stop "
        "if the task is completed."
    )
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


@dataclass
class PreparedState:
    input_ids: torch.Tensor
    images: object


class NavilaStatePreprocessor:
    def __init__(self, tokenizer, image_processor, model_config, *, device="cuda", dtype=torch.float16):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.device = device
        self.dtype = dtype

    def __call__(self, frames: Sequence, instruction: str) -> PreparedState:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        prompt = build_navigation_prompt(instruction, len(frames))
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        image_tensor = process_images(frames, self.image_processor, self.model_config)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)
        return PreparedState(input_ids=input_ids, images=[image_tensor])


def load_safe_navila(
    model_path: str,
    *,
    device: str = "cuda",
    add_lora: bool = True,
    checkpoint: str | None = None,
):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    from .model import SafeNavilaActorCritic, add_lora_adapters

    model_name = get_model_name_from_path(model_path)
    tokenizer, base_model, image_processor, _ = load_pretrained_model(model_path, model_name, None)
    base_model.requires_grad_(False)
    if checkpoint:
        from peft import PeftModel

        base_model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=add_lora)
    elif add_lora:
        base_model = add_lora_adapters(base_model)
    safe_model = SafeNavilaActorCritic(base_model, tokenizer).to(device)
    if checkpoint:
        safe_model.load_safe_heads(checkpoint, map_location=device)
    preprocessor = NavilaStatePreprocessor(tokenizer, image_processor, base_model.config, device=device)
    return safe_model, preprocessor

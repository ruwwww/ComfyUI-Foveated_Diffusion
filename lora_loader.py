"""
LoRA loader for Foveated Diffusion — wraps ComfyUI LoRA loading with
HuggingFace auto-download fallback.
"""

import os
import sys

import torch


HF_REPO = "bchao1/foveated-diffusion"
IMAGE_LORA_FILES = {
    "random": "image/fov_random.safetensors",
    "saliency": "image/fov_saliency.safetensors",
    "bbox": "image/fov_bbox.safetensors",
}


def resolve_lora_path(lora_mode: str, lora_path_from_input: str = "") -> str:
    """
    Resolve a LoRA checkpoint path.

    Args:
        lora_mode: one of "auto_download_random", "auto_download_saliency",
                   "auto_download_bbox", "from_path"
        lora_path_from_input: required when lora_mode="from_path"

    Returns:
        absolute path to the .safetensors file

    Raises:
        ValueError if the mode/path is invalid or file not found.
    """
    if lora_mode == "from_path":
        path = lora_path_from_input.strip()
        if not path:
            raise ValueError(
                "FoveatedDiffusion/LoadLoRA: lora_path is empty but lora_mode='from_path'. "
                "Provide a path to the LoRA safetensors file."
            )
        if not os.path.isfile(path):
            raise ValueError(
                f"FoveatedDiffusion/LoadLoRA: LoRA file not found at '{path}'"
            )
        return os.path.abspath(path)

    if lora_mode.startswith("auto_download"):
        mode_key = lora_mode[len("auto_download_"):]
        if mode_key not in IMAGE_LORA_FILES:
            raise ValueError(
                f"FoveatedDiffusion/LoadLoRA: unknown auto_download mode '{lora_mode}'. "
                f"Available: {list('auto_download_' + k for k in IMAGE_LORA_FILES)}"
            )
        filename = IMAGE_LORA_FILES[mode_key]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "FoveatedDiffusion/LoadLoRA: huggingface_hub is required for auto_download. "
                "Install with: pip install huggingface_hub"
            )
        return hf_hub_download(repo_id=HF_REPO, filename=filename)

    raise ValueError(
        f"FoveatedDiffusion/LoadLoRA: unknown lora_mode '{lora_mode}'"
    )


def load_lora_weights(model_patcher, lora_path: str, strength: float = 1.0):
    """
    Load LoRA weights into a cloned ModelPatcher.

    Args:
        model_patcher: cloned ModelPatcher from model.clone()
        lora_path: path to .safetensors LoRA file
        strength: LoRA strength multiplier

    Returns:
        model_patcher with LoRA applied
    """
    import logging
    import comfy.utils
    import comfy.lora
    import comfy.lora_convert
    import comfy.model_patcher

    logger = logging.getLogger("comfyui_foveated_diffusion")

    lora_sd = comfy.utils.load_torch_file(lora_path)
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)

    key_map = comfy.lora.model_lora_keys_unet(model_patcher.model)
    loaded = comfy.lora.load_lora(lora_sd, key_map)

    loaded_keys = list(loaded.keys())
    logger.info(
        f"Loaded {len(loaded_keys)} LoRA key(s) from {lora_path} "
        f"(strength={strength})"
    )
    if len(loaded_keys) == 0:
        logger.warning(
            f"No LoRA keys were mapped from {lora_path} — the LoRA may use an "
            f"unsupported naming convention. "
            f"Check that the safetensors keys match Flux2 parameter names. "
            f"First 5 lora keys: {list(lora_sd.keys())[:5]}"
        )
    elif len(loaded_keys) < 10:
        logger.warning(
            f"Only {len(loaded_keys)} LoRA key(s) loaded — this is fewer than expected "
            f"for a full Flux2 LoRA. The LoRA may be partially applied."
        )

    model_patcher.add_patches(loaded, strength)

    return model_patcher
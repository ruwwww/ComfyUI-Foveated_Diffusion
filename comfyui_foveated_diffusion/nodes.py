"""
ComfyUI Foveated Diffusion — Node Definitions.

Five nodes:
    1. FoveationMaskNode   — generates binary foveation mask in token-grid space
    2. LoadFoveatedLoRA    — loads foveated FLUX.2 Klein LoRA into cloned ModelPatcher
    3. FoveatedKSampler    — full foveated sampling loop (DIFFUSION_MODEL wrapper)
    4. FoveatedVAEDecode   — decode foveated latents with HR/LR blend (merge mode)
    5. FoveationMaskPreview — overlay mask on image for visualization
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.patcher_extension
from einops import rearrange

from .crpa_attention import build_crpa_state, crpa_attn1_patch, crpa_attn1_output_patch
from .foveated_tokenizer import (
    build_foveated_tokens,
    build_crpa_img_ids,
    reconstruct_tokens,
    build_fovea_mask,
)
from .lora_loader import resolve_lora_path, load_lora_weights


# ---------------------------------------------------------------------------
# Node 1: FoveationMaskNode
# ---------------------------------------------------------------------------

class FoveationMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mask_shape": (["circular", "square", "ellipse"], {"default": "circular"}),
                "center_x": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Horizontal gaze position. -1=left, +1=right, 0=center.",
                    },
                ),
                "center_y": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Vertical gaze position. -1=top, +1=bottom, 0=center.",
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Foveal radius as fraction of image half-width. "
                            "0.30 = ~30% of image width is high-res."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("FOVEATION_MASK",)
    RETURN_NAMES = ("foveation_mask",)
    FUNCTION = "build_mask"
    CATEGORY = "foveated_diffusion"

    def build_mask(self, latent, mask_shape, center_x, center_y, radius):
        samples = latent["samples"]
        B, C, H_lat, W_lat = samples.shape

        # For Flux2, the token grid IS the latent grid (patch_size=1)
        # Verify divisibility for common lr_factor values
        mask = build_fovea_mask(
            h_tok=H_lat,
            w_tok=W_lat,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            mask_shape=mask_shape,
            device=samples.device,
            lr_factor=2,
        )

        fov_mask = {
            "mask": mask,          # (H_tok, W_tok) BoolTensor, True = HR region
            "h": H_lat,
            "w": W_lat,
            "lr_factor": 2,         # placeholder; overridden by LoadFoveatedLoRA in pipeline
        }
        return (fov_mask,)

    @classmethod
    def IS_CHANGED(cls, latent, mask_shape, center_x, center_y, radius):
        return (mask_shape, center_x, center_y, radius)


# ---------------------------------------------------------------------------
# Node 2: LoadFoveatedLoRA
# ---------------------------------------------------------------------------

class LoadFoveatedLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_mode": (
                    [
                        "auto_download_random",
                        "auto_download_saliency",
                        "auto_download_bbox",
                        "from_path",
                    ],
                    {"default": "auto_download_random"},
                ),
                "lr_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 4,
                        "step": 2,
                        "tooltip": (
                            "LR periphery downsampling factor. "
                            "2 = 4x fewer peripheral tokens. "
                            "4 = 16x fewer peripheral tokens."
                        ),
                    },
                ),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Required when lora_mode='from_path'. Absolute path to .safetensors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "INT")
    RETURN_NAMES = ("model", "lr_factor")
    FUNCTION = "load_lora"
    CATEGORY = "foveated_diffusion"

    def load_lora(self, model, lora_mode, lr_factor, lora_strength, lora_path=""):
        model_clone = model.clone()
        model_clone.model_options["transformer_options"]["fov_lr_factor"] = lr_factor

        if lora_strength <= 0.0:
            return (model_clone, lr_factor)

        resolved_path = resolve_lora_path(lora_mode, lora_path)
        model_clone = load_lora_weights(model_clone, resolved_path, lora_strength)

        return (model_clone, lr_factor)


# ---------------------------------------------------------------------------
# DIFFUSION_MODEL wrapper for foveated tokenization
# ---------------------------------------------------------------------------

def _foveated_diffusion_model_wrapper(executor, x, timestep, context, y, guidance,
                                       ref_latents, control, transformer_options,
                                       **kwargs):
    """Wrapper injected via WrappersMP.DIFFUSION_MODEL.

    1. Foveated-tokenize the latent (top-left LR selection)
    2. Call forward_orig with the foveated token sequence
       (standard attention + LoRA handles the mixed-resolution tokens)
    3. Detokenize DiT output back to full-resolution latent
    """
    import logging
    logger = logging.getLogger("comfyui_foveated_diffusion")

    fov_state = transformer_options.get("fov_state", None)
    if fov_state is None:
        logger.warning(
            "FoveatedDiffusion: called without fov_state — passing through. "
            "Wire a FoveatedKSampler into the pipeline."
        )
        return executor(x, timestep, context, y, guidance,
                        ref_latents, control, transformer_options, **kwargs)

    model = executor.class_obj  # Flux nn.Module
    mask = fov_state["mask"]
    lr_factor = fov_state["lr_factor"]

    B, C_lat, H_orig, W_orig = x.shape
    patch_size = model.patch_size
    n_axes = len(model.params.axes_dim)

    # ── 1. Full-res tokenisation via process_img ───────────────────
    img, img_ids = model.process_img(x, transformer_options=transformer_options)
    C_patched = img.shape[2]

    h_len = ((H_orig + (patch_size // 2)) // patch_size)
    w_len = ((W_orig + (patch_size // 2)) // patch_size)

    img_spatial = img.view(B, h_len, w_len, C_patched)
    img_ids_spatial = img_ids[0].view(h_len, w_len, n_axes)

    # ── 2. Foveate tokens (top-left selection for LR blocks) ─────
    img_fov, fov_indices = build_foveated_tokens(img_spatial, mask, lr_factor)
    img_ids_fov = build_crpa_img_ids(img_ids_spatial, mask, lr_factor, x.device, torch.float32)
    img_ids_fov = img_ids_fov.unsqueeze(0).expand(B, -1, -1)

    n_tokens_input = h_len * w_len
    n_tokens_fov = img_fov.shape[1]
    logger.info(
        "FoveatedDiffusion: %d -> %d tokens (%.1f%% reduction, lr_factor=%d)",
        n_tokens_input, n_tokens_fov,
        (1.0 - n_tokens_fov / n_tokens_input) * 100,
        lr_factor,
    )

    # ── 3. Build txt_ids (same as _forward) ────────────────────────
    txt_ids = torch.zeros((B, context.shape[1], n_axes), device=x.device, dtype=torch.float32)
    if len(model.params.txt_ids_dims) > 0:
        for i in model.params.txt_ids_dims:
            txt_ids[:, :, i] = torch.linspace(
                0, context.shape[1] - 1, steps=context.shape[1],
                device=x.device, dtype=torch.float32,
            )

    # ── 4. Build CRPA state (dual RoPE, resolution masks) ─────────
    crpa_state = build_crpa_state(
        img_ids_fov=img_ids_fov,
        txt_ids=txt_ids,
        resolution_mask=fov_indices["resolution_mask"],
        resolution_mask_top_left=fov_indices["resolution_mask_top_left"],
        lr_factor=lr_factor,
        pe_embedder=model.pe_embedder,
    )

    # ── 5. Inject CRPA attention patches ──────────────────────────
    transformer_options = transformer_options.copy()
    transformer_options["crpa_state"] = crpa_state
    to = transformer_options

    comfy.patcher_extension.merge_nested_dicts(
        to.setdefault("patches", {}),
        {"attn1_patch": [crpa_attn1_patch], "attn1_output_patch": [crpa_attn1_output_patch]},
        copy_dict1=False,
    )

    # ── 6. Run forward_orig with foveated sequence ────────────────
    out = model.forward_orig(
        img_fov, img_ids_fov, context, txt_ids,
        timestep, y, guidance, control,
        transformer_options=transformer_options,
    )

    # ── 5. Detokenize ─────────────────────────────────────────────
    out_full = reconstruct_tokens(
        out, fov_indices, B, C_patched, h_len, w_len, mask, lr_factor,
        device=x.device, dtype=x.dtype,
    )
    out_full = rearrange(
        out_full, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_len, w=w_len, ph=patch_size, pw=patch_size,
    )[:, :, :H_orig, :W_orig]

    return out_full



# ---------------------------------------------------------------------------
# Node 3: FoveatedKSampler
# ---------------------------------------------------------------------------

class FoveatedKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "foveation_mask": ("FOVEATION_MASK",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": (
                    "INT",
                    {"default": 50, "min": 1, "max": 200},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "decode_mode": (
                    ["direct", "merge"],
                    {"default": "direct"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "foveated_diffusion"

    def sample(
        self, model, positive, negative, latent_image, foveation_mask,
        seed, steps, cfg, sampler_name, scheduler, denoise, decode_mode,
    ):
        lr_factor = model.model_options.get(
            "transformer_options", {}
        ).get("fov_lr_factor", 2)

        mask_tensor = foveation_mask["mask"]
        mask_h = foveation_mask["h"]
        mask_w = foveation_mask["w"]

        samples = latent_image["samples"]
        B, C, H_lat, W_lat = samples.shape

        if mask_h != H_lat or mask_w != W_lat:
            raise ValueError(
                f"FoveatedDiffusion/KSampler: mask shape ({mask_h},{mask_w}) "
                f"does not match latent shape ({H_lat},{W_lat})"
            )

        if H_lat % lr_factor != 0 or W_lat % lr_factor != 0:
            raise ValueError(
                f"FoveatedDiffusion/KSampler: latent dims ({H_lat},{W_lat}) "
                f"not divisible by lr_factor ({lr_factor})"
            )

        # Clone model and inject wrapper + fov_state
        model_clone = model.clone()
        te = model_clone.model_options.get("transformer_options", {})
        te["fov_state"] = {
            "mask": mask_tensor,
            "lr_factor": lr_factor,
        }

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "foveated_diffusion",
            _foveated_diffusion_model_wrapper,
            te,
            is_model_options=False,
        )
        model_clone.model_options["transformer_options"] = te

        # Run sampling (ComfyUI standard path)
        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)
        noise_mask = latent_image.get("noise_mask", None)

        latents_out = comfy.sample.sample(
            model_clone,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image["samples"],
            denoise=denoise,
            noise_mask=noise_mask,
            seed=seed,
        )

        if decode_mode == "merge":
            return ({"samples": latents_out},)

        # direct mode: reconstruct and blend before returning
        out = self._direct_decode(
            latents_out, mask_tensor, lr_factor, model
        )
        return ({"samples": out},)

    @staticmethod
    def _direct_decode(samples, mask, lr_factor, model):
        """
        Direct decode: return the DiT-reconstructed latent as-is.

        The DIFFUSION_MODEL wrapper already reconstructs full-resolution tokens
        via nearest-neighbor expansion of LR tokens at each denoising step.
        No additional post-processing is needed — any further smoothing would
        double-degrade the periphery and introduce boundary artifacts.
        """
        return samples


# ---------------------------------------------------------------------------
# Node 4: FoveatedVAEDecode
# ---------------------------------------------------------------------------

class FoveatedVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "foveation_mask": ("FOVEATION_MASK",),
                "blend_mode": (
                    ["hard", "soft_gaussian"],
                    {"default": "hard"},
                ),
                "blend_sigma": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.5,
                        "tooltip": "Gaussian blur sigma for soft boundary.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "foveated_diffusion"

    def decode(self, samples, vae, foveation_mask, blend_mode, blend_sigma):
        """
        Merge-mode decode: decode HR and LR latents separately,
        then blend in pixel space.
        """
        latent = samples["samples"]
        mask = foveation_mask["mask"]
        lr_factor = foveation_mask.get("lr_factor", 2)
        B, C, H_lat, W_lat = latent.shape

        # Build region masks in latent space
        mask_float = mask.float().to(latent.device)

        # Decode the fully reconstructed high-res latent directly (no zeroing out)
        hr_image = vae.decode(latent)

        # LR latent: downsample latent spatially to its native low-res resolution
        lr_latent = F.avg_pool2d(latent, kernel_size=lr_factor, stride=lr_factor)
        # Decode at native low-res resolution
        lr_image = vae.decode(lr_latent)

        # VAE outputs images in NHWC format: (B, H, W, C)
        # Permute to NCHW for spatial interpolation and blending: (B, C, H, W)
        hr_image_nchw = hr_image.permute(0, 3, 1, 2)
        lr_image_nchw = lr_image.permute(0, 3, 1, 2)

        # Upsample mask and low-res image to pixel resolution
        _, _, H_pix, W_pix = hr_image_nchw.shape
        lr_image_nchw = F.interpolate(
            lr_image_nchw, size=(H_pix, W_pix), mode="bicubic", align_corners=False
        )
        mask_pixel = F.interpolate(
            mask_float.unsqueeze(0).unsqueeze(0), size=(H_pix, W_pix), mode="bilinear", align_corners=False
        )
        mask_pixel = mask_pixel.clamp(0.0, 1.0)

        if blend_mode == "soft_gaussian" and blend_sigma > 0:
            mask_pixel = _gaussian_blur_2d(
                mask_pixel, blend_sigma, latent.device, mask_pixel.dtype
            )
            mask_pixel = mask_pixel.clamp(0.0, 1.0)

        merged_nchw = lr_image_nchw * (1.0 - mask_pixel) + hr_image_nchw * mask_pixel
        # Permute back to ComfyUI NHWC format
        merged = merged_nchw.permute(0, 2, 3, 1)

        return (merged,)


def _gaussian_blur_2d(x, sigma, device, dtype):
    """Separable Gaussian blur on 4D tensor (B, C, H, W)."""
    import math

    k = max(3, int(math.ceil(3 * sigma)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2 + 1e-6))
    g = g / g.sum()
    pad = k // 2
    out = F.conv2d(x, g.view(1, 1, k, 1), padding=(pad, 0))
    out = F.conv2d(out, g.view(1, 1, 1, k), padding=(0, pad))
    return out


# ---------------------------------------------------------------------------
# Node 5: FoveationMaskPreview
# ---------------------------------------------------------------------------

class FoveationMaskPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "foveation_mask": ("FOVEATION_MASK",),
                "overlay_color": (
                    ["white", "red", "green", "blue"],
                    {"default": "white"},
                ),
                "overlay_alpha": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "foveated_diffusion"

    def preview(self, image, foveation_mask, overlay_color, overlay_alpha):
        mask = foveation_mask["mask"].float()
        mask_h, mask_w = mask.shape

        # image shape: (B, H, W, C) in ComfyUI IMAGE type
        B, H_img, W_img, C_img = image.shape

        # Upsample mask to image resolution
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask = F.interpolate(
            mask, size=(H_img, W_img), mode="nearest"
        ).squeeze(0).squeeze(0)  # (H_img, W_img)

        color_map = {
            "white": torch.tensor([1.0, 1.0, 1.0], device=image.device),
            "red": torch.tensor([1.0, 0.0, 0.0], device=image.device),
            "green": torch.tensor([0.0, 1.0, 0.0], device=image.device),
            "blue": torch.tensor([0.0, 0.0, 1.0], device=image.device),
        }
        color = color_map[overlay_color].view(1, 1, 1, 3)

        overlay = mask.unsqueeze(-1).unsqueeze(0) * overlay_alpha  # (1, H, W, 1)
        result = image * (1.0 - overlay) + color * overlay

        return (result,)
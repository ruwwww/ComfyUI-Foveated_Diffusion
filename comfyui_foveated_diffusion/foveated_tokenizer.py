"""
Foveated Tokenization — core math for mixed-resolution latent token sequences.

All functions are pure tensor math with no ComfyUI dependencies.
Designed for FLUX.2 Klein where patch_size=1 (tokens == latent pixels).

Token layout: the flux DiT expects (B, L, C) where L=H*W in token space.
Foveated tokenization replaces L with a smaller L_fov = m + (H//d)*(W//d):
  - HR tokens: all tokens at positions where mask==1 are kept
  - LR tokens: one token per d×d block in the periphery is kept
"""

import torch
import torch.nn.functional as torch_F
from einops import rearrange


def build_foveated_tokens(
    img: torch.Tensor,
    mask: torch.BoolTensor,
    lr_factor: int = 2,
):
    """
    Foveated-tokenize a spatial token tensor.

    Args:
        img: token tensor (B, H, W, C) in spatial layout.
             From Flux process_img output, reshaped to spatial.
        mask: foveal mask (H, W) bool tensor, True = HR region.
        lr_factor: peripheral downsample factor d (2 or 4).

    Returns:
        img_fov: foveated token sequence (B, L_fov, C)
        fov_indices: dict with bookkeeping for reconstruction:
          - "hr_indices": (m,) LongTensor — flat indices into (H*W) where HR tokens live
          - "H": int, "W": int — original spatial dims
          - "lr_factor": int
          - "mask": BoolTensor — original mask (for reconstruction)
    """
    B, H, W, C = img.shape
    if H % lr_factor != 0 or W % lr_factor != 0:
        raise ValueError(
            f"FoveatedDiffusion: latent dims ({H},{W}) must be divisible by lr_factor ({lr_factor}). "
            f"Pad or resize the latent first."
        )
    if mask.shape != (H, W):
        raise ValueError(
            f"FoveatedDiffusion: mask shape {tuple(mask.shape)} does not match "
            f"token spatial dims ({H}, {W})"
        )

    h_d, w_d = H // lr_factor, W // lr_factor
    device = img.device
    dtype = img.dtype

    img_flat = img.view(B, H * W, C)  # (B, H*W, C)
    mask_flat = mask.reshape(-1)  # (H*W,)

    # Identify HR tokens: positions where mask == True
    hr_indices = torch.where(mask_flat)[0]  # (m,)
    hr_tokens = img_flat[:, hr_indices, :]  # (B, m, C)

    # LR blocks: each d×d block contributes 1 token (block mean)
    img_blocks = img.view(B, h_d, lr_factor, w_d, lr_factor, C)
    img_blocks = img_blocks.permute(0, 1, 3, 2, 4, 5)  # (B, h_d, w_d, d, d, C)
    img_blocks = img_blocks.reshape(B, h_d * w_d, lr_factor * lr_factor, C)

    # LR token: block mean — consistent with build_crpa_img_ids (position mean)
    lr_tokens = img_blocks.mean(dim=2)  # (B, h_d*w_d, C)

    # Concatenate HR tokens + LR tokens
    img_fov = torch.cat([hr_tokens, lr_tokens], dim=1)  # (B, m + h_d*w_d, C)

    fov_indices = {
        "hr_indices": hr_indices,        # (m,) — flat indices of HR tokens
        "H": H,
        "W": W,
        "lr_factor": lr_factor,
        "mask": mask,                    # original mask
    }
    return img_fov, fov_indices


def build_crpa_img_ids(
    img_ids: torch.Tensor,
    mask: torch.BoolTensor,
    lr_factor: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build CRPA-corrected img_ids for the foveated token sequence.

    The img_ids tensor encodes (index_axis, row, col, ...) for each token.
    For FLUX2 this is (T, H, W, L) or (index, height, width) depending on axes_dim.

    CRPA rule:
      - HR tokens keep their original img_ids
      - LR tokens get the CENTER coordinate of their d×d block, in HR coordinate space

    This ensures HR tokens use exact grid positions and LR tokens use block centers,
    creating a unified HR coordinate system without phase aliasing.

    Args:
        img_ids: (H, W, n_axes) float tensor of position IDs for each spatial token,
                 as produced by process_img for a single sample (batch dim removed).
        mask: (H, W) BoolTensor, True = HR region.
        lr_factor: d.
        device, dtype: target tensor specs.

    Returns:
        img_ids_fov: (L_fov, n_axes) float tensor.
    """
    H, W, n_axes = img_ids.shape
    if mask.shape != (H, W):
        raise ValueError(
            f"FoveatedDiffusion: mask shape {tuple(mask.shape)} != img_ids spatial ({H}, {W})"
        )
    if H % lr_factor != 0 or W % lr_factor != 0:
        raise ValueError(
            f"FoveatedDiffusion: img_ids spatial dims ({H},{W}) not divisible by "
            f"lr_factor ({lr_factor})"
        )

    h_d, w_d = H // lr_factor, W // lr_factor

    mask_flat = mask.reshape(-1)
    hr_indices = torch.where(mask_flat)[0]
    hr_ids = img_ids.reshape(H * W, n_axes)[hr_indices]  # (m, n_axes)

    # Compute LR token img_ids: mean of block positions → block center
    img_ids_blocks = img_ids.view(h_d, lr_factor, w_d, lr_factor, n_axes)
    img_ids_blocks = img_ids_blocks.permute(0, 2, 1, 3, 4)  # (h_d, w_d, d, d, n_axes)

    # Mean over the d×d spatial positions within each block.
    # For even d this gives the exact geometric center (e.g. (0.5, 0.5) for d=2),
    # which matches the block-mean token content in build_foveated_tokens.
    lr_ids = img_ids_blocks.mean(dim=(2, 3))  # (h_d, w_d, n_axes)
    lr_ids = lr_ids.reshape(h_d * w_d, n_axes)

    img_ids_fov = torch.cat([hr_ids, lr_ids], dim=0).to(device=device, dtype=dtype)
    return img_ids_fov


def reconstruct_tokens(
    tokens: torch.Tensor,
    fov_indices: dict,
    B: int,
    C: int,
    H: int,
    W: int,
    mask: torch.BoolTensor,
    lr_factor: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Reconstruct full-resolution token sequence from foveated tokens.

    HR tokens are scattered back to their original positions.
    LR tokens are upsampled via nearest-neighbor to fill their d×d blocks.

    Args:
        tokens: (B, L_fov, C) foveated token sequence from DiT output.
        fov_indices: bookkeeping dict from build_foveated_tokens.
        B, C, H, W: target output shape (B, H*W, C).
        mask: (H, W) BoolTensor.
        lr_factor: d.

    Returns:
        tokens_full: (B, H*W, C) full-resolution token sequence.
    """
    if device is None:
        device = tokens.device
    if dtype is None:
        dtype = tokens.dtype

    hr_indices = fov_indices["hr_indices"]   # (m,)
    h_d, w_d = H // lr_factor, W // lr_factor

    # Allocate full output
    tokens_full = torch.zeros(B, H * W, C, device=device, dtype=dtype)

    # Scatter HR tokens back
    m = hr_indices.shape[0]
    tokens_full[:, hr_indices, :] = tokens[:, :m, :]

    # Upsample LR tokens: nearest-neighbor via F.interpolate
    lr_tokens = tokens[:, m:, :]  # (B, h_d*w_d, C)
    lr_4d = lr_tokens.view(B, h_d, w_d, C).permute(0, 3, 1, 2)  # (B, C, h_d, w_d)
    lr_expanded_4d = torch_F.interpolate(lr_4d, size=(H, W), mode="nearest")
    lr_expanded = lr_expanded_4d.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # Build LR-only token mask: positions wholly inside LR blocks
    mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor)
    mask_blocks = mask_blocks.permute(0, 2, 1, 3).reshape(h_d, w_d, lr_factor * lr_factor)
    is_lr_block = mask_blocks.sum(dim=-1) == 0  # (h_d, w_d)
    is_lr_4d = is_lr_block.float()[None, None, :, :]  # (1, 1, h_d, w_d)
    lr_mask_4d = torch_F.interpolate(is_lr_4d, size=(H, W), mode="nearest")  # (1, 1, H, W)
    lr_mask = lr_mask_4d.squeeze().bool().view(-1)  # (H*W,)

    # Fill LR positions
    tokens_full[:, lr_mask, :] = lr_expanded[:, lr_mask, :]

    return tokens_full


def build_fovea_mask(
    h_tok: int,
    w_tok: int,
    center_x: float = 0.0,
    center_y: float = 0.0,
    radius: float = 0.30,
    mask_shape: str = "circular",
    device: torch.device = None,
    lr_factor: int = 2,
) -> torch.Tensor:
    """
    Build a binary foveation mask in token grid space.

    Mask values: True = high-resolution (foveal) region, False = periphery.

    Args:
        h_tok, w_tok: token grid dimensions (e.g., 64 for 1024/16).
        center_x: horizontal center, normalized [-1, 1], 0 = center.
        center_y: vertical center, normalized [-1, 1], 0 = center.
        radius: foveal radius as fraction of image half-width.
        mask_shape: "circular", "square", or "ellipse".
        device: torch device.
        lr_factor: d, used only for enforcing divisibility constraints.

    Returns:
        mask: (h_tok, w_tok) BoolTensor.
    """
    if device is None:
        device = torch.device("cpu")

    if h_tok % lr_factor != 0 or w_tok % lr_factor != 0:
        raise ValueError(
            f"FoveatedDiffusion: token grid ({h_tok},{w_tok}) must be divisible "
            f"by lr_factor ({lr_factor})"
        )

    y_range = torch.arange(h_tok, device=device, dtype=torch.float32)
    x_range = torch.arange(w_tok, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

    # Convert normalized [-1,1] center to pixel-space center
    cx = (center_x + 1.0) * 0.5 * (w_tok - 1)
    cy = (center_y + 1.0) * 0.5 * (h_tok - 1)

    # Normalize coordinates to [-1, 1] range for radius comparison
    x_norm = (xx - cx) / (w_tok * 0.5)
    y_norm = (yy - cy) / (h_tok * 0.5)
    dist_sq = x_norm ** 2 + y_norm ** 2

    if mask_shape == "circular":
        mask = dist_sq <= radius ** 2
    elif mask_shape == "square":
        mask = (x_norm.abs() <= radius) & (y_norm.abs() <= radius)
    elif mask_shape == "ellipse":
        mask = (x_norm / 1.0) ** 2 + (y_norm / 1.0) ** 2 <= radius ** 2
    else:
        raise ValueError(
            f"FoveatedDiffusion: unknown mask_shape '{mask_shape}'. "
            f"Expected 'circular', 'square', or 'ellipse'."
        )

    return mask
"""
Foveated Tokenization — core math for mixed-resolution latent token sequences.

Based on the official implementation of Foveated Diffusion:
- High-res blocks keep all lr_factor*lr_factor tokens.
- Low-res blocks keep only the top-left token.
- Keeps token sequence fully aligned and non-overlapping.
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
        mask: foveal mask (H, W) bool tensor, True = HR region.
        lr_factor: peripheral downsample factor d.

    Returns:
        img_fov: foveated token sequence (B, L_fov, C)
        fov_indices: dict with bookkeeping for reconstruction:
          - "valid_tokens": bool tensor of shape (h_d, w_d, n_per_block)
          - "is_low_res_block": bool tensor of shape (h_d, w_d)
          - "H": int, "W": int
          - "lr_factor": int
    """
    B, H, W, C = img.shape
    if H % lr_factor != 0 or W % lr_factor != 0:
        raise ValueError(f"Latent dims ({H},{W}) must be divisible by lr_factor ({lr_factor}).")
    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match token spatial dims ({H}, {W})")

    n_per_block = lr_factor * lr_factor
    h_d, w_d = H // lr_factor, W // lr_factor
    device = img.device

    # Reshape image into blocks: (B, h_d, lr_factor, w_d, lr_factor, C)
    # Permute and reshape to (B, h_d, w_d, n_per_block, C)
    img_blocks = img.view(B, h_d, lr_factor, w_d, lr_factor, C)
    img_blocks = img_blocks.permute(0, 1, 3, 2, 4, 5).reshape(B, h_d, w_d, n_per_block, C)

    # Classify blocks
    mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
    is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
    is_low_res_block = ~is_high_res_block

    # Prepare downsampled low-res tokens for low-res blocks
    # Average downsampling (like the official average mode)
    # img is (B, H, W, C)
    img_spatial = img.permute(0, 3, 1, 2) # (B, C, H, W)
    img_down = torch_F.interpolate(img_spatial, scale_factor=1.0 / lr_factor, mode="bilinear") * float(lr_factor)
    img_down = img_down.permute(0, 2, 3, 1) # (B, h_d, w_d, C)

    # Construct output sequence
    output_blocks = img_blocks.clone()
    # Replace index 0 of low-res blocks with the downsampled representation
    output_blocks_flat = output_blocks.view(B, h_d * w_d, n_per_block, C)
    img_down_flat = img_down.reshape(B, h_d * w_d, C)
    is_low_res_block_flat = is_low_res_block.view(-1)
    
    output_blocks_flat[:, is_low_res_block_flat, 0, :] = img_down_flat[:, is_low_res_block_flat, :]

    # Mask to select valid tokens
    valid_tokens = torch.ones(h_d, w_d, n_per_block, device=device, dtype=torch.bool)
    valid_tokens.view(h_d * w_d, n_per_block)[is_low_res_block_flat, 1:] = False

    valid_tokens_flat = valid_tokens.view(-1)
    img_fov = output_blocks.view(B, -1, C)[:, valid_tokens_flat, :]

    # Calculate resolution masks for CRPA Attention
    resolution_mask_grid = torch.ones(H, W, device=device)
    res_mask_blocks = resolution_mask_grid.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
    output_res_mask = res_mask_blocks.clone()
    output_res_mask[is_low_res_block, :] = 0.0

    tl_mask_blocks = torch.zeros_like(res_mask_blocks)
    tl_mask_blocks[..., 0] = 1.0
    tl_mask_blocks[is_low_res_block, :] = 1.0

    resolution_mask = output_res_mask.view(-1)[valid_tokens_flat]
    resolution_mask_top_left = tl_mask_blocks.view(-1)[valid_tokens_flat]

    fov_indices = {
        "valid_tokens": valid_tokens,
        "is_low_res_block": is_low_res_block,
        "resolution_mask": resolution_mask,
        "resolution_mask_top_left": resolution_mask_top_left,
        "H": H,
        "W": W,
        "lr_factor": lr_factor,
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
    Build img_ids matching the valid tokens exactly.
    """
    H, W, n_axes = img_ids.shape
    h_d, w_d = H // lr_factor, W // lr_factor
    n_per_block = lr_factor * lr_factor

    grid_2d = img_ids.view(H, W, n_axes)
    grid_blocks = grid_2d.view(h_d, lr_factor, w_d, lr_factor, n_axes).permute(0, 2, 1, 3, 4).reshape(h_d, w_d, n_per_block, n_axes)

    mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
    is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
    is_low_res_block = ~is_high_res_block

    output_grid = grid_blocks.clone()
    # For LR blocks, ensure coordinates of the first token represent the top-left of the block
    low_res_coords = grid_blocks[..., 0, :]
    output_grid.view(h_d * w_d, n_per_block, n_axes)[is_low_res_block.view(-1), 0, :] = low_res_coords[is_low_res_block]

    valid_tokens = torch.ones(h_d, w_d, n_per_block, device=device, dtype=torch.bool)
    valid_tokens.view(h_d * w_d, n_per_block)[is_low_res_block.view(-1), 1:] = False

    output_grid = output_grid.view(-1, n_axes)
    valid_tokens_flat = valid_tokens.view(-1)

    return output_grid[valid_tokens_flat].to(device=device, dtype=dtype)


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
    Upsamples/reconstructs the foveated token representation back to full size.
    """
    if device is None:
        device = tokens.device
    if dtype is None:
        dtype = tokens.dtype

    valid_tokens = fov_indices["valid_tokens"]
    is_low_res_block = fov_indices["is_low_res_block"]
    h_d, w_d = H // lr_factor, W // lr_factor
    n_per_block = lr_factor * lr_factor

    # 1. Scatter the reduced sequence back into full grid shape
    reconstructed = torch.zeros(B, h_d * w_d * n_per_block, C, device=device, dtype=dtype)
    valid_tokens_flat = valid_tokens.view(-1)
    reconstructed[:, valid_tokens_flat, :] = tokens

    # 2. For low-res blocks, copy top-left token (index 0) to all block locations (nearest neighbor)
    reconstructed = reconstructed.view(B, h_d * w_d, n_per_block, C)
    is_low_res_block_flat = is_low_res_block.view(-1)
    
    # Broadcast index 0 values to index 0:4
    low_res_vals = reconstructed[:, is_low_res_block_flat, 0:1, :]
    reconstructed[:, is_low_res_block_flat, :, :] = low_res_vals.expand(-1, -1, n_per_block, -1)

    # 3. Permute back to spatial representation
    reconstructed = reconstructed.view(B, h_d, w_d, lr_factor, lr_factor, C)
    reconstructed = reconstructed.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, C)

    return reconstructed


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
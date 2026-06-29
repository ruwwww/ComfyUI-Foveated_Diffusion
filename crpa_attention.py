"""
CRPA (Cross-Resolution Phase-Aligned) Attention for FLUX2 foveated generation.
Matches the official diffsynth_fov implementation.
"""

import torch
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention


def build_crpa_state(
    *,
    img_ids_fov: torch.Tensor,       # (B, L_fov, n_axes)
    txt_ids: torch.Tensor,           # (B, T, n_axes)
    resolution_mask: torch.Tensor,   # (L_fov,) 1 for HR, 0 for LR
    resolution_mask_top_left: torch.Tensor, # (L_fov,) 1 for TL/LR, 0 otherwise
    lr_factor: int,
    pe_embedder,                     # model.pe_embedder
):
    B, L_fov, n_axes = img_ids_fov.shape
    T = txt_ids.shape[1]
    device = img_ids_fov.device

    # ── Dual position-ID sequences (keeping Batch Dimension B) ─────
    ids_hr = torch.cat([txt_ids, img_ids_fov], dim=1)                # (B, T+L_fov, n_axes)

    img_ids_lr = img_ids_fov.clone()
    
    import logging
    logger = logging.getLogger("comfyui_foveated_diffusion")
    logger.info(
        "CRPA Debug: img_ids_fov shape=%s, first 3 tokens: %s",
        tuple(img_ids_fov.shape), img_ids_fov[0, :3].tolist()
    )

    img_ids_lr[:, :, 1] = img_ids_lr[:, :, 1] / float(lr_factor)    # H / d
    img_ids_lr[:, :, 2] = img_ids_lr[:, :, 2] / float(lr_factor)    # W / d
    
    logger.info(
        "CRPA Debug: img_ids_lr divided coordinates: first 3 tokens: %s",
        img_ids_lr[0, :3].tolist()
    )

    ids_lr = torch.cat([txt_ids, img_ids_lr], dim=1)                # (B, T+L_fov, n_axes)

    pe_hr = pe_embedder(ids_hr)   # (B, 1, T+L_fov, pe_dim//2, 2, 2)
    pe_lr = pe_embedder(ids_lr)   # (B, 1, T+L_fov, pe_dim//2, 2, 2)

    # ── Resolution masks (including text) ──────────────────────────
    txt_mask = torch.ones(T, dtype=torch.bool, device=device)
    
    # res_mask: True for text and HR image tokens
    res_mask = torch.cat([txt_mask, resolution_mask.to(device=device, dtype=torch.bool)])
    
    # tl_mask: True for text and top-left/LR image tokens
    tl_mask = torch.cat([txt_mask, resolution_mask_top_left.to(device=device, dtype=torch.bool)])

    hr_idx = torch.where(res_mask)[0]
    lr_idx = torch.where(~res_mask)[0]
    tl_idx = torch.where(tl_mask)[0]

    return {
        "pe_hr": pe_hr,          # (B, 1, T+L_fov, pe_dim//2, 2, 2)
        "pe_lr": pe_lr,          # (B, 1, T+L_fov, pe_dim//2, 2, 2)
        "hr_idx": hr_idx,
        "lr_idx": lr_idx,
        "tl_idx": tl_idx,
    }


def crpa_attn1_patch(q, k, v, pe, attn_mask, extra_options):
    crpa = extra_options.get("crpa_state")
    if crpa is None:
        return {"q": q, "k": k, "v": v, "pe": pe}

    heads = q.shape[1]
    head_dim = q.shape[-1]
    B = q.shape[0]
    N = q.shape[2]
    pe_hr, pe_lr = crpa["pe_hr"], crpa["pe_lr"]
    hr_idx, lr_idx, tl_idx = crpa["hr_idx"], crpa["lr_idx"], crpa["tl_idx"]

    # Output accumulator: (B, N, heads*head_dim)
    out = torch.zeros(B, N, heads * head_dim, dtype=q.dtype, device=q.device)

    # ── HR path: Q_HR + HR RoPE  →  attend to all K + HR RoPE ───
    if hr_idx.numel() > 0:
        q_hr = q[:, :, hr_idx, :].contiguous()
        pe_hr_sliced = pe_hr[:, :, hr_idx].contiguous()
        q_hr_r = apply_rope1(q_hr, pe_hr_sliced)
        k_hr_r = apply_rope1(k, pe_hr)
        out_hr = optimized_attention(q_hr_r, k_hr_r, v, heads, skip_reshape=True)
        out[:, hr_idx, :] = out_hr

    # ── LR path: Q_LR + LR RoPE  →  attend to subsampled K + LR RoPE ───
    if lr_idx.numel() > 0:
        q_lr = q[:, :, lr_idx, :].contiguous()
        pe_lr_sliced = pe_lr[:, :, lr_idx].contiguous()
        q_lr_r = apply_rope1(q_lr, pe_lr_sliced)

        k_sub = k[:, :, tl_idx, :].contiguous()
        v_sub = v[:, :, tl_idx, :].contiguous()
        pe_lr_tl_sliced = pe_lr[:, :, tl_idx].contiguous()
        k_lr_r = apply_rope1(k_sub, pe_lr_tl_sliced)

        out_lr = optimized_attention(q_lr_r, k_lr_r, v_sub, heads, skip_reshape=True)
        out[:, lr_idx, :] = out_lr

    extra_options["__crpa_out__"] = out

    # Replace Q with zeros so the standard attention call is a fast no-op.
    zero_q = torch.zeros(B, heads, N, head_dim, dtype=q.dtype, device=q.device)
    return {"q": zero_q, "k": k, "v": v, "pe": pe}


def crpa_attn1_output_patch(attn, extra_options):
    return extra_options.pop("__crpa_out__", attn)
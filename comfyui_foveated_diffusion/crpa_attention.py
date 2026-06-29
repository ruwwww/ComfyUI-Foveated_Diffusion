"""
CRPA (Cross-Resolution Phase-Aligned) Attention for FLUX2 foveated generation.

Implements dual-RoPE attention with key subsampling as described in
arXiv 2603.23491.  Injected as attn1_patch / attn1_output_patch into
ComfyUI's Flux2 DoubleStreamBlock and SingleStreamBlock.
"""

import torch
from comfy.ldm.flux.math import _apply_rope1
from comfy.ldm.modules.attention import optimized_attention


# ── CRPA State Computed by the DIFFUSION_MODEL Wrapper ─────────────────

def build_crpa_state(
    *,
    img_ids_fov: torch.Tensor,       # (B, L_fov, n_axes)
    txt_ids: torch.Tensor,           # (B, T, n_axes)
    hr_count: int,                   # first m tokens = HR
    hr_indices: torch.Tensor,        # (m,) long  flat indices in (H*W) space
    lr_factor: int,
    W: int,                          # token grid width
    pe_embedder,                     # model.pe_embedder
):
    B, L_fov, n_axes = img_ids_fov.shape
    T = txt_ids.shape[1]
    device = img_ids_fov.device

    # ── Dual position-ID sequences ─────────────────────────────────
    ids_hr = torch.cat([txt_ids[0], img_ids_fov[0]], dim=0)          # (T+L_fov, n_axes)

    img_ids_lr = img_ids_fov[0].clone()
    img_ids_lr[:, 1] = img_ids_lr[:, 1] / float(lr_factor)          # H / d
    img_ids_lr[:, 2] = img_ids_lr[:, 2] / float(lr_factor)          # W / d
    ids_lr = torch.cat([txt_ids[0], img_ids_lr], dim=0)              # (T+L_fov, n_axes)

    pe_hr = pe_embedder(ids_hr)   # (1, 1, T+L_fov, pe_dim)
    pe_lr = pe_embedder(ids_lr)

    # ── Resolution masks ───────────────────────────────────────────
    # Image resolution mask: first m tokens = HR
    img_res = torch.zeros(L_fov, dtype=torch.bool, device=device)
    img_res[:hr_count] = True
    txt_mask = torch.ones(T, dtype=torch.bool, device=device)
    res_mask = torch.cat([txt_mask, img_res])                       # (T+L_fov,)

    # Top-left key-subsampling mask
    img_tl = torch.zeros(L_fov, dtype=torch.bool, device=device)
    # HR tokens: keep only top-left of each d × d block
    hr_flat = hr_indices.to(device)
    hr_row, hr_col = hr_flat // W, hr_flat % W
    hr_is_tl = (hr_row % lr_factor == 0) & (hr_col % lr_factor == 0)
    img_tl[:hr_count] = hr_is_tl
    # LR tokens: all are single-per-block → all are TL
    img_tl[hr_count:] = True
    tl_mask = torch.cat([txt_mask, img_tl])                         # (T+L_fov,)

    # Pre-compute index tensors for the attention patch
    hr_idx = torch.where(res_mask)[0]
    lr_idx = torch.where(~res_mask)[0]
    tl_idx = torch.where(tl_mask)[0]

    return {
        "pe_hr": pe_hr,          # (1, 1, T+L_fov, 64, 2, 2) — HR RoPE
        "pe_lr": pe_lr,          # (1, 1, T+L_fov, 64, 2, 2) — LR RoPE
        "hr_idx": hr_idx,        # (num_hr,)   int indices
        "lr_idx": lr_idx,        # (num_lr,)   int indices
        "tl_idx": tl_idx,        # (num_tl,)   int indices for LR key subsample
    }


# ── Attention Patches (injected as attn1_patch / attn1_output_patch) ──

def crpa_attn1_patch(q, k, v, pe, attn_mask, extra_options):
    crpa = extra_options.get("crpa_state")
    if crpa is None:
        return {"q": q, "k": k, "v": v, "pe": pe}

    heads = q.shape[1]
    head_dim = q.shape[-1]
    B = q.shape[0]
    pe_hr, pe_lr = crpa["pe_hr"], crpa["pe_lr"]
    hr_idx, lr_idx, tl_idx = crpa["hr_idx"], crpa["lr_idx"], crpa["tl_idx"]
    out = torch.zeros_like(q)

    # ── HR path: Q_HR + HR RoPE  →  attend to all K + HR RoPE ───
    if hr_idx.numel() > 0:
        q_hr = q[:, :, hr_idx, :]
        q_hr_r = _apply_rope1(q_hr, pe_hr[:, :, hr_idx])
        k_hr_r = _apply_rope1(k, pe_hr)
        out_hr = optimized_attention(q_hr_r, k_hr_r, v, heads, skip_reshape=True)
        out[:, :, hr_idx, :] = out_hr

    # ── LR path: Q_LR + LR RoPE  →  attend to subsampled K + LR RoPE ───
    if lr_idx.numel() > 0:
        q_lr = q[:, :, lr_idx, :]
        q_lr_r = _apply_rope1(q_lr, pe_lr[:, :, lr_idx])

        k_sub = k[:, :, tl_idx, :]
        v_sub = v[:, :, tl_idx, :]
        k_lr_r = _apply_rope1(k_sub, pe_lr[:, :, tl_idx])

        out_lr = optimized_attention(q_lr_r, k_lr_r, v_sub, heads, skip_reshape=True)
        out[:, :, lr_idx, :] = out_lr

    extra_options["__crpa_out__"] = out

    # Replace Q with zeros so the standard attention call is a fast no-op.
    # The real attention output is in __crpa_out__ and will be substituted
    # by crpa_attn1_output_patch.
    N = q.shape[2]
    zero_q = torch.zeros(B, heads, N, head_dim, dtype=q.dtype, device=q.device)
    return {"q": zero_q, "k": k, "v": v, "pe": pe}


def crpa_attn1_output_patch(attn, extra_options):
    return extra_options.pop("__crpa_out__", attn)
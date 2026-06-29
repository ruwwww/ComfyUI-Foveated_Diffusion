"""FoveatedDiffusion — FLUX.2 Klein foveated generation."""

from .nodes import (
    FoveationMaskNode,
    LoadFoveatedLoRA,
    FoveatedKSampler,
    FoveatedVAEDecode,
    FoveationMaskPreview,
)

NODE_CLASS_MAPPINGS = {
    "FoveationMask":          FoveationMaskNode,
    "LoadFoveatedLoRA":       LoadFoveatedLoRA,
    "FoveatedKSampler":       FoveatedKSampler,
    "FoveatedVAEDecode":      FoveatedVAEDecode,
    "FoveationMaskPreview":   FoveationMaskPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoveationMask":          "Foveation Mask (FovDiff)",
    "LoadFoveatedLoRA":       "Load Foveated LoRA (FovDiff)",
    "FoveatedKSampler":       "Foveated KSampler (FovDiff)",
    "FoveatedVAEDecode":      "Foveated VAE Decode (FovDiff)",
    "FoveationMaskPreview":   "Foveation Mask Preview (FovDiff)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
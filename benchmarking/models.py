"""Model definitions for GeoClipItaly fine-tuned checkpoint."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class GeoClipItaly(nn.Module):
    """GeoCLIP + LoRA + regression head (matches GeoClip_Training_V3_3.ipynb)."""

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        from geoclip.model import GeoCLIP
        from peft import LoraConfig, TaskType, get_peft_model

        if lora_alpha is None:
            lora_alpha = lora_rank * 2
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

        self.geoclip = GeoCLIP(from_pretrained=True)
        for param in self.geoclip.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.geoclip.image_encoder.CLIP.vision_model = get_peft_model(
            self.geoclip.image_encoder.CLIP.vision_model,
            lora_config,
        )
        for p in self.geoclip.image_encoder.CLIP.visual_projection.parameters():
            p.requires_grad = True
        for p in self.geoclip.image_encoder.mlp.parameters():
            p.requires_grad = True
        self.geoclip.eval()

        self.regressor = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 2), nn.Sigmoid(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.geoclip.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vm_base = self.geoclip.image_encoder.CLIP.vision_model.base_model
        out = vm_base(pixel_values=x)
        pooled = (
            out.pooler_output
            if (hasattr(out, "pooler_output") and out.pooler_output is not None)
            else out.last_hidden_state[:, 0, :]
        )
        proj = self.geoclip.image_encoder.CLIP.visual_projection(pooled)
        feats = self.geoclip.image_encoder.mlp(proj)
        return self.regressor(feats)


def patch_geoclip_image_encoder(model) -> None:
    """Compatibility patch for newer transformers returning pooled outputs."""

    def _patched_forward(x):
        enc = model.image_encoder
        raw = enc.CLIP.vision_model(pixel_values=x)
        pooled = (
            raw.pooler_output
            if (hasattr(raw, "pooler_output") and raw.pooler_output is not None)
            else raw.last_hidden_state[:, 0, :]
        )
        projected = enc.CLIP.visual_projection(pooled)
        return enc.mlp(projected)

    model.image_encoder.forward = _patched_forward


def load_finetuned_checkpoint(
    model: GeoClipItaly,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return checkpoint if isinstance(checkpoint, dict) else {}

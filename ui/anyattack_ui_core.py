from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from models.model import CLIPEncoder, Decoder


_DEFAULT_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def _pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_bytes_to_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    img = _pil_from_bytes(data)
    return _DEFAULT_IMAGE_TRANSFORM(img).unsqueeze(0).to(device)


@dataclass(frozen=True)
class AnyAttackModels:
    clip: CLIPEncoder
    decoder: Decoder
    device: torch.device
    model_name: str
    decoder_path: str


def load_models(
    model_name: str,
    decoder_path: str,
    device: torch.device,
    *,
    embed_dim: int = 512,
) -> AnyAttackModels:
    clip = CLIPEncoder(model_name).to(device).eval()

    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    state = torch.load(decoder_path, map_location="cpu")
    state_dict = state["decoder_state_dict"] if isinstance(state, dict) and "decoder_state_dict" in state else state

    try:
        decoder.load_state_dict(state_dict)
    except Exception:
        # tolerate DataParallel checkpoints (keys prefixed by "module.")
        new_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            new_state[new_k] = v
        decoder.load_state_dict(new_state)

    return AnyAttackModels(
        clip=clip,
        decoder=decoder,
        device=device,
        model_name=model_name,
        decoder_path=decoder_path,
    )


@torch.no_grad()
def generate_adv_tensor(
    models: AnyAttackModels,
    clean_img: torch.Tensor,
    target_img: torch.Tensor,
    *,
    eps: float = 16.0 / 255.0,
) -> torch.Tensor:
    img_emb = models.clip.encode_img(target_img)
    origin_noise = models.decoder(img_emb)
    noise = torch.clamp(origin_noise, -eps, eps)
    adv = torch.clamp(clean_img + noise, 0, 1)
    return adv


def save_tensor_image(img: torch.Tensor, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(img, str(path))


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    if not name:
        return "image"
    return name


def ensure_dir(p: str | os.PathLike) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)


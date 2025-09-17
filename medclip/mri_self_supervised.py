"""Self-supervised pretraining utilities for 3D MRI volumes.

This module implements a light-weight 3D masked autoencoder (MAE) and a
SimCLR-style contrastive learner that can be trained on brain MRI scans
without requiring paired text labels.  The encoder produced by either
pretraining strategy can be exported and reused for downstream fine-tuning
tasks such as disease classification or genotype prediction.

The implementation focuses on NIfTI (``.nii``/``.nii.gz``) volumes with a
single intensity channel.  Volumes are resized to a fixed ``volume_size`` so
that the number of patches is consistent across the dataset.  Basic 3D data
augmentations (cropping, flipping, rotation, Gaussian noise and intensity
scaling) are provided to support contrastive learning.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _list_nifti_files(data_source: str) -> List[str]:
    """Return a sorted list of NIfTI files from a directory or manifest.

    Args:
        data_source: Either a directory that contains ``.nii`` / ``.nii.gz``
            files (recursively) or a text/CSV file with one absolute file path
            per line.
    """

    if os.path.isdir(data_source):
        result: List[str] = []
        for root, _, files in os.walk(data_source):
            for filename in files:
                if filename.endswith((".nii", ".nii.gz")):
                    result.append(os.path.join(root, filename))
        return sorted(result)

    if not os.path.exists(data_source):
        raise FileNotFoundError(f"Could not locate data source: {data_source}")

    # treat as a manifest file
    result = []
    with open(data_source, "r", encoding="utf-8") as handle:
        for line in handle:
            path = line.strip()
            if path:
                result.append(path)
    return sorted(result)


def _resize_volume(volume: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
    """Resize a 3D volume to ``size`` using trilinear interpolation."""

    tensor = torch.from_numpy(volume[None, None, ...])
    resized = F.interpolate(
        tensor.float(), size=size, mode="trilinear", align_corners=False
    )
    return resized[0, 0].cpu().numpy()


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume while guarding against zero variance."""

    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        std = 1.0
    volume = (volume - mean) / std
    return volume.astype(np.float32)


def _random_crop(volume: np.ndarray, min_scale: float = 0.6) -> np.ndarray:
    """Apply a random crop with side length sampled from ``[min_scale, 1]``."""

    assert 0 < min_scale <= 1, "min_scale should be within (0, 1]."
    depth, height, width = volume.shape
    scale = random.uniform(min_scale, 1.0)
    crop_d = max(1, int(depth * scale))
    crop_h = max(1, int(height * scale))
    crop_w = max(1, int(width * scale))

    start_d = random.randint(0, depth - crop_d)
    start_h = random.randint(0, height - crop_h)
    start_w = random.randint(0, width - crop_w)

    return volume[
        start_d : start_d + crop_d,
        start_h : start_h + crop_h,
        start_w : start_w + crop_w,
    ]


def _random_flip(volume: np.ndarray) -> np.ndarray:
    for axis in range(3):
        if random.random() < 0.5:
            volume = np.flip(volume, axis=axis)
    return volume


def _random_rotate_90(volume: np.ndarray) -> np.ndarray:
    """Random 90-degree rotation around a random pair of axes."""

    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    k = random.randint(0, 3)
    if k:
        volume = np.rot90(volume, k=k, axes=axes)
    return volume


def _apply_gaussian_noise(volume: np.ndarray, max_std: float) -> np.ndarray:
    if max_std <= 0:
        return volume
    noise_std = random.uniform(0.0, max_std)
    if noise_std > 0:
        volume = volume + np.random.normal(0.0, noise_std, size=volume.shape)
    return volume


def _apply_intensity_shift(
    volume: np.ndarray, max_scale: float, max_shift: float
) -> np.ndarray:
    if max_scale <= 0 and max_shift <= 0:
        return volume
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    shift = random.uniform(-max_shift, max_shift)
    return volume * scale + shift


class RandomMRITransform:
    """Random data augmentation for 3D MRI volumes."""

    def __init__(
        self,
        target_size: Tuple[int, int, int],
        min_crop_scale: float = 0.6,
        rotation_prob: float = 0.5,
        gaussian_noise_std: float = 0.1,
        intensity_scale: float = 0.1,
        intensity_shift: float = 0.1,
    ) -> None:
        self.target_size = target_size
        self.min_crop_scale = min_crop_scale
        self.rotation_prob = rotation_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        augmented = _random_crop(volume, self.min_crop_scale)
        augmented = _random_flip(augmented)
        if random.random() < self.rotation_prob:
            augmented = _random_rotate_90(augmented)
        augmented = _apply_gaussian_noise(augmented, self.gaussian_noise_std)
        augmented = _apply_intensity_shift(
            augmented, self.intensity_scale, self.intensity_shift
        )
        augmented = _resize_volume(augmented, self.target_size)
        augmented = _normalize_volume(augmented)
        return augmented


class MRI3DDataset(Dataset):
    """Dataset that loads NIfTI volumes for self-supervised pretraining."""

    def __init__(
        self,
        data_source: str,
        volume_size: Tuple[int, int, int] = (96, 96, 96),
        contrastive_views: int = 0,
        transform: Optional[RandomMRITransform] = None,
    ) -> None:
        super().__init__()
        self.file_paths = _list_nifti_files(data_source)
        if not self.file_paths:
            raise ValueError(
                "No NIfTI files were found. Please ensure `data_source` points "
                "to a directory containing `.nii`/`.nii.gz` files or a text "
                "manifest with absolute file paths."
            )
        self.volume_size = volume_size
        self.contrastive_views = contrastive_views
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_volume(self, index: int) -> np.ndarray:
        filepath = self.file_paths[index]
        array = np.asarray(nib.load(filepath).get_fdata())
        array = array.astype(np.float32)
        array = _resize_volume(array, self.volume_size)
        array = _normalize_volume(array)
        return array

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        volume = self._load_volume(index)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)
        sample: Dict[str, torch.Tensor] = {"volume": volume_tensor}

        if self.contrastive_views > 0:
            if self.transform is None:
                raise ValueError(
                    "A transform must be provided when `contrastive_views` > 0."
                )
            views: List[torch.Tensor] = []
            for _ in range(self.contrastive_views):
                augmented = self.transform(volume.copy())
                views.append(torch.from_numpy(augmented).unsqueeze(0))
            sample["views"] = torch.stack(views, dim=0)

        return sample


def _build_learnable_pos_embed(num_tokens: int, embed_dim: int) -> nn.Parameter:
    pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    nn.init.trunc_normal_(pos_embed, std=0.02)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """3D patch embedding implemented with a convolutional layer."""

    def __init__(
        self,
        volume_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if any(s % p != 0 for s, p in zip(volume_size, patch_size)):
            raise ValueError(
                "`volume_size` must be divisible by `patch_size` along each axis."
            )
        self.patch_size = patch_size
        self.grid_size = tuple(s // p for s, p in zip(volume_size, patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers with ``batch_first`` layout."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class MaskedAutoencoder3D(nn.Module):
    """Masked autoencoder (MAE) for self-supervised MRI pretraining."""

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_chans, embed_dim)
        self.pos_embed = _build_learnable_pos_embed(
            self.patch_embed.num_patches, embed_dim
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = _build_learnable_pos_embed(
            self.patch_embed.num_patches, decoder_embed_dim
        )
        self.decoder = TransformerEncoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
        )
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_volume)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            nn.init.zeros_(self.decoder_pred.bias)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by per-sample shuffling."""

        batch_size, num_patches, _ = x.shape
        len_keep = int(num_patches * (1 - mask_ratio))

        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)
        batch_size, len_keep, dim = x.shape
        num_patches = self.patch_embed.num_patches
        mask_tokens = self.mask_token.repeat(batch_size, num_patches - len_keep, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, dim),
        )
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x)
        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        assert imgs.shape[2] == self.volume_size[0]
        assert imgs.shape[3] == self.volume_size[1]
        assert imgs.shape[4] == self.volume_size[2]

        B, C, D, H, W = imgs.shape
        d = D // p[0]
        h = H // p[1]
        w = W // p[2]
        x = imgs.reshape(B, C, d, p[0], h, p[1], w, p[2])
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, d * h * w, C * p[0] * p[1] * p[2])
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, N, L = x.shape
        C = L // (p[0] * p[1] * p[2])
        d = self.volume_size[0] // p[0]
        h = self.volume_size[1] // p[1]
        w = self.volume_size[2] // p[2]
        x = x.reshape(B, d, h, w, C, p[0], p[1], p[2])
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(
            B, C, self.volume_size[0], self.volume_size[1], self.volume_size[2]
        )
        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask}

    def reconstruct(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        result = self.forward(imgs, mask_ratio)
        volume = self.unpatchify(result["pred"])
        return volume

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dict containing only the encoder components."""

        state = {}
        for key, value in self.state_dict().items():
            if key.startswith("patch_embed") or key.startswith("encoder") or key == "pos_embed":
                state[key] = value
        return state


class ContrastiveLearner3D(nn.Module):
    """SimCLR-style contrastive learner for 3D MRI volumes."""

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = _build_learnable_pos_embed(
            self.patch_embed.num_patches + 1, embed_dim
        )
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        hidden_dim = embed_dim * 2
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.projection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x[:, 0]

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_a = self.forward_features(view_a)
        feature_b = self.forward_features(view_b)
        proj_a = F.normalize(self.projection_head(feature_a), dim=-1)
        proj_b = F.normalize(self.projection_head(feature_b), dim=-1)

        logits = proj_a @ proj_b.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = (loss_a + loss_b) * 0.5
        return {"loss": loss, "logits": logits, "proj_a": proj_a, "proj_b": proj_b}

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if key.startswith("patch_embed") or key.startswith("encoder") or key in {"cls_token", "pos_embed"}:
                state[key] = value
        return state


@dataclass
class SelfSupervisedConfig:
    """Configuration for MRI self-supervised pretraining."""

    data_source: str
    output_dir: str
    mode: str = "mae"  # either "mae" or "contrastive"
    volume_size: Tuple[int, int, int] = (96, 96, 96)
    patch_size: Tuple[int, int, int] = (16, 16, 16)
    batch_size: int = 2
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    mask_ratio_range: Tuple[float, float] = (0.5, 0.7)
    log_every: int = 10
    temperature: float = 0.07
    projection_dim: int = 128
    num_workers: int = 4

    def sample_mask_ratio(self) -> float:
        low, high = self.mask_ratio_range
        return random.uniform(low, high)


class SelfSupervisedPretrainer:
    """Utility class that orchestrates the self-supervised training loop."""

    def __init__(self, config: SelfSupervisedConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

        if config.mode.lower() == "mae":
            self.model: nn.Module = MaskedAutoencoder3D(
                volume_size=config.volume_size,
                patch_size=config.patch_size,
            )
            self.dataset = MRI3DDataset(
                config.data_source,
                volume_size=config.volume_size,
                contrastive_views=0,
                transform=None,
            )
        elif config.mode.lower() == "contrastive":
            self.model = ContrastiveLearner3D(
                volume_size=config.volume_size,
                patch_size=config.patch_size,
                temperature=config.temperature,
                projection_dim=config.projection_dim,
            )
            transform = RandomMRITransform(config.volume_size)
            self.dataset = MRI3DDataset(
                config.data_source,
                volume_size=config.volume_size,
                contrastive_views=2,
                transform=transform,
            )
        else:
            raise ValueError("mode must be either 'mae' or 'contrastive'")

    def dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def train(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        loader = self.dataloader()
        global_step = 0
        for epoch in range(self.config.epochs):
            for batch in loader:
                optimizer.zero_grad()
                if isinstance(self.model, MaskedAutoencoder3D):
                    inputs = batch["volume"].to(device)
                    loss_dict = self.model(inputs, mask_ratio=self.config.sample_mask_ratio())
                else:
                    views = batch["views"].to(device)
                    loss_dict = self.model(views[:, 0], views[:, 1])
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()

                if global_step % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config.epochs} | Step {global_step} "
                        f"| Loss: {loss.item():.4f}"
                    )
                global_step += 1

        # Save both the full model and the encoder for downstream fine-tuning.
        model_path = os.path.join(self.config.output_dir, "self_supervised_model.pt")
        torch.save(self.model.state_dict(), model_path)

        encoder_path = os.path.join(self.config.output_dir, "encoder.pt")
        torch.save(self.model.encoder_state_dict(), encoder_path)
        print(
            f"Saved full model to {model_path} and encoder weights to {encoder_path}."
            "\nLoad the encoder weights into your downstream model with "
            "`load_state_dict` for fine-tuning."
        )


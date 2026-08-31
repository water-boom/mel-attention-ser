"""Dataset and Audio Loading with In-Fold Normalization & SpecAugment."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .frontend import AudioFrontend
from .split import parse_ravdess_filename


def load_audio_file(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file as mono 1D float32 tensor, resampled to target_sr if needed."""
    wav, sr = sf.read(file_path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if sr != target_sr:
        # High quality polyphase antialiasing resample
        gcd = math_gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = scipy.signal.resample_poly(wav, up, down).astype(np.float32)

    return torch.from_numpy(wav)


def math_gcd(a: int, b: int) -> int:
    """Helper for integer greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


class SpecAugment:
    """Time and Frequency Masking for Mel-spectrograms (Train only)."""

    def __init__(self, freq_mask_max: int = 16, time_mask_max: int = 24, num_masks: int = 2):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_masks = num_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masking on (3, F, T) tensor."""
        x = x.clone()
        _, f_bins, t_frames = x.shape

        for _ in range(self.num_masks):
            # Frequency masking
            f_len = torch.randint(0, self.freq_mask_max + 1, (1,)).item()
            if f_len > 0 and f_bins > f_len:
                f0 = torch.randint(0, f_bins - f_len, (1,)).item()
                x[:, f0 : f0 + f_len, :] = 0.0

            # Time masking
            t_len = torch.randint(0, self.time_mask_max + 1, (1,)).item()
            if t_len > 0 and t_frames > t_len:
                t0 = torch.randint(0, t_frames - t_len, (1,)).item()
                x[:, :, t0 : t0 + t_len] = 0.0

        return x


class SERDataset(Dataset):
    """Speech Emotion Recognition Dataset with In-Fold Standardization."""

    def __init__(
        self,
        file_paths: List[str],
        frontend: Optional[AudioFrontend] = None,
        cache_dir: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        augment: bool = False,
        target_sr: int = 16000,
    ):
        self.file_paths = file_paths
        self.frontend = frontend or AudioFrontend(sample_rate=target_sr)
        self.cache_dir = cache_dir
        self.mean = mean
        self.std = std
        self.augment = augment
        self.spec_aug = SpecAugment() if augment else None
        self.target_sr = target_sr

        # Pre-parse labels and actors
        self.samples = []
        for p in file_paths:
            try:
                label, actor = parse_ravdess_filename(p)
                self.samples.append((p, label, actor))
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, file_path: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        stem = Path(file_path).stem
        return os.path.join(self.cache_dir, f"{stem}.npy")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        file_path, label, actor = self.samples[idx]
        cache_path = self._get_cache_path(file_path)

        if cache_path and os.path.exists(cache_path):
            feat = torch.from_numpy(np.load(cache_path)).float()
        else:
            wav = load_audio_file(file_path, self.target_sr)
            with torch.no_grad():
                feat = self.frontend(wav).squeeze(0)  # (3, 128, 300)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, feat.cpu().numpy())

        # In-fold standardization
        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / (self.std + 1e-6)

        # SpecAugment (training only)
        if self.augment and self.spec_aug is not None:
            feat = self.spec_aug(feat)

        return feat, label, actor

    @staticmethod
    def compute_fold_stats(
        file_paths: List[str],
        frontend: AudioFrontend,
        cache_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std per mel-frequency bin over all training fold items."""
        accum_sum = torch.zeros(3, 128, 1)
        accum_sq = torch.zeros(3, 128, 1)
        total_frames = 0

        for p in file_paths:
            cache_path = os.path.join(cache_dir, f"{Path(p).stem}.npy") if cache_dir else None
            if cache_path and os.path.exists(cache_path):
                feat = torch.from_numpy(np.load(cache_path)).float()
            else:
                wav = load_audio_file(p, frontend.sample_rate)
                with torch.no_grad():
                    feat = frontend(wav).squeeze(0)

            # Sum over time dimension (dim=2)
            accum_sum += feat.sum(dim=2, keepdim=True)
            accum_sq += (feat ** 2).sum(dim=2, keepdim=True)
            total_frames += feat.shape[2]

        mean = accum_sum / total_frames
        var = (accum_sq / total_frames) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return mean, std

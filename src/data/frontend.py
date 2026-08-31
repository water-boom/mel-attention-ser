"""Pure PyTorch Audio Frontend: STFT, Mel-filterbank, Log-compression, Time-normalization, Delta channels.

Zero external torchaudio C++ dependency for maximum cross-platform reliability.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    norm: str = "slaney",
) -> torch.Tensor:
    """Create Slaney-style triangular Mel filterbank matrix in PyTorch.

    Returns:
        Tensor of shape (n_mels, n_fft // 2 + 1)
    """
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            if center != left and j < n_freqs:
                weights[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center and j < n_freqs:
                weights[i - 1, j] = (right - j) / (right - center)

        if norm == "slaney":
            enorm = 2.0 / (hz_points[i + 1] - hz_points[i - 1])
            weights[i - 1, :] *= enorm

    return torch.from_numpy(weights)


def compute_delta(features: torch.Tensor, order: int = 1, width: int = 5) -> torch.Tensor:
    """Compute Delta (1st derivative) or Delta-Delta (2nd derivative) along time axis.

    Args:
        features: Tensor of shape (..., T)
        order: 1 for delta, 2 for delta-delta
        width: odd filter width (default 5)
    Returns:
        Tensor of shape (..., T)
    """
    if order == 0:
        return features

    half_width = width // 2
    weights = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=features.device)
    denom = 2.0 * sum(i**2 for i in range(1, half_width + 1))
    kernel = (weights / denom).view(1, 1, -1)

    # Pad time dimension (last dim)
    orig_shape = features.shape
    x = features.reshape(-1, 1, orig_shape[-1])
    padded = F.pad(x, (half_width, half_width), mode="replicate")
    delta1 = F.conv1d(padded, kernel)

    if order == 1:
        return delta1.reshape(orig_shape)
    elif order == 2:
        padded2 = F.pad(delta1, (half_width, half_width), mode="replicate")
        delta2 = F.conv1d(padded2, kernel)
        return delta2.reshape(orig_shape)
    else:
        raise ValueError(f"Unsupported delta order: {order}")


class AudioFrontend(nn.Module):
    """Pure PyTorch Frontend for Log-Mel Spectrogram + Delta Feature Extraction.

    Outputs (B, 3, n_mels, target_frames):
        Channel 0: Static Log-Mel Energy
        Channel 1: Delta (Velocity)
        Channel 2: Delta-Delta (Acceleration)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        target_frames: int = 300,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.eps = eps

        # Register window and filterbank as buffers
        self.register_buffer("window", torch.hann_window(win_length))
        mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max, norm="slaney")
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: Tensor of shape (B, T_raw) or (T_raw,)
        Returns:
            Tensor of shape (B, 3, n_mels, target_frames)
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        device = waveform.device
        window = self.window.to(device)
        mel_fb = self.mel_fb.to(device)

        # 1. Short-Time Fourier Transform (STFT)
        # return_complex=True -> (B, n_freqs, T_frames)
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # 2. Power Spectrogram: |X(f, t)|^2
        power_spec = stft_complex.abs().pow(2)  # (B, n_freqs, T_frames)

        # 3. Mel Filterbank Application: mel_fb @ power_spec
        # mel_fb is (n_mels, n_freqs), power_spec is (B, n_freqs, T_frames)
        mel_spec = torch.matmul(mel_fb, power_spec)  # (B, n_mels, T_frames)

        # 4. Log Compression: Log(mel_power + eps)
        log_mel = torch.log(mel_spec + self.eps)  # (B, n_mels, T_frames)

        # 5. Duration Normalization to target_frames (default 300)
        b, m, t = log_mel.shape
        if t < self.target_frames:
            pad_amount = self.target_frames - t
            # Reflect pad if possible, else replicate
            if t > 1 and pad_amount < t:
                log_mel = F.pad(log_mel, (0, pad_amount), mode="reflect")
            else:
                log_mel = F.pad(log_mel, (0, pad_amount), mode="replicate")
        elif t > self.target_frames:
            # Center crop
            start = (t - self.target_frames) // 2
            log_mel = log_mel[:, :, start : start + self.target_frames]

        # 6. Compute Delta and Delta-Delta channels
        delta1 = compute_delta(log_mel, order=1)
        delta2 = compute_delta(log_mel, order=2)

        # Stack into 3 channels: (B, 3, n_mels, target_frames)
        features = torch.stack([log_mel, delta1, delta2], dim=1)
        return features

"""Unit tests for AcousticModifier and Causal Masking logic."""

import numpy as np
import pytest
import torch
from src.causal.masking_probe import apply_frame_mask
from src.causal.vocoder import AcousticModifier


def test_acoustic_modifier_f0_shift():
    f0 = np.array([0.0, 100.0, 200.0, 0.0, 300.0])
    # Shift up 12 semitones (1 octave = double frequency)
    f0_mod = AcousticModifier.shift_f0_semitones(f0, semitones=12.0)
    assert f0_mod[0] == 0.0
    assert np.isclose(f0_mod[1], 200.0)
    assert np.isclose(f0_mod[2], 400.0)
    assert f0_mod[3] == 0.0
    assert np.isclose(f0_mod[4], 600.0)


def test_acoustic_modifier_energy():
    wav = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    # +6 dB is approx factor of 2
    wav_mod = AcousticModifier.scale_energy(wav, gain_db=6.02)
    assert np.all(np.abs(wav_mod) <= 1.0)
    assert np.isclose(wav_mod[0], 0.2, atol=1e-2)


def test_apply_frame_mask():
    x = torch.ones(2, 3, 128, 100)
    # Give high scores to first 10 frames
    scores = torch.zeros(2, 100)
    scores[:, :10] = 10.0

    # Top-10% masking (10 frames)
    masked_top = apply_frame_mask(x, scores, mask_ratio=0.1, mode="top")
    assert torch.all(masked_top[:, :, :, :10] == 0.0)
    assert torch.all(masked_top[:, :, :, 10:] == 1.0)

    # Bottom-10% masking (10 frames)
    masked_bot = apply_frame_mask(x, scores, mask_ratio=0.1, mode="bottom")
    assert torch.all(masked_bot[:, :, :, :10] == 1.0)
    assert torch.sum(masked_bot == 0.0) > 0

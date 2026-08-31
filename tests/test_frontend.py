"""Unit tests for AudioFrontend and Delta Feature Extraction."""

import pytest
import torch
from src.data.frontend import AudioFrontend, compute_delta, create_mel_filterbank


def test_mel_filterbank_shape():
    fb = create_mel_filterbank(sample_rate=16000, n_fft=1024, n_mels=128)
    assert fb.shape == (128, 513)
    assert torch.all(fb >= 0)
    assert not torch.isnan(fb).any()


def test_delta_computation():
    x = torch.randn(2, 128, 300)
    delta1 = compute_delta(x, order=1)
    delta2 = compute_delta(x, order=2)
    assert delta1.shape == x.shape
    assert delta2.shape == x.shape
    assert not torch.isnan(delta1).any()
    assert not torch.isnan(delta2).any()


def test_frontend_forward_standard():
    frontend = AudioFrontend(sample_rate=16000, n_mels=128, target_frames=300)
    # 3 seconds of 16kHz audio = 48000 samples
    wav = torch.randn(4, 48000)
    feat = frontend(wav)
    assert feat.shape == (4, 3, 128, 300)
    assert not torch.isnan(feat).any()
    assert not torch.isinf(feat).any()


def test_frontend_forward_variable_lengths():
    frontend = AudioFrontend(sample_rate=16000, n_mels=128, target_frames=300)
    # Short audio (0.5s = 8000 samples)
    short_wav = torch.randn(1, 8000)
    feat_short = frontend(short_wav)
    assert feat_short.shape == (1, 3, 128, 300)

    # Long audio (8s = 128000 samples)
    long_wav = torch.randn(1, 128000)
    feat_long = frontend(long_wav)
    assert feat_long.shape == (1, 3, 128, 300)

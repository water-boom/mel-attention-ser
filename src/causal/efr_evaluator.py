"""Emotion Flip Rate (EFR) Evaluator for Causal Acoustic Feedback Verification."""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .vocoder import AcousticModifier, WorldVocoder
from ..data.frontend import AudioFrontend
from ..data.dataset import load_audio_file
from ..data.split import parse_ravdess_filename


class EFREvaluator:
    """Evaluates 8x8 Emotion Flip Rate matrix via WORLD acoustic parameter intervention."""

    def __init__(
        self,
        classifier: nn.Module,
        frontend: AudioFrontend,
        priors: Dict[str, Dict],
        device: torch.device,
        vocoder: Optional[WorldVocoder] = None,
    ):
        self.classifier = classifier
        self.frontend = frontend
        self.priors = priors
        self.device = device
        self.vocoder = vocoder or WorldVocoder(sample_rate=frontend.sample_rate)

    @torch.no_grad()
    def evaluate_file(self, wav_path: str, src_emotion: int, target_emotion: int, intensity: float = 1.0) -> int:
        """Transform a single wav from src_emotion to target_emotion and classify."""
        self.classifier.eval()

        if src_emotion == target_emotion:
            # Baseline test without modification
            wav = load_audio_file(wav_path, self.frontend.sample_rate).numpy()
        else:
            src_p = self.priors.get(str(src_emotion), {})
            tgt_p = self.priors.get(str(target_emotion), {})

            # Compute physical deltas
            f0_src = src_p.get("f0_mean_hz", 160.0)
            f0_tgt = tgt_p.get("f0_mean_hz", 160.0)
            delta_semitones = 12.0 * np.log2(max(f0_tgt, 1.0) / max(f0_src, 1.0))

            std_src = src_p.get("log_f0_std", 0.2)
            std_tgt = tgt_p.get("log_f0_std", 0.2)
            f0_var_scale = std_tgt / max(std_src, 0.05)

            rms_src = src_p.get("rms_db", -40.0)
            rms_tgt = tgt_p.get("rms_db", -40.0)
            gain_db = rms_tgt - rms_src

            raw_wav = load_audio_file(wav_path, self.frontend.sample_rate).numpy()
            wav = AcousticModifier.modify_speech(
                raw_wav,
                self.vocoder,
                delta_semitones=delta_semitones,
                f0_var_scale=f0_var_scale,
                gain_db=gain_db,
                intensity=intensity,
            )

        wav_t = torch.from_numpy(wav).unsqueeze(0)
        feat = self.frontend(wav_t).to(self.device)
        logits = self.classifier(feat)
        pred = logits.argmax(dim=-1).item()
        return pred

    def evaluate_matrix(
        self,
        test_file_paths: List[str],
        intensity: float = 1.0,
        num_classes: int = 8,
    ) -> np.ndarray:
        """Compute the complete 8x8 EFR matrix over all test audio files.

        Returns:
            matrix of shape (num_classes, num_classes) with percentages [0.0, 1.0]
        """
        counts = np.zeros((num_classes, num_classes), dtype=int)
        flips = np.zeros((num_classes, num_classes), dtype=int)

        for p in test_file_paths:
            try:
                src_e, _ = parse_ravdess_filename(p)
            except ValueError:
                continue

            for tgt_e in range(num_classes):
                counts[src_e, tgt_e] += 1
                pred = self.evaluate_file(p, src_e, tgt_e, intensity=intensity)
                if pred == tgt_e:
                    flips[src_e, tgt_e] += 1

        efr_matrix = np.divide(
            flips.astype(float),
            np.maximum(counts, 1),
            out=np.zeros_like(flips, dtype=float),
        )
        return efr_matrix

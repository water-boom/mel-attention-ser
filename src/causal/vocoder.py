"""WORLD Vocoder Analysis, Synthesis & Rule-Based Acoustic Modifier."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import scipy.signal

try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False


class WorldVocoder:
    """Wrapper for WORLD Vocoder decomposition and synthesis."""

    def __init__(self, sample_rate: int = 16000, frame_period: float = 5.0):
        self.sample_rate = sample_rate
        self.frame_period = frame_period

    def extract(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract F0, Spectrogram (SP), and Aperiodicity (AP).

        Returns:
            f0: (T_frames,)
            sp: (T_frames, n_freqs)
            ap: (T_frames, n_freqs)
        """
        if not PYWORLD_AVAILABLE:
            raise ImportError("pyworld is required for WORLD vocoder extraction. Please install pyworld.")

        wav_double = wav.astype(np.float64)
        _f0, t = pw.harvest(wav_double, self.sample_rate, frame_period=self.frame_period)
        f0 = pw.stonemask(wav_double, _f0, t, self.sample_rate)
        sp = pw.cheaptrick(wav_double, f0, t, self.sample_rate)
        ap = pw.d4c(wav_double, f0, t, self.sample_rate)
        return f0, sp, ap

    def synthesize(self, f0: np.ndarray, sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
        """Synthesize waveform from F0, SP, and AP."""
        if not PYWORLD_AVAILABLE:
            raise ImportError("pyworld is required for WORLD vocoder synthesis.")

        y = pw.synthesize(f0.astype(np.float64), sp.astype(np.float64), ap.astype(np.float64), self.sample_rate, self.frame_period)
        return y.astype(np.float32)


class AcousticModifier:
    """Pure rule-based acoustic physical parameters modifier."""

    @staticmethod
    def shift_f0_semitones(f0: np.ndarray, semitones: float) -> np.ndarray:
        """Pitch shift in semitones (only on voiced segments where f0 > 0)."""
        factor = 2.0 ** (semitones / 12.0)
        new_f0 = f0.copy()
        voiced = new_f0 > 0
        new_f0[voiced] = new_f0[voiced] * factor
        return new_f0

    @staticmethod
    def scale_f0_variance(f0: np.ndarray, scale: float) -> np.ndarray:
        """Scale pitch dynamics / variance on log-F0 space while preserving geometric mean."""
        new_f0 = f0.copy()
        voiced = new_f0 > 0
        if np.sum(voiced) < 2 or scale <= 0:
            return new_f0

        log_f0 = np.log(new_f0[voiced])
        mean_log = np.mean(log_f0)
        scaled_log = mean_log + scale * (log_f0 - mean_log)
        new_f0[voiced] = np.exp(scaled_log)
        return new_f0

    @staticmethod
    def scale_energy(wav: np.ndarray, gain_db: float, max_clip_db: float = 12.0) -> np.ndarray:
        """Adjust waveform RMS energy by gain_db with soft limiter protection."""
        # Clamp gain to prevent extreme clipping
        gain_db = float(np.clip(gain_db, -max_clip_db, max_clip_db))
        linear_gain = 10.0 ** (gain_db / 20.0)
        wav_mod = wav * linear_gain
        # Hard limit to [-0.99, 0.99]
        return np.clip(wav_mod, -0.99, 0.99).astype(np.float32)

    @classmethod
    def modify_speech(
        cls,
        wav: np.ndarray,
        vocoder: WorldVocoder,
        delta_semitones: float = 0.0,
        f0_var_scale: float = 1.0,
        gain_db: float = 0.0,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Apply combined physical acoustic transformation."""
        f0, sp, ap = vocoder.extract(wav)

        # Scale transformation parameters by intensity [0.0, 1.0]
        eff_semitones = delta_semitones * intensity
        eff_var_scale = 1.0 + (f0_var_scale - 1.0) * intensity
        eff_gain_db = gain_db * intensity

        # Modify F0
        mod_f0 = cls.shift_f0_semitones(f0, eff_semitones)
        mod_f0 = cls.scale_f0_variance(mod_f0, eff_var_scale)

        # Synthesize modified waveform
        mod_wav = vocoder.synthesize(mod_f0, sp, ap)

        # Modify energy
        mod_wav = cls.scale_energy(mod_wav, eff_gain_db)
        return mod_wav

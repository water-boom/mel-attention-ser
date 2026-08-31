"""Script to pre-extract Log-Mel features into data/cache/ for fast training."""

import argparse
import glob
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import yaml

from src.data.frontend import AudioFrontend
from src.data.dataset import load_audio_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir(value: str) -> str:
    """Resolve a data-dir value into an absolute path.

    Supports: absolute path, path relative to the repo root, and the
    ``${VAR:-default}`` env-var form (so no local absolute path needs to be
    hardcoded in the repo).
    """
    if value.startswith("${") and ":-" in value and value.endswith("}"):
        inner = value[2:-1]
        var, _, default = inner.partition(":-")
        value = os.environ.get(var, default)
    value = os.path.expanduser(value)
    p = Path(value)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


def main():
    parser = argparse.ArgumentParser(description="Pre-extract Log-Mel features to cache.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = resolve_data_dir(args.data_dir or cfg["data"]["raw_data_dir"])
    cache_dir = args.cache_dir or cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in '{data_dir}'. Please verify data directory.")
        return

    print(f"Found {len(wav_files)} audio files. Pre-extracting Log-Mel features...")
    frontend = AudioFrontend(
        sample_rate=cfg["data"]["sample_rate"],
        n_fft=cfg["data"]["n_fft"],
        win_length=cfg["data"]["win_length"],
        hop_length=cfg["data"]["hop_length"],
        n_mels=cfg["data"]["n_mels"],
        target_frames=cfg["data"]["target_frames"],
    )

    count = 0
    with torch.no_grad():
        for p in tqdm(wav_files):
            stem = Path(p).stem
            out_path = os.path.join(cache_dir, f"{stem}.npy")
            if not os.path.exists(out_path):
                wav = load_audio_file(p, cfg["data"]["sample_rate"])
                feat = frontend(wav).squeeze(0).cpu().numpy()  # (3, 128, 300)
                np.save(out_path, feat)
                count += 1

    print(f"Done! Cached {count} new features in '{cache_dir}'.")


if __name__ == "__main__":
    main()

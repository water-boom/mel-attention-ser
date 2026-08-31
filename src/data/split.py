"""Speaker-Independent 5-Fold Stratified Cross-Validation Splitter.

Guarantees 100% Actor isolation between train and validation sets across all folds.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def parse_ravdess_filename(file_path: str) -> Tuple[int, int]:
    """Parse emotion ID (0-7) and actor ID (1-24) from RAVDESS filename.

    Filename format: 03-01-XX-01-01-01-YY.wav
    XX: emotion (01:neutral, 02:calm, 03:happy, 04:sad, 05:angry, 06:fearful, 07:disgust, 08:surprised)
    YY: actor (01-24)

    Returns:
        (emotion_label_0_to_7, actor_id_1_to_24)
    """
    stem = Path(file_path).stem
    parts = stem.split("-")
    if len(parts) >= 7:
        emotion_raw = int(parts[2])
        actor_id = int(parts[6])
        emotion_label = emotion_raw - 1  # 0-indexed
        return emotion_label, actor_id
    else:
        # Fallback or synthetic parsing
        raise ValueError(f"Invalid RAVDESS filename format: {stem}")


class SpeakerSplitter:
    """Stratified Group K-Fold Splitter for Speaker-Independent Evaluation."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def create_splits(self, file_list: List[str]) -> List[Dict]:
        """Generate 5-Fold splits from a list of audio file paths.

        Returns:
            List of dicts, each containing:
                'fold': int,
                'train_files': List[str],
                'val_files': List[str],
                'train_actors': List[int],
                'val_actors': List[int]
        """
        emotions = []
        actors = []
        valid_files = []

        for f in file_list:
            try:
                e, a = parse_ravdess_filename(f)
                emotions.append(e)
                actors.append(a)
                valid_files.append(f)
            except ValueError:
                continue

        if not valid_files:
            raise ValueError("No valid RAVDESS audio files found to create splits.")

        X = np.array(valid_files)
        y = np.array(emotions)
        groups = np.array(actors)

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            train_files = X[train_idx].tolist()
            val_files = X[val_idx].tolist()
            train_actors = sorted(list(set(groups[train_idx].tolist())))
            val_actors = sorted(list(set(groups[val_idx].tolist())))

            # Academic integrity assertion: Zero intersection
            assert set(train_actors).isdisjoint(set(val_actors)), (
                f"Speaker leakage detected in fold {fold_idx}! "
                f"Intersection: {set(train_actors) & set(val_actors)}"
            )

            folds.append({
                "fold": fold_idx,
                "train_files": train_files,
                "val_files": val_files,
                "train_actors": train_actors,
                "val_actors": val_actors,
                "num_train": len(train_files),
                "num_val": len(val_files),
            })

        return folds

    def save_splits(self, folds: List[Dict], output_path: str):
        """Save split definitions to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2, ensure_ascii=False)

    def load_splits(self, input_path: str) -> List[Dict]:
        """Load split definitions from JSON and verify integrity."""
        with open(input_path, "r", encoding="utf-8") as f:
            folds = json.load(f)

        for fold in folds:
            train_actors = set(fold["train_actors"])
            val_actors = set(fold["val_actors"])
            assert train_actors.isdisjoint(val_actors), (
                f"Integrity check failed for fold {fold['fold']}: actor overlap detected."
            )
        return folds

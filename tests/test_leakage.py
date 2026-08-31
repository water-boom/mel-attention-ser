"""Unit tests for Speaker Leakage Isolation and Filename Parsing."""

import pytest
from src.data.split import SpeakerSplitter, parse_ravdess_filename


def test_parse_ravdess_filename():
    fn = "03-01-05-01-02-01-12.wav"
    emotion, actor = parse_ravdess_filename(fn)
    assert emotion == 4  # 05 -> index 4 (angry)
    assert actor == 12


def test_speaker_split_zero_leakage():
    # Simulate full RAVDESS 1440 files
    fake_files = []
    for actor in range(1, 25):
        for emotion in range(1, 9):
            for rep in range(1, 8):
                fake_files.append(f"03-01-{emotion:02d}-01-01-01-{actor:02d}.wav")

    splitter = SpeakerSplitter(n_splits=5, seed=42)
    folds = splitter.create_splits(fake_files)

    assert len(folds) == 5

    all_val_actors = []
    for fold in folds:
        train_actors = set(fold["train_actors"])
        val_actors = set(fold["val_actors"])

        # Strict academic integrity assertions
        assert train_actors.isdisjoint(val_actors), (
            f"Leakage in fold {fold['fold']}: {train_actors & val_actors}"
        )
        assert len(train_actors) > 0
        assert len(val_actors) > 0
        all_val_actors.extend(list(val_actors))

    # All 24 actors should appear in validation across 5 folds
    assert len(set(all_val_actors)) == 24

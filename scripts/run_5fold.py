"""Script to run 5-Fold Speaker-Independent Cross-Validation on Attention Zoo."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.frontend import AudioFrontend
from src.data.split import SpeakerSplitter
from src.data.dataset import SERDataset
from src.models.registry import build_model, list_models
from src.engine.trainer import Trainer, EarlyStopping


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
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross-Validation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="List of model keys to benchmark")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = resolve_data_dir(args.data_dir or cfg["data"]["raw_data_dir"])
    cache_dir = cfg["data"]["cache_dir"]
    epochs = args.epochs or cfg["train"]["max_epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    patience = cfg["train"]["patience"]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_run = args.models or cfg["models"]
    print(f"=== Starting 5-Fold Benchmark on Device: {device} ===")
    print(f"Models to evaluate: {models_to_run}")

    # 1. Collect audio files & build/load splits
    # folds.json stores paths relative to the RAVDESS data root (portable,
    # no local absolute paths leak); we join them with data_dir at load time.
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"Warning: No .wav found in '{data_dir}'. Generating synthetic demo split...")
        wav_files = [f"03-01-{e:02d}-01-01-01-{a:02d}.wav" for a in range(1, 25) for e in range(1, 9)]

    splitter = SpeakerSplitter(n_splits=cfg["eval"]["n_splits"], seed=cfg["seed"])
    folds_path = os.path.join(PROJECT_ROOT, cfg["eval"]["folds_path"])
    if os.path.exists(folds_path):
        folds = splitter.load_splits(folds_path)
        for fold in folds:
            for key in ("train_files", "val_files"):
                fold[key] = [
                    f if os.path.isabs(f) else os.path.join(data_dir, f)
                    for f in fold[key]
                ]
    else:
        # Save splits relative to data_dir so the repo stays portable.
        folds = splitter.create_splits(wav_files)
        for fold in folds:
            for key in ("train_files", "val_files"):
                fold[key] = [os.path.relpath(f, data_dir) for f in fold[key]]
        splitter.save_splits(folds, folds_path)

    frontend = AudioFrontend(
        sample_rate=cfg["data"]["sample_rate"],
        n_fft=cfg["data"]["n_fft"],
        win_length=cfg["data"]["win_length"],
        hop_length=cfg["data"]["hop_length"],
        n_mels=cfg["data"]["n_mels"],
        target_frames=cfg["data"]["target_frames"],
    )

    all_results = {}
    dynamics_all = {}
    os.makedirs(cfg["analysis"]["output_dir"], exist_ok=True)

    for model_name in models_to_run:
        print(f"\n" + "=" * 50)
        print(f" Benchmarking Model: {model_name.upper()} (5 Folds)")
        print("=" * 50)

        fold_f1s = []
        fold_wars = []
        fold_uars = []
        dynamics_all[model_name] = []

        for fold_info in folds:
            fold_idx = fold_info["fold"]
            train_files = fold_info["train_files"]
            val_files = fold_info["val_files"]

            print(f"\n--- [Fold {fold_idx + 1}/5] Train: {len(train_files)} files | Val: {len(val_files)} files ---")

            # In-fold stats
            mean, std = SERDataset.compute_fold_stats(train_files, frontend, cache_dir=cache_dir)

            train_ds = SERDataset(train_files, frontend, cache_dir=cache_dir, mean=mean, std=std, augment=True)
            val_ds = SERDataset(val_files, frontend, cache_dir=cache_dir, mean=mean, std=std, augment=False)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = build_model(model_name, num_classes=8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["train"]["min_lr"])
            early_stopping = EarlyStopping(patience=patience, mode="max")

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
                grad_clip=cfg["train"]["grad_clip"],
                early_stopping=early_stopping,
            )

            history = trainer.fit(train_loader, val_loader, max_epochs=epochs, verbose=True)
            best_m = history["best_val_metrics"] or {"macro_f1": 0.0, "war": 0.0, "uar": 0.0}

            fold_f1s.append(best_m["macro_f1"])
            fold_wars.append(best_m["war"])
            fold_uars.append(best_m["uar"])

            dynamics_all[model_name].append({
                "fold": fold_idx,
                "entropy_traj": history["entropy_traj"],
                "diversity_traj": history["diversity_traj"],
            })

            print(f"[Fold {fold_idx + 1} Best] Macro-F1: {best_m['macro_f1']:.4f} | WAR: {best_m['war']:.4f} | UAR: {best_m['uar']:.4f}")

        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        mean_war, std_war = np.mean(fold_wars), np.std(fold_wars)
        mean_uar, std_uar = np.mean(fold_uars), np.std(fold_uars)

        all_results[model_name] = {
            "macro_f1_mean": float(mean_f1),
            "macro_f1_std": float(std_f1),
            "war_mean": float(mean_war),
            "war_std": float(std_war),
            "uar_mean": float(mean_uar),
            "uar_std": float(std_uar),
            "fold_f1s": fold_f1s,
        }

    # Save benchmark markdown report
    report_path = os.path.join(cfg["analysis"]["output_dir"], "benchmark.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 5-Fold Speaker-Independent Benchmark Report\n\n")
        f.write("| Model Name | Macro-F1 (mean±std) | WAR / Accuracy | UAR / Recall | Mechanism |\n")
        f.write("|---|---|---|---|---|\n")
        for m_name, res in all_results.items():
            f.write(
                f"| `{m_name}` | **{res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}** | "
                f"{res['war_mean']:.4f} ± {res['war_std']:.4f} | "
                f"{res['uar_mean']:.4f} ± {res['uar_std']:.4f} | "
                f"{MODEL_REGISTRY_DESC.get(m_name, '')} |\n"
            )

    # Save dynamics data
    dynamics_path = os.path.join(cfg["analysis"]["output_dir"], "dynamics.json")
    with open(dynamics_path, "w", encoding="utf-8") as f:
        json.dump(dynamics_all, f, indent=2)

    print(f"\n=== Benchmark Complete! Results saved to '{report_path}' ===")


MODEL_REGISTRY_DESC = {
    "cnn_base": "4-Layer CNN + GAP (Baseline)",
    "cnn_se": "4-Layer CNN + Channel SE Attention",
    "cnn_coord": "4-Layer CNN + Coordinate Spectro-Temporal Attention",
    "cnn_mhap": "4-Layer CNN + 4-Head Temporal Pooling (Ours)",
    "cnn_asp": "4-Layer CNN + Attentive Statistics Pooling (1st & 2nd Order)",
    "w2v2_mhap": "Frozen wav2vec2-base + MHAP Head",
}


if __name__ == "__main__":
    main()

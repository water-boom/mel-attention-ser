"""Script to generate Learning Dynamics and Causal Masking analysis figures."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_dynamics(dynamics_data: dict, out_dir: str):
    """Plot attention entropy trajectory across epochs."""
    plt.figure(figsize=(8, 5))
    has_data = False

    for model_name, folds_data in dynamics_data.items():
        if not folds_data:
            continue
        all_trajs = []
        for f in folds_data:
            traj = [item["normalized_entropy"] for item in f.get("entropy_traj", [])]
            if traj:
                all_trajs.append(traj)

        if all_trajs:
            min_len = min(len(t) for t in all_trajs)
            truncated = np.array([t[:min_len] for t in all_trajs])
            mean_traj = np.mean(truncated, axis=0)
            std_traj = np.std(truncated, axis=0)
            epochs = np.arange(1, min_len + 1)

            plt.plot(epochs, mean_traj, label=f"{model_name}", lw=2)
            plt.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj, alpha=0.15)
            has_data = True

    if has_data:
        plt.title("Learning Dynamics: Attention Entropy Evolution ($H_{norm}$)", fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Normalized Attention Entropy", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, "dynamics_entropy.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved entropy dynamics plot to: {out_path}")


def plot_head_diversity(dynamics_data: dict, out_dir: str):
    """Plot multi-head diversity / specialization over epochs."""
    plt.figure(figsize=(8, 5))
    has_data = False

    for model_name, folds_data in dynamics_data.items():
        if not folds_data:
            continue
        all_trajs = []
        for f in folds_data:
            traj = [item["head_diversity"] for item in f.get("diversity_traj", [])]
            if traj:
                all_trajs.append(traj)

        if all_trajs:
            min_len = min(len(t) for t in all_trajs)
            truncated = np.array([t[:min_len] for t in all_trajs])
            mean_traj = np.mean(truncated, axis=0)
            std_traj = np.std(truncated, axis=0)
            epochs = np.arange(1, min_len + 1)

            plt.plot(epochs, mean_traj, label=f"{model_name}", lw=2)
            plt.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj, alpha=0.15)
            has_data = True

    if has_data:
        plt.title("Multi-Head Specialization Dynamics ($1 - \\cos(h_i, h_j)$)", fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Pairwise Head Diversity", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, "head_diversity.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved head diversity plot to: {out_path}")


def plot_causal_masking_curve(masking_results: dict, out_dir: str):
    """Plot Macro-F1 degradation curves under Top-K vs Bottom-K masking."""
    plt.figure(figsize=(8, 5))
    ratios = sorted([float(k) for k in masking_results.get("top", {}).keys()])
    top_f1 = [masking_results["top"][r] for r in ratios]
    bot_f1 = [masking_results["bottom"][r] for r in ratios]
    rnd_f1 = [masking_results["random"][r] for r in ratios]

    plt.plot(ratios, top_f1, "r-o", label="Mask Top-K (Highest Attention)", lw=2)
    plt.plot(ratios, bot_f1, "g-s", label="Mask Bottom-K (Lowest Attention)", lw=2)
    plt.plot(ratios, rnd_f1, "b--^", label="Mask Random-K (Baseline)", lw=2)

    plt.title("Causal Perturbation: Macro-F1 vs Frame Masking Ratio", fontsize=13)
    plt.xlabel("Masking Ratio", fontsize=11)
    plt.ylabel("Macro-F1", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "causal_masking_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved causal masking curve to: {out_path}")


def main():
    results_dir = "results"
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    dyn_path = os.path.join(results_dir, "dynamics.json")
    if os.path.exists(dyn_path):
        with open(dyn_path, "r", encoding="utf-8") as f:
            dynamics_data = json.load(f)
        plot_entropy_dynamics(dynamics_data, fig_dir)
        plot_head_diversity(dynamics_data, fig_dir)
    else:
        print(f"Notice: '{dyn_path}' not found yet. Run 5-fold benchmark first.")


if __name__ == "__main__":
    main()

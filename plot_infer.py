#!/usr/bin/env python3
"""
plot_infer.py — Plot average inference speed vs average AUPR from benchmark results.

Usage:
    python plot_infer.py
    python plot_infer.py --input results/benchmark_all.json
    python plot_infer.py --input results/our.json
    python plot_infer.py --output results/plot/infer_vs_aupr.png
"""

import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]

ARCH_DISPLAY_NAME = {
    "bocknet": "BockNet",
    "our": "Ours",
    "pa2e": "PA2E",
    "gthad": "GT-HAD",
    "superad": "SuperAD",
    "sglnet": "SGLNet",
    "autoad": "Auto-AD",
    "otad": "OT-AD",
    "dms2f": "DMS2F-HAD",
}


def main():
    parser = argparse.ArgumentParser(description="Plot avg inference speed vs avg AUPR.")
    parser.add_argument("--input", default="results/benchmark_all.json",
                        help="Path to result JSON (default: results/benchmark_all.json)")
    parser.add_argument("--output", default="results/plot/infer_vs_aupr.png",
                        help="Output plot path (default: results/plot/infer_vs_aupr.png)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    archs = []
    infer_times = []
    aupr_scores = []

    for arch_key, arch_results in data.items():
        times = []
        auprs = []
        for ft in ALL_FOOD_TYPES:
            ft_res = arch_results.get(ft)
            if ft_res is None:
                continue
            t = ft_res.get("infer_time_sec")
            p = ft_res.get("pr_auc")
            if t is not None and p is not None:
                times.append(t)
                auprs.append(p)
        if len(times) == 0:
            continue
        archs.append(arch_key)
        infer_times.append(np.mean(times))
        aupr_scores.append(np.mean(auprs))

    if len(archs) == 0:
        print("No data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(archs)))

    for i, arch in enumerate(archs):
        label = ARCH_DISPLAY_NAME.get(arch, arch)
        ax.scatter(infer_times[i], aupr_scores[i], c=[colors[i]], s=120, zorder=5)
        ax.annotate(label, (infer_times[i], aupr_scores[i]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9,
                    fontweight='bold')

    ax.set_xlabel("Avg Inference Time (s)", fontsize=12)
    ax.set_ylabel("Avg AUPR (PR-AUC)", fontsize=12)
    ax.set_title("Inference Speed vs AUPR across Architectures", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved → {args.output}")

    print(f"\n{'Architecture':<16}  {'Avg Infer (s)':<14}  {'Avg AUPR':<10}")
    print("-" * 44)
    for arch, t, p in sorted(zip(archs, infer_times, aupr_scores),
                              key=lambda x: x[1]):
        name = ARCH_DISPLAY_NAME.get(arch, arch)
        print(f"{name:<16}  {t:<14.4f}  {p:<10.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
benchmark.py — Run all architectures against the food anomaly dataset.

Usage:
    python benchmark.py
    python benchmark.py --food Almond
    python benchmark.py --food Almond Pistachio --output-dir ./results
    python benchmark.py --skip ours           # skip one architecture
    python benchmark.py --only gt-had         # run a single architecture
"""

# (imports keep the same)
import argparse
import importlib
import json
import os
import sys
import time
import traceback
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_duration(value):
    """Parse a duration string like '30s', '1m', '5m' into seconds."""
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("must be a string")
    m = re.match(r'^(\d+)\s*([sm]?)$', value)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid duration: '{value}'. Use e.g. 30s, 1m, 5m")
    num = int(m.group(1))
    unit = m.group(2) or 's'
    return num * 60 if unit == 'm' else num

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]
ALL_ARCHS = [
    "bocknet",
    "our",
    "pa2e",
    "gthad",
    "superad",
    "sglnet",
    "autoad",
    "otad",
    # "cbar",
]
ARCH_MODULE_MAP = {
    "bocknet": "scripts.bocknet.__main__",
    "our": "scripts.our.__main__",
    "pa2e": "scripts.pa2e.__main__",
    "gthad": "scripts.gthad.__main__",
    "superad": "scripts.superad.__main__",
    "sglnet": "scripts.sglnet.__main__",
    "autoad": "scripts.autoad.__main__",
    "otad": "scripts.otad.__main__",
    # "cbar": "scripts.cbar.__main__",
}
ARCH_RESULT_FILE = {
    "bocknet": "bocknet.json",
    "our": "our.json",
    "pa2e": "pa2e.json",
    "gthad": "gthad.json",
    "superad": "superad.json",
    "sglnet": "sglnet.json",
    "autoad": "autoad.json",
    "otad": "otad.json",
    # "cbar": "cbar.json",
}


def run_arch(arch: str, food_types: list, output_dir: str, extra_kwargs: dict, cooldown: int = 0) -> dict:
    """Import and call benchmark_food_type for each food type in an architecture."""
    print(f"\n{'#' * 80}")
    print(f"# ARCHITECTURE: {arch}")
    print(f"{'#' * 80}")

    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    arch_results = {}
    anomaly_dir = os.path.join(output_dir, 'anomaly-map')
    os.makedirs(anomaly_dir, exist_ok=True)
    for fi, ft in enumerate(food_types):
        try:
            # Merge architecture-specific kwargs with common kwargs
            kwargs = {**extra_kwargs.get("common", {}), **extra_kwargs.get(arch, {}), "anomaly_dir": anomaly_dir}
            result = bench_fn(ft, **kwargs)
            arch_results[ft] = result
        except Exception as e:
            print(f"\n❌ [{arch}] Error on {ft}: {e}")
            traceback.print_exc()

        # Clear GPU VRAM between food types
        if fi < len(food_types) - 1:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if cooldown > 0:
                print(f"  Cooling down GPU for {cooldown}s...")
                time.sleep(cooldown)

    # Save per-architecture JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, ARCH_RESULT_FILE[arch])
    with open(out_path, "w") as f:
        # Handle numpy serialization
        def convert_numpy(obj):
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(arch_results, f, indent=2, default=convert_numpy)
    print(f"\n✅ [{arch}] Results saved → {out_path}")

    return arch_results


def print_summary(all_results: dict):
    """Print a combined cross-architecture summary table."""
    food_types = sorted({ft for results in all_results.values() for ft in results})
    archs = list(all_results.keys())

    col = 16

    print(f"\n{'=' * 80}\n{'FINAL CROSS-ARCHITECTURE SUMMARY':^80}\n{'=' * 80}")

    # ROC table
    print("\n📊 ROC-AUC Scores:")
    roc_header = f"{'Architecture':<20}"
    for ft in food_types:
        roc_header += f"  {ft:>{col}}"
    roc_header += f"  {'Average':>{col}}"
    print(roc_header)
    print("─" * len(roc_header))

    # Calculate averages and sort
    roc_avgs = {}
    for arch in archs:
        scores = []
        for ft in food_types:
            r = all_results[arch].get(ft)
            if r is not None:
                if "original_model" in r:
                    roc = r["original_model"].get("roc_auc")
                else:
                    roc = r.get("roc_auc")
                if roc is not None and not np.isnan(roc):
                    scores.append(roc)
        roc_avgs[arch] = np.mean(scores) if scores else float("nan")

    sorted_archs_roc = sorted(archs, key=lambda a: roc_avgs[a], reverse=True)

    for arch in sorted_archs_roc:
        row = f"{arch:<20}"
        for ft in food_types:
            r = all_results[arch].get(ft)
            if r is None:
                row += f"  {'N/A':>{col}}"
            elif "original_model" in r:
                roc = r["original_model"]["roc_auc"]
                roc_str = (
                    f"{roc:>.4f}" if (roc is not None and not np.isnan(roc)) else "nan"
                )
                row += f"  {roc_str:>{col}}"
            else:
                roc = r.get("roc_auc")
                roc_str = (
                    f"{roc:>.4f}" if (roc is not None and not np.isnan(roc)) else "nan"
                )
                row += f"  {roc_str:>{col}}"
        avg_str = f"{roc_avgs[arch]:>.4f}" if not np.isnan(roc_avgs[arch]) else "nan"
        row += f"  {avg_str:>{col}}"
        print(row)

    # PR table
    print("\n📊 PR-AUC Scores:")
    pr_header = f"{'Architecture':<20}"
    for ft in food_types:
        pr_header += f"  {ft:>{col}}"
    pr_header += f"  {'Average':>{col}}"
    print(pr_header)
    print("─" * len(pr_header))

    # Calculate averages and sort
    pr_avgs = {}
    for arch in archs:
        scores = []
        for ft in food_types:
            r = all_results[arch].get(ft)
            if r is not None:
                if "original_model" in r:
                    pr = r["original_model"].get("pr_auc")
                else:
                    pr = r.get("pr_auc")
                if pr is not None and not np.isnan(pr):
                    scores.append(pr)
        pr_avgs[arch] = np.mean(scores) if scores else float("nan")

    sorted_archs_pr = sorted(archs, key=lambda a: pr_avgs[a], reverse=True)

    for arch in sorted_archs_pr:
        row = f"{arch:<20}"
        for ft in food_types:
            r = all_results[arch].get(ft)
            if r is None:
                row += f"  {'N/A':>{col}}"
            elif "original_model" in r:
                pr = r["original_model"]["pr_auc"]
                pr_str = (
                    f"{pr:>.4f}" if (pr is not None and not np.isnan(pr)) else "nan"
                )
                row += f"  {pr_str:>{col}}"
            else:
                pr = r.get("pr_auc")
                pr_str = (
                    f"{pr:>.4f}" if (pr is not None and not np.isnan(pr)) else "nan"
                )
                row += f"  {pr_str:>{col}}"
        avg_str = f"{pr_avgs[arch]:>.4f}" if not np.isnan(pr_avgs[arch]) else "nan"
        row += f"  {avg_str:>{col}}"
        print(row)

    print()


def plot_curves(all_results: dict, output_dir: str):
    """Plot ROC/PR curves, anomaly maps, and box plots per food type."""
    food_types = sorted({ft for results in all_results.values() for ft in results})
    archs = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(archs)))

    plot_dir = os.path.join(output_dir, 'plot')
    anomaly_dir = os.path.join(output_dir, 'anomaly-map')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)

    for ft in food_types:
        # ── ROC Curve ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
        for arch, color in zip(archs, colors):
            r = all_results[arch].get(ft)
            if r is None or "fpr" not in r:
                continue
            roc_auc = r.get("roc_auc", float("nan"))
            ax.plot(r["fpr"], r["tpr"], color=color, lw=1.5, label=f"{arch} (AUC={roc_auc:.4f})")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {ft}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        roc_path = os.path.join(plot_dir, f"roc_curve_{ft}.png")
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"  ROC curve saved → {roc_path}")

        # ── PR Curve ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        for arch, color in zip(archs, colors):
            r = all_results[arch].get(ft)
            if r is None or "precision" not in r:
                continue
            pr_auc = r.get("pr_auc", float("nan"))
            ax.plot(r["recall"], r["precision"], color=color, lw=1.5, label=f"{arch} (AP={pr_auc:.4f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curves — {ft}')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pr_path = os.path.join(plot_dir, f"pr_curve_{ft}.png")
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        print(f"  PR curve saved → {pr_path}")

        # ── Anomaly Map Heatmaps ─────────────────────────────────────────
        for arch in archs:
            score_path = os.path.join(anomaly_dir, f'{arch}_{ft}_scores.npy')
            if not os.path.exists(score_path):
                continue
            scores = np.load(score_path)
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                scores = (scores - smin) / (smax - smin)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(scores, cmap='hot', aspect='auto')
            plt.colorbar(im, ax=ax, label='Normalized Score [0,1]')
            ax.set_title(f'{arch} — {ft} Anomaly Map')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            fig.tight_layout()
            map_path = os.path.join(anomaly_dir, f'{arch}_{ft}_map.png')
            fig.savefig(map_path, dpi=150)
            plt.close(fig)
        print(f"  Anomaly maps saved → {anomaly_dir}/")

        # ── Box Plot: anomaly vs normal per architecture ─────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        label_path = os.path.join(anomaly_dir, f'{ft}_labels.npy')
        if os.path.exists(label_path):
            gt = np.load(label_path).astype(bool)
            positions = []
            labels_list = []
            data_normal = []
            data_anomaly = []
            for i, arch in enumerate(archs):
                score_path = os.path.join(anomaly_dir, f'{arch}_{ft}_scores.npy')
                if not os.path.exists(score_path):
                    continue
                scores = np.load(score_path)
                # Normalize per architecture to [0,1] for comparable box plots
                smin, smax = scores.min(), scores.max()
                if smax > smin:
                    scores = (scores - smin) / (smax - smin)
                else:
                    scores = np.zeros_like(scores)
                scores_flat = scores.flatten()
                gt_flat = gt.flatten() if gt.shape == scores.shape else gt.flatten()[:len(scores_flat)]
                mask = gt_flat[:len(scores_flat)]
                norm = scores_flat[~mask]
                anom = scores_flat[mask]
                if len(norm) > 0 and len(anom) > 0:
                    data_normal.append(norm)
                    data_anomaly.append(anom)
                    positions.append(i)
                    labels_list.append(arch)
            if data_normal:
                bp1 = ax.boxplot(data_normal, positions=[p - 0.2 for p in positions],
                                 widths=0.3, patch_artist=True,
                                 boxprops=dict(facecolor='steelblue', alpha=0.7),
                                 medianprops=dict(color='white'))
                bp2 = ax.boxplot(data_anomaly, positions=[p + 0.2 for p in positions],
                                 widths=0.3, patch_artist=True,
                                 boxprops=dict(facecolor='crimson', alpha=0.7),
                                 medianprops=dict(color='white'))
                ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Normal', 'Anomaly'], loc='upper right')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels_list, fontsize=8)
                ax.set_xlabel('Architecture')
                ax.set_ylabel('Anomaly Score')
                ax.set_title(f'Score Distribution — {ft}')
                ax.grid(True, alpha=0.3, axis='y')
                fig.tight_layout()
                box_path = os.path.join(plot_dir, f'boxplot_{ft}.png')
                fig.savefig(box_path, dpi=150)
                print(f"  Box plot saved → {box_path}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Full benchmark: run all HSI food anomaly architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py
  python benchmark.py --food Almond
  python benchmark.py --only gthad
  python benchmark.py --skip gthad --output-dir ./results/run1
  python benchmark.py --dry-run                  # train only 1 epoch/iter per architecture
  python benchmark.py --dry-run --retrain no     # load saved models without retraining
        """,
    )
    parser.add_argument(
        "--food",
        nargs="*",
        default=None,
        metavar="FOOD",
        help="Food types (default: all)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="ARCH",
        help=f"Run only these architectures: {ALL_ARCHS}",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        metavar="ARCH",
        help="Skip these architectures",
    )
    parser.add_argument(
        "--dry-run",
        "-dr",
        action="store_true",
        help="Train for 1 iteration/epoch only to validate the script works",
    )
    parser.add_argument(
        "--retrain",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Retrain models or load saved ones (default: yes)",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for JSON results (default: ./results)",
    )
    parser.add_argument(
        "--cooldown",
        type=parse_duration,
        default="0s",
        metavar="DURATION",
        help="GPU cooldown between architectures (e.g. 30s, 1m, 5m). Default: 0s",
    )
    args = parser.parse_args()

    # ── food types ──────────────────────────────────────────────────────
    food_types = args.food if (args.food and args.food != ["all"]) else ALL_FOOD_TYPES
    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            sys.exit(1)

    # For dry-run, only test with Almond (faster validation)
    if args.dry_run:
        food_types = ["Almond"]

    # ── architectures ───────────────────────────────────────────────────
    archs = ALL_ARCHS
    if args.only:
        archs = [a for a in ALL_ARCHS if a in args.only]
    if args.skip:
        archs = [a for a in archs if a not in args.skip]
    if not archs:
        print("❌ No architectures selected.")
        sys.exit(1)

    # ── build extra kwargs ───────────────────────────────────────────────
    extra_kwargs = {
        "common": {
            "dry_run": args.dry_run,
            "retrain": (args.retrain == "yes"),
        },
    }

    dry_run_str = " [DRY RUN]" if args.dry_run else ""

    print(f"\n{'=' * 80}")
    print(f"  HSI Food Anomaly Benchmark{dry_run_str}")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"  Food types    : {', '.join(food_types)}")
    print(f"  Retrain       : {args.retrain}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'=' * 80}")

    # ── run ─────────────────────────────────────────────────────────────
    all_results = {}
    total_start = time.time()

    for arch in archs:
        arch_start = time.time()
        all_results[arch] = run_arch(arch, food_types, args.output_dir, extra_kwargs, args.cooldown)
        elapsed = time.time() - arch_start
        print(f"\n⏱  [{arch}] total wall time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall time: {total_elapsed:.1f}s")

    # ── combined JSON ────────────────────────────────────────────────────
    combined_path = os.path.join(args.output_dir, "benchmark_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Combined results saved → {combined_path}")

    print_summary(all_results)

    # ── plot curves ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  Generating plots (ROC/PR curves, anomaly maps, box plots)...")
    print(f"  → results/plot/     : ROC/PR curves and box plots")
    print(f"  → results/anomaly-map/ : anomaly score heatmaps")
    print(f"{'=' * 80}")
    plot_curves(all_results, args.output_dir)


if __name__ == "__main__":
    main()

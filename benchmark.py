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
    "dms2f",
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
    "dms2f": "scripts.dms2f.__main__",
    # "cbar": "scripts.cbar.__main__",
}
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
ARCH_RESULT_FILE = {
    "bocknet": "bocknet.json",
    "our": "our.json",
    "pa2e": "pa2e.json",
    "gthad": "gthad.json",
    "superad": "superad.json",
    "sglnet": "sglnet.json",
    "autoad": "autoad.json",
    "otad": "otad.json",
    "dms2f": "dms2f.json",
    # "cbar": "cbar.json",
}


def aggregate_runs(runs: list) -> dict:
    """Aggregate per-seed result dicts into mean ± std for each metric.

    Scalar metrics (roc_auc, pr_auc, ...) are stored as mean with a
    parallel ``<key>_std`` entry. Curve metrics (fpr/tpr/precision/recall)
    are interpolated onto a shared grid and stored as mean curves with
    ``<key>_std`` entries.
    """
    import numpy as np

    SCALAR_KEYS = [
        "roc_auc", "pr_auc", "f1_tnr95", "acc_tnr95",
        "infer_time_sec", "max_vram_gb", "n_params", "gflops",
        "train_time_sec", "converged_epoch",
    ]
    agg = {}

    for k in SCALAR_KEYS:
        vals = [r.get(k) for r in runs if r.get(k) is not None]
        if not vals:
            continue
        vals = [float(v) for v in vals]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Curves: interpolate onto shared grid then average
    grid = np.linspace(0.0, 1.0, 200)
    for x_key, y_key in [("fpr", "tpr"), ("recall", "precision")]:
        ys = []
        for r in runs:
            x = r.get(x_key)
            y = r.get(y_key)
            if x is None or y is None:
                continue
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            order = np.argsort(x)
            x, y = x[order], y[order]
            if len(x) < 2:
                continue
            ys.append(np.interp(grid, x, y))
        if not ys:
            continue
        ys = np.stack(ys)
        agg[x_key] = grid.tolist()
        agg[y_key] = np.mean(ys, axis=0).tolist()
        agg[f"{y_key}_std"] = np.std(ys, axis=0).tolist()

    agg["n_seeds"] = len(runs)
    first = runs[0]
    if "detectmap_shape" in first:
        agg["detectmap_shape"] = first["detectmap_shape"]
    return agg


def fmt_mean_std(mean, std, nd=4):
    """Format 'mean ± std' for a summary cell."""
    import numpy as np

    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "nan"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def run_arch(arch: str, food_types: list, output_dir: str, extra_kwargs: dict,
             cooldown: int = 0, n_seeds: int = 1) -> dict:
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
        per_seed = []
        try:
            for s in range(n_seeds):
                # Merge architecture-specific kwargs with common kwargs
                kwargs = {**extra_kwargs.get("common", {}),
                          **extra_kwargs.get(arch, {}),
                          "anomaly_dir": anomaly_dir,
                          "seed": s}
                result = bench_fn(ft, **kwargs)
                per_seed.append(result)
                print(f"  [{arch}] {ft} — seed {s} done")
            arch_results[ft] = aggregate_runs(per_seed)
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


def _get_metric(r, key, subkey=None):
    """Fetch mean and std for a metric, handling nested 'original_model' dicts."""
    if r is None:
        return None, None
    src = r.get("original_model") if ("original_model" in r and subkey is None) or subkey == "original_model" else r
    if not isinstance(src, dict):
        return None, None
    mean = src.get(key)
    if mean is None:
        return None, None
    return float(mean), float(src.get(f"{key}_std", 0.0))


def _metric_table(title, all_results, food_types, key, nd=4, nested=False, sort=True, arrow=""):
    """Print a table of mean ± std per food type for a scalar metric."""
    archs = list(all_results.keys())
    col = 22
    header = f"{'Architecture':<20}"
    for ft in food_types:
        header += f"  {ft:>{col}}"
    header += f"  {'Average':>{col}}"
    print(f"\n{arrow} {title}:")
    print(header)
    print("─" * len(header))

    avgs = {}
    for arch in archs:
        means = []
        for ft in food_types:
            m, _ = _get_metric(r=all_results[arch].get(ft), key=key, subkey=("original_model" if nested else None))
            if m is not None and not np.isnan(m):
                means.append(m)
        avgs[arch] = np.mean(means) if means else float("nan")

    ordered = sorted(avgs, key=lambda a: avgs[a], reverse=True) if sort else archs
    for arch in ordered:
        name = ARCH_DISPLAY_NAME.get(arch, arch)
        row = f"{name:<20}"
        for ft in food_types:
            m, s = _get_metric(all_results[arch].get(ft), key, subkey=("original_model" if nested else None))
            val = fmt_mean_std(m, s, nd) if m is not None else "N/A"
            row += f"  {val:>{col}}"
        avg_str = f"{avgs[arch]:>.4f}" if not np.isnan(avgs[arch]) else "nan"
        row += f"  {avg_str:>{col}}"
        print(row)
    return avgs


def _res_cell(all_results, arch, key, div=1.0, nd=2):
    """Aggregate a resource metric's mean ± std across foods."""
    means, stds = [], []
    for r in all_results[arch].values():
        if not r or r.get(key) is None:
            continue
        means.append(float(r[key]) / div)
        stds.append(float(r.get(f"{key}_std", 0.0)) / div)
    if not means:
        return "nan"
    return f"{np.mean(means):>{nd + 4}.{nd}f} ± {np.mean(stds):.{nd}f}"


def _n_seeds(all_results):
    for results in all_results.values():
        for r in results.values():
            if r and r.get("n_seeds") is not None:
                return int(r["n_seeds"])
    return 1


def print_summary(all_results: dict):
    """Print a combined cross-architecture summary table (mean ± std)."""
    food_types = sorted({ft for results in all_results.values() for ft in results})
    archs = list(all_results.keys())

    print(f"\n{'=' * 80}\n{'FINAL CROSS-ARCHITECTURE SUMMARY':^80}\n{'=' * 80}")
    print("Cells are mean ± std; architecture averages sorted by mean.")

    _metric_table("ROC-AUC Scores", all_results, food_types, "roc_auc", arrow="📊")
    _metric_table("PR-AUC Scores", all_results, food_types, "pr_auc", arrow="📊")
    _metric_table("F1 @ 95% TNR", all_results, food_types, "f1_tnr95", arrow="📊")
    _metric_table("Accuracy @ 95% TNR", all_results, food_types, "acc_tnr95", arrow="📊")

    # Resource table: VRAM / params / FLOPs / inference time
    col = 14
    print(f"\n🖥  Resources (mean ± std across foods, {_n_seeds(all_results)} seeds):")
    res_header = f"{'Architecture':<20}  {'Infer s':>{col}}  {'VRAM GB':>{col}}  {'Params M':>{col}}  {'GFLOPs':>{col}}"
    print(res_header)
    print("─" * len(res_header))
    for arch in archs:
        name = ARCH_DISPLAY_NAME.get(arch, arch)
        infer = _res_cell(all_results, arch, "infer_time_sec")
        vram = _res_cell(all_results, arch, "max_vram_gb")
        params = _res_cell(all_results, arch, "n_params", div=1e6)
        flops = _res_cell(all_results, arch, "gflops")
        print(f"{name:<20}  {infer:>{col}}  {vram:>{col}}  {params:>{col}}  {flops:>{col}}")

    print()


def plot_curves(all_results: dict, output_dir: str):
    """Plot ROC/PR curves, anomaly maps, and box plots per food type."""
    food_types = sorted({ft for results in all_results.values() for ft in results})
    archs = list(all_results.keys())
    archs = [a for a in archs if a != 'our'] + ([a for a in archs if a == 'our'] if 'our' in archs else [])
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
            name = ARCH_DISPLAY_NAME.get(arch, arch)
            ax.plot(r["fpr"], r["tpr"], color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.4f})")
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
            name = ARCH_DISPLAY_NAME.get(arch, arch)
            ax.plot(r["recall"], r["precision"], color=color, lw=1.5, label=f"{name} (AP={pr_auc:.4f})")
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
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(scores, cmap='hot', aspect='auto')
            plt.colorbar(im, ax=ax, label='Anomaly Score')
            name = ARCH_DISPLAY_NAME.get(arch, arch)
            ax.set_title(f'{name} — {ft} Anomaly Map')
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
                scores = np.load(score_path).flatten()
                gt_flat = gt.flatten() if gt.shape == scores.shape else gt.flatten()[:len(scores)]
                mask = gt_flat[:len(scores)]
                normal_scores = scores[~mask]
                anomaly_scores = scores[mask]
                if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                    s_min, s_max = scores.min(), scores.max()
                    if s_max > s_min:
                        scores_norm = (scores - s_min) / (s_max - s_min)
                    else:
                        scores_norm = np.zeros_like(scores)
                    data_normal.append(scores_norm[~mask])
                    data_anomaly.append(scores_norm[mask])
                    positions.append(i)
                    labels_list.append(ARCH_DISPLAY_NAME.get(arch, arch))
            if data_normal:
                bp1 = ax.boxplot(data_normal, positions=[p - 0.2 for p in positions],
                                 widths=0.3, patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='steelblue', alpha=0.7),
                                 medianprops=dict(color='white'))
                bp2 = ax.boxplot(data_anomaly, positions=[p + 0.2 for p in positions],
                                 widths=0.3, patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='crimson', alpha=0.7),
                                 medianprops=dict(color='white'))
                ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Normal', 'Anomaly'], loc='upper right')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels_list, fontsize=8)
                ax.set_xlabel('Architecture')
                ax.set_ylabel('Anomaly Score (min-max scaled)')
                ax.set_title(f'Score Distribution — {ft}')
                ax.set_ylim(-0.02, 0.4)
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Skip running, load existing JSON results and regenerate plots only",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="N",
        help="Number of seeds to average results over (default: 5)",
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

    # ── plot-only mode: load existing results and regenerate plots ─────
    if args.plot:
        print(f"\n{'=' * 80}")
        print(f"  Plot-only mode — loading existing results from {args.output_dir}")
        print(f"{'=' * 80}")

        combined_path = os.path.join(args.output_dir, "benchmark_all.json")
        if os.path.isfile(combined_path):
            with open(combined_path) as f:
                all_results = json.load(f)
            print(f"  ✅ Loaded {combined_path}")
        else:
            all_results = {}
            for arch in ALL_ARCHS:
                result_path = os.path.join(args.output_dir, ARCH_RESULT_FILE[arch])
                if os.path.isfile(result_path):
                    with open(result_path) as f:
                        loaded = json.load(f)
                    stitched = {ft: loaded[ft] for ft in food_types if ft in loaded}
                    if stitched:
                        all_results[arch] = stitched
            print(f"  ✅ Stitched {len(all_results)} architectures from per-arch JSONs")

        print_summary(all_results)
        plot_curves(all_results, args.output_dir)
        return

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
    print(f"  Seeds         : {args.seeds}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'=' * 80}")

    # ── run ─────────────────────────────────────────────────────────────
    all_results = {}
    total_start = time.time()

    for arch in archs:
        arch_start = time.time()
        all_results[arch] = run_arch(arch, food_types, args.output_dir, extra_kwargs,
                                     args.cooldown, n_seeds=args.seeds)
        elapsed = time.time() - arch_start
        print(f"\n⏱  [{arch}] total wall time: {elapsed:.1f}s")

        # VRAM clear between architectures
        if arch != archs[-1]:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if args.cooldown > 0:
                print(f"  Architecture switch: cooling down GPU for {args.cooldown}s...")
                time.sleep(args.cooldown)

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall time: {total_elapsed:.1f}s")

    # ── stitch existing results for architectures not re-run ──────────────
    for candidate_arch in ALL_ARCHS:
        if candidate_arch not in all_results:
            result_path = os.path.join(args.output_dir, ARCH_RESULT_FILE[candidate_arch])
            if os.path.isfile(result_path):
                with open(result_path) as f:
                    loaded = json.load(f)
                stitched = {ft: loaded[ft] for ft in food_types if ft in loaded}
                if stitched:
                    all_results[candidate_arch] = stitched
                    print(f"  📎 [{candidate_arch}] stitched from {result_path}")

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

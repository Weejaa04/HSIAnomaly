#!/usr/bin/env python3
"""
noise.py — Test model robustness against salt & pepper noise at 5%, 10%, 20%.

Compares performance on clean data vs multiple levels of salt & pepper noise for
all architectures. Loads clean results from existing benchmark output, then
runs each architecture with each noise level and compares.

Usage:
    python noise.py
    python noise.py --food Almond
    python noise.py --only gthad otad
    python noise.py --dry-run
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import re
import contextlib
import numpy as np


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
NOISE_ORDER = [20, 10, 5]
MAIN_ARCHS = [
    "bocknet",
    "our",
    "pa2e",
    "gthad",
    "superad",
    "sglnet",
    "autoad",
    "otad",
    "dms2f",
]
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
}
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
}
DATASET_DIR = "AnomalyonFood/Dataset"


NOISE_SEED = 42


def generate_noise_data(food_types, noise_percentages=None, seed=None):
    """Generate data_{pct}.hdr + data_{pct} for each noise percentage if missing."""
    if noise_percentages is None:
        noise_percentages = NOISE_ORDER
    if seed is None:
        seed = NOISE_SEED
    np.random.seed(seed)
    import spectral.io.envi as envi

    for ft in food_types:
        for split in ["Train", "Test"]:
            for pct in noise_percentages:
                hdr_path = os.path.join(DATASET_DIR, ft, split, "data.hdr")
                noisy_hdr = os.path.join(DATASET_DIR, ft, split, f"data_{pct}.hdr")
                noisy_dat = os.path.join(DATASET_DIR, ft, split, f"data_{pct}")

                mask_path = os.path.join(DATASET_DIR, ft, split, f"data_{pct}.noise_mask.npy")

                if os.path.exists(noisy_hdr) and os.path.exists(noisy_dat) and os.path.exists(mask_path):
                    print(f"  ✅ {ft}/{split} data_{pct}.hdr: already exists")
                    continue

                print(f"  Generating noisy data for {ft}/{split} data_{pct}.hdr ({pct}%)...")
                img = envi.open(hdr_path)
                data = np.array(img.load(), dtype=np.float32)

                h, w, bands = data.shape
                total = h * w
                num_noisy = int(total * (pct / 100))
                num_salt = num_noisy // 2

                noisy = data.copy()
                coords = np.random.choice(total, num_noisy, replace=False)
                for b in range(bands):
                    flat = noisy[:, :, b].ravel()
                    flat[coords[:num_salt]] = 1.0
                    flat[coords[num_salt:]] = 0.0
                    noisy[:, :, b] = flat.reshape(h, w)
                noise_mask = np.zeros(h * w, dtype=bool)
                noise_mask[coords] = True
                aggregate_mask = noise_mask.reshape(h, w)

                envi.save_image(noisy_hdr, noisy, dtype=np.float32,
                                metadata=img.metadata, ext="", force=True)
                np.save(mask_path, aggregate_mask.astype(np.uint8))
                print(f"  ✅ {ft}/{split} data_{pct}.hdr: saved")


@contextlib.contextmanager
def swap_to_noisy(food_types, noise_pct):
    """Swap data.hdr ↔ data_{pct}.hdr using .bak files in the same directory."""
    swaps = []
    try:
        for ft in food_types:
            for split in ["Train", "Test"]:
                base = os.path.join(DATASET_DIR, ft, split)
                clean_hdr = os.path.join(base, "data.hdr")
                clean_raw = os.path.join(base, "data.raw")
                noisy_hdr = os.path.join(base, f"data_{noise_pct}.hdr")
                noisy_raw = os.path.join(base, f"data_{noise_pct}")

                if not (os.path.exists(noisy_hdr) and os.path.exists(noisy_raw)):
                    continue

                os.rename(clean_hdr, clean_hdr + ".clean_bak")
                if os.path.exists(clean_raw):
                    os.rename(clean_raw, clean_raw + ".clean_bak")
                os.rename(noisy_hdr, clean_hdr)
                os.rename(noisy_raw, clean_raw)
                swaps.append((ft, split))
                print(f"  📦 {ft}/{split}: data.hdr → .clean_bak, data_{noise_pct}.hdr → data.hdr")

        yield
    finally:
        for ft, split in reversed(swaps):
            base = os.path.join(DATASET_DIR, ft, split)
            clean_hdr = os.path.join(base, "data.hdr")
            clean_raw = os.path.join(base, "data.raw")
            noisy_hdr = os.path.join(base, f"data_{noise_pct}.hdr")
            noisy_raw = os.path.join(base, f"data_{noise_pct}")

            os.rename(clean_hdr, noisy_hdr)
            os.rename(clean_raw, noisy_raw)
            os.rename(clean_hdr + ".clean_bak", clean_hdr)
            os.rename(clean_raw + ".clean_bak", clean_raw)
            print(f"  🔙 {ft}/{split}: restored")


def run_arch_noise(arch, food_types, output_dir, extra_kwargs, cooldown=0):
    """Import and call benchmark_food_type for each food type with noisy data."""
    print(f"\n{'#' * 80}\n# ARCHITECTURE: {arch:20} | NOISY\n{'#' * 80}")

    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    arch_results = {}
    for fi, ft in enumerate(food_types):
        try:
            kwargs = {
                **extra_kwargs.get("common", {}),
                **extra_kwargs.get(arch, {}),
            }
            result = bench_fn(ft, **kwargs)
            arch_results[ft] = result
        except Exception as e:
            print(f"\n❌ [{arch}] [noisy] Error on {ft}: {e}")
            traceback.print_exc()

        if fi < len(food_types) - 1:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if cooldown > 0:
                print(f"  Cooling down GPU for {cooldown}s...")
                time.sleep(cooldown)

    return arch_results


def print_comparison_summary(all_results):
    """Print comparison table: clean vs noisy metrics per food type for all noise levels."""
    archs = sorted(set(k.split("|")[0].strip() for k in all_results.keys()))
    food_types = sorted(set(k.split("|")[2].strip() for k in all_results.keys() if "|" in k))

    if not food_types:
        print("No results to display")
        return

    conditions = set()
    for k in all_results.keys():
        if "|" in k:
            conditions.add(k.split("|")[1])
    noise_levels = sorted([c for c in conditions if c.startswith("noisy_")], reverse=True)

    col = 14

    print(f"\n{'=' * 120}\n{'NOISE ROBUSTNESS ANALYSIS SUMMARY':^120}\n{'=' * 120}")

    for ft in food_types:
        print(f"\nFood Type: {ft}\n")

        print(f"{'Architecture':<20}", end="")
        print(f"  {'Condition':<10}", end="")
        print(f"  {'PR-AUC':>{col}}", end="")
        print(f"  {'ROC-AUC':>{col}}", end="")
        print(f"  {'PR Δ':>{col}}", end="")
        print(f"  {'ROC Δ':>{col}}")
        print("─" * 120)

        for arch in sorted(archs):
            clean_key = f"{arch}|clean|{ft}"

            def get_metrics(key):
                r = all_results.get(key)
                if r is None:
                    return None, None
                if isinstance(r, dict) and "original_model" in r:
                    return r["original_model"].get("pr_auc"), r["original_model"].get("roc_auc")
                return r.get("pr_auc"), r.get("roc_auc")

            clean_pr, clean_roc = get_metrics(clean_key)

            name = ARCH_DISPLAY_NAME.get(arch, arch)

            def fmt(val):
                return f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"

            def delta_str(clean, noisy):
                if clean is None or noisy is None or np.isnan(clean) or np.isnan(noisy):
                    return "N/A"
                d = noisy - clean
                return f"{d:+.4f}"

            print(f"{name:<20}  {'clean':<10}  {fmt(clean_pr):>{col}}  {fmt(clean_roc):>{col}}  {'—':>{col}}  {'—':>{col}}")

            for noise_level in noise_levels:
                noisy_key = f"{arch}|{noise_level}|{ft}"
                noisy_pr, noisy_roc = get_metrics(noisy_key)
                label = noise_level.replace("noisy_", "n=")
                print(f"{name:<20}  {label:<10}  {fmt(noisy_pr):>{col}}  {fmt(noisy_roc):>{col}}  {delta_str(clean_pr, noisy_pr):>{col}}  {delta_str(clean_roc, noisy_roc):>{col}}")

    print()


def plot_noise_1x3(arch, food_type, noise_pct, output_dir, anomaly_dir):
    """Generate 1x3 combo + individual plots: GT, Noise Mask, Predicted Label."""
    import matplotlib.pyplot as plt

    base_path = os.path.join(DATASET_DIR, food_type, 'Test')

    gt_path = os.path.join(base_path, 'label.npy')
    if not os.path.exists(gt_path):
        print(f"  ⚠️  1x3: GT not found at {gt_path}")
        return
    gt = np.load(gt_path)
    gt_binary = (gt != 2).astype(int)

    mask_path = os.path.join(base_path, f'data_{noise_pct}.noise_mask.npy')
    if not os.path.exists(mask_path):
        print(f"  ⚠️  1x3: noise mask not found at {mask_path}")
        return
    noise_mask = np.load(mask_path)

    predicted_path = os.path.join(anomaly_dir, f'{arch}_{food_type}_scores.npy')
    if not os.path.exists(predicted_path):
        print(f"  ⚠️  1x3: scores not found at {predicted_path}")
        return
    predicted = np.load(predicted_path)

    out_dir = os.path.join(output_dir, str(noise_pct))
    os.makedirs(out_dir, exist_ok=True)
    arch_name = ARCH_DISPLAY_NAME.get(arch, arch)

    # Individual plots
    for name, arr, cmap in [('gt', gt_binary, 'hot'), ('noise_mask', noise_mask, 'gray'), ('predicted', predicted, 'hot')]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(arr, cmap=cmap, aspect='auto')
        ax.axis('off')
        save_path = os.path.join(out_dir, f'{name}_{arch}_{food_type}_{noise_pct}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 1x3 combo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_binary, cmap='hot', aspect='auto')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(noise_mask, cmap='gray', aspect='auto')
    axes[1].set_title(f'Noise Mask ({noise_pct}%)')
    axes[1].axis('off')

    axes[2].imshow(predicted, cmap='hot', aspect='auto')
    axes[2].set_title(f'Predicted ({arch_name})')
    axes[2].axis('off')

    save_path = os.path.join(out_dir, f'noise_1x3_{arch}_{food_type}_{noise_pct}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🖼️  plots saved: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model performance: clean vs 5%, 10%, 20% salt & pepper noise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python noise.py                          # Run all architectures with all noise levels
  python noise.py --food Almond            # Test on Almond only
  python noise.py --only gthad otad        # Run only GT-HAD and OTAD
  python noise.py --dry-run                # Validate pipeline (1 epoch/iter per arch)
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
        help=f"Run only these architectures: {MAIN_ARCHS}",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        metavar="ARCH",
        help="Skip these architectures",
    )
    parser.add_argument(
        "--level",
        nargs="*",
        type=int,
        default=None,
        metavar="PCT",
        help=f"Noise levels to run: {NOISE_ORDER} (default: all)",
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
        default="./results/noise",
        help="Directory for JSON results (default: ./results/noise)",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="./results",
        help="Directory with existing benchmark results to reuse for clean data (default: ./results)",
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
        help="Inference only with saved weights (no training). Generates scores and plots.",
    )
    args = parser.parse_args()

    food_types = args.food if (args.food and args.food != ["all"]) else ALL_FOOD_TYPES
    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            sys.exit(1)

    if args.dry_run:
        food_types = ["Almond"]

    archs = MAIN_ARCHS
    if args.only:
        archs = [a for a in MAIN_ARCHS if a in args.only]
    if args.skip:
        archs = [a for a in archs if a not in args.skip]
    if not archs:
        print("❌ No architectures selected.")
        sys.exit(1)

    # Filter noise levels
    if args.level:
        for lvl in args.level:
            if lvl not in NOISE_ORDER:
                print(f"❌ Unknown noise level: {lvl}. Available: {NOISE_ORDER}")
                sys.exit(1)
        noise_order = [l for l in NOISE_ORDER if l in args.level]
    else:
        noise_order = list(NOISE_ORDER)

    # ── plot mode: inference only, no training ────────────────────────
    if args.plot:
        print(f"\n{'=' * 100}")
        print(f"  Plot mode — inference only with saved weights")
        print(f"{'=' * 100}")
        args.retrain = "no"

    extra_kwargs = {
        "common": {
            "dry_run": args.dry_run,
            "retrain": (args.retrain == "yes"),
        },
    }

    dry_run_str = " [DRY RUN]" if args.dry_run else ""

    print(f"\n{'=' * 100}")
    print(f"  HSI Food Anomaly - Noise Robustness Test (5%, 10%, 20% Salt & Pepper){dry_run_str}")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"  Food types    : {', '.join(food_types)}")
    print(f"  Noise order   : {', '.join(f'{pct}%' for pct in noise_order)} (highest first)")
    print(f"  Retrain       : {args.retrain}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Benchmark dir : {args.benchmark_dir} (reused for clean results)")
    print(f"{'=' * 100}")

    # Generate all noise levels if missing
    print(f"\n{'─' * 60}")
    print("  Ensuring noisy data exists...")
    generate_noise_data(food_types)
    print(f"{'─' * 60}")

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    all_results = {}
    total_start = time.time()

    # Load clean results once (for all architectures, not just selected ones)
    for arch in MAIN_ARCHS:
        clean_path = os.path.join(args.benchmark_dir, ARCH_RESULT_FILE[arch])
        if os.path.isfile(clean_path):
            with open(clean_path) as f:
                clean_results = json.load(f)
            print(f"  ✅ [{arch}] clean results loaded from {clean_path}")
            for ft, result in clean_results.items():
                if ft in food_types:
                    all_results[f"{arch}|clean|{ft}"] = result
        else:
            print(f"  ⚠️  [{arch}] no clean results found at {clean_path}")
            for ft in food_types:
                all_results[f"{arch}|clean|{ft}"] = None

    # Run noise tests: highest noise first (20% → 10% → 5%)
    for noise_pct in noise_order:
        pct_dir = os.path.join(args.output_dir, str(noise_pct))
        os.makedirs(pct_dir, exist_ok=True)

        anomaly_dir = os.path.join(pct_dir, 'anomaly_maps')
        os.makedirs(anomaly_dir, exist_ok=True)
        extra_kwargs["common"]["anomaly_dir"] = anomaly_dir

        print(f"\n{'#' * 100}")
        print(f"# NOISE LEVEL: {noise_pct}% Salt & Pepper{' ' * 50}")
        print(f"{'#' * 100}")

        for arch in archs:
            arch_start = time.time()

            if args.retrain == "yes":
                extra_kwargs[arch] = {"noise_suffix": f"_{noise_pct}"}

            print(f"\n  🔄 [{arch}] swapping to data_{noise_pct}.hdr ({noise_pct}%)...")
            with swap_to_noisy(food_types, noise_pct):
                noisy_results = run_arch_noise(arch, food_types, args.output_dir, extra_kwargs, args.cooldown)
            print(f"  ✅ [{arch}] restored clean data")

            # Generate 1x3 visualization: GT, Noise Mask, Predicted Label
            for ft in food_types:
                plot_noise_1x3(arch, ft, noise_pct, args.output_dir, anomaly_dir)

            for ft, result in noisy_results.items():
                all_results[f"{arch}|noisy_{noise_pct}|{ft}"] = result

            # Save per-architecture noise results
            arch_path = os.path.join(pct_dir, f"{arch}.json")
            with open(arch_path, "w") as f:
                json.dump(noisy_results, f, indent=2, default=convert_numpy)
            print(f"  💾 {arch} results → {arch_path}")

            elapsed = time.time() - arch_start
            print(f"\n⏱  [{arch}] {noise_pct}% noise wall time: {elapsed:.1f}s")

            if arch != archs[-1]:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                if args.cooldown > 0:
                    print(f"  Architecture switch: cooling down GPU for {args.cooldown}s...")
                    time.sleep(args.cooldown)

        # Save combined results for this noise level (clean + noisy_{pct})
        combined = {}
        for k, v in all_results.items():
            parts = k.split("|", 2)
            if len(parts) == 3:
                cond = parts[1]
                if cond == "clean" or cond == f"noisy_{noise_pct}":
                    combined[k] = v
        combined_path = os.path.join(pct_dir, "noise_all.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=convert_numpy)
        print(f"  💾 Combined → {combined_path}")

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall time: {total_elapsed:.1f}s")

    # ── stitch existing noisy results for architectures not re-run ────────
    for candidate_arch in MAIN_ARCHS:
        if candidate_arch in archs:
            continue
        all_path = os.path.join(args.output_dir, "all.json")
        if os.path.isfile(all_path):
            with open(all_path) as f:
                prev_results = json.load(f)
            stitched = False
            for noise_pct in noise_order:
                for ft in food_types:
                    key = f"{candidate_arch}|noisy_{noise_pct}|{ft}"
                    if key in prev_results and key not in all_results:
                        all_results[key] = prev_results[key]
                        stitched = True
            if stitched:
                print(f"  📎 [{candidate_arch}] noise results stitched from {all_path}")

    # ── per-architecture totals ──────────────────────────────────────────
    for arch in archs:
        arch_total = {}
        for k, v in all_results.items():
            if k.startswith(f"{arch}|"):
                arch_total[k] = v
        arch_path = os.path.join(args.output_dir, f"{arch}.json")
        with open(arch_path, "w") as f:
            json.dump(arch_total, f, indent=2, default=convert_numpy)
        print(f"  💾 {arch} total → {arch_path}")

    # ── all results combined ─────────────────────────────────────────────
    all_path = os.path.join(args.output_dir, "all.json")
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"  💾 All results → {all_path}")

    print_comparison_summary(all_results)


if __name__ == "__main__":
    main()

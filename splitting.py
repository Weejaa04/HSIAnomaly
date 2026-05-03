#!/usr/bin/env python3
"""
splitting.py — Compare Spatial Guillotine vs Random Sampling across architectures.

This script leverages the benchmark infrastructure to run each architecture twice:
once with spatial guillotine splitting (fixed row blocks with dead zone) and once
with random sampling (37.5% train, 12.5% val). The output quantifies information
leakage in random sampling by comparing metric gaps.

Usage:
    python splitting.py
    python splitting.py --food Almond
    python splitting.py --only gthad otad
    python splitting.py --dry-run
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import numpy as np

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]
# Main architectures only (skip ablation, sglnet)
MAIN_ARCHS = [
    "bocknet",
    "our",
    "pa2e",
    "gthad",
    "superad",
    "autoad",
    "otad",
]
ARCH_MODULE_MAP = {
    "bocknet": "scripts.bocknet.__main__",
    "our": "scripts.our.__main__",
    "pa2e": "scripts.pa2e.__main__",
    "gthad": "scripts.gthad.__main__",
    "superad": "scripts.superad.__main__",
    "autoad": "scripts.autoad.__main__",
    "otad": "scripts.otad.__main__",
}


def run_arch_with_split(
    arch: str, food_types: list, split_method: str, output_dir: str, extra_kwargs: dict
) -> dict:
    """
    Import and call benchmark_food_type for each food type with a specific split method.
    
    Args:
        arch: Architecture name
        food_types: List of food types to benchmark
        split_method: "spatial" or "random"
        output_dir: Output directory for results
        extra_kwargs: Additional kwargs to pass
    
    Returns:
        Dictionary of results {food_type: metrics}
    """
    print(
        f"\n{'#' * 80}\n# ARCHITECTURE: {arch:20} | SPLIT_METHOD: {split_method:10}\n{'#' * 80}"
    )

    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    arch_results = {}
    for ft in food_types:
        try:
            kwargs = {
                **extra_kwargs.get("common", {}),
                **extra_kwargs.get(arch, {}),
                "split_method": split_method,
            }
            result = bench_fn(ft, **kwargs)
            arch_results[ft] = result
        except Exception as e:
            print(f"\n❌ [{arch}] [{split_method}] Error on {ft}: {e}")
            traceback.print_exc()

    return arch_results


def print_comparison_summary(all_results: dict):
    """
    Print a comparison table showing metrics for both spatial and random splits.
    
    Format:
    | Architecture | Split Method | PR-AUC (Almond) | ROC-AUC (Almond) |
    """
    archs = sorted(set(k.split("|")[0].strip() for k in all_results.keys()))
    food_types = sorted(
        set(k.split("|")[1].strip() for k in all_results.keys() if "|" in k)
    )

    if not food_types:
        print("No results to display")
        return

    # Pick the first food type for summary (or user can modify)
    ft = food_types[0]

    col = 14

    print(f"\n{'=' * 100}\n{'SPLITTING ANALYSIS SUMMARY':^100}\n{'=' * 100}")
    print(f"\nFood Type: {ft}\n")

    print(f"{'Architecture':<20}", end="")
    print(f"  {'Split Method':<15}", end="")
    print(f"  {'PR-AUC':>{col}}", end="")
    print(f"  {'ROC-AUC':>{col}}")
    print("─" * 100)

    for arch in sorted(archs):
        for split_method in ["spatial", "random"]:
            key = f"{arch}|{split_method}|{ft}"
            if key in all_results:
                r = all_results[key]
                pr = r.get("pr_auc", float("nan"))
                roc = r.get("roc_auc", float("nan"))
                
                # Handle cases where result has nested structure
                if isinstance(r, dict) and "original_model" in r:
                    pr = r["original_model"].get("pr_auc", float("nan"))
                    roc = r["original_model"].get("roc_auc", float("nan"))

                pr_str = f"{pr:.4f}" if not np.isnan(pr) else "N/A"
                roc_str = f"{roc:.4f}" if not np.isnan(roc) else "N/A"

                print(f"{arch:<20}  {split_method:<15}  {pr_str:>{col}}  {roc_str:>{col}}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Spatial Guillotine vs Random Sampling splitting methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python splitting.py                          # Run all main architectures with both methods
  python splitting.py --food Almond            # Test on Almond only
  python splitting.py --only gthad otad        # Run only GT-HAD and OTAD
  python splitting.py --dry-run                # Validate pipeline (1 epoch/iter per arch)
  python splitting.py --retrain no             # Load saved models without retraining
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
        default="./results/splitting",
        help="Directory for JSON results (default: ./results/splitting)",
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
    archs = MAIN_ARCHS
    if args.only:
        archs = [a for a in MAIN_ARCHS if a in args.only]
    if args.skip:
        archs = [a for a in archs if a not in args.skip]
    if not archs:
        print("❌ No architectures selected.")
        sys.exit(1)

    # ── split methods ───────────────────────────────────────────────────
    split_methods = ["spatial", "random"]

    # ── build extra kwargs ───────────────────────────────────────────────
    extra_kwargs = {
        "common": {
            "dry_run": args.dry_run,
            "retrain": (args.retrain == "yes"),
        },
    }

    dry_run_str = " [DRY RUN]" if args.dry_run else ""

    print(f"\n{'=' * 100}")
    print(f"  HSI Food Anomaly - Splitting Analysis (Guillotine vs Random){dry_run_str}")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"  Food types    : {', '.join(food_types)}")
    print(f"  Split methods : {', '.join(split_methods)}")
    print(f"  Retrain       : {args.retrain}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'=' * 100}")

    # ── run ─────────────────────────────────────────────────────────────
    all_results = {}
    total_start = time.time()

    for arch in archs:
        arch_start = time.time()
        for split_method in split_methods:
            arch_results = run_arch_with_split(
                arch, food_types, split_method, args.output_dir, extra_kwargs
            )
            # Store with combined key: arch|split_method|food_type
            for ft, result in arch_results.items():
                key = f"{arch}|{split_method}|{ft}"
                all_results[key] = result

        elapsed = time.time() - arch_start
        print(f"\n⏱  [{arch}] total wall time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall time: {total_elapsed:.1f}s")

    # ── combined JSON ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    combined_path = os.path.join(args.output_dir, "splitting_comparison.json")
    with open(combined_path, "w") as f:
        # Handle numpy serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"✅ Combined results saved → {combined_path}")

    print_comparison_summary(all_results)


if __name__ == "__main__":
    main()

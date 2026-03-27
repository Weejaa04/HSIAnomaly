#!/usr/bin/env python3
"""
ablation.py — Run ablation study for the 'our' architecture.

Ablation axes:
  1. HSI Calibration  : mean | mean_per_column (★) | none
  2. Encoder type     : cnn (★) | linear
  3. Loss function    : mse (★) | mae
  4. Scoring method   : nnmb (★) | dsvdd
  5. Partial windows  : partial (★) | non_partial_cnn | non_partial_linear

(★) = baseline/current setting

Usage:
    python ablation.py
    python ablation.py --food Almond
    python ablation.py --dry-run
    python ablation.py --retrain no
    python ablation.py --output-dir ./results/ablation
"""

import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import importlib

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]

# ── Ablation axis groupings (for table printing) ─────────────────────────────
ABLATION_GROUPS = {
    "1. HSI Calibration": [
        ("calib_mean",            "Mean"),
        ("calib_mean_per_column", "Mean per column ★"),
        ("calib_none",            "No normalization"),
    ],
    "2. Encoder Type": [
        ("encoder_cnn",    "CNN (Conv1D) ★"),
        ("encoder_linear", "Linear"),
    ],
    "3. Loss Function": [
        ("loss_mse", "MSE ★"),
        ("loss_mae", "MAE"),
    ],
    "4. Scoring Method": [
        ("scoring_nnmb",  "Nearest-Neighbour Memory Bank ★"),
        ("scoring_dsvdd", "DSVDD center distance"),
    ],
    "5. Partial Windows": [
        ("partial_yes",        "Partial ★"),
        ("partial_no_cnn",     "Non-Partial (CNN)"),
        ("partial_no_linear",  "Non-Partial (Linear)"),
    ],
}


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_ablation(food_types: list, output_dir: str, dry_run: bool, retrain: bool) -> dict:
    """Run full ablation over all food types. Returns nested dict: variant → food → result."""

    module = importlib.import_module("scripts.ablation.__main__")
    bench_fn = module.benchmark_food_type

    # Outer key = food type; inner key = variant name → result
    food_results: dict = {}

    for ft in food_types:
        print(f"\n{'#'*80}\n#  FOOD TYPE: {ft}\n{'#'*80}")
        try:
            food_results[ft] = bench_fn(ft, dry_run=dry_run, retrain=retrain)
        except Exception as e:
            print(f"\n❌ Error on {ft}: {e}")
            traceback.print_exc()
            food_results[ft] = {}

    # Reshape: variant → food → result  (for easier per-axis table printing)
    by_variant: dict = {}
    for ft, variants in food_results.items():
        for variant, result in variants.items():
            by_variant.setdefault(variant, {})[ft] = result

    return by_variant


def print_group_table(group_name: str, entries: list, food_types: list, by_variant: dict):
    """Print a summary table for one ablation axis group."""
    col = 16
    print(f"\n{'═'*80}")
    print(f"  Ablation: {group_name}")
    print(f"{'═'*80}")

    for metric_label, metric_key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]:
        print(f"\n  📊 {metric_label}:")
        header = f"  {'Variant':<30}"
        for ft in food_types:
            header += f"  {ft:>{col}}"
        header += f"  {'Avg':>{col}}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for variant_key, variant_label in entries:
            row = f"  {variant_label:<30}"
            scores = []
            for ft in food_types:
                r = by_variant.get(variant_key, {}).get(ft)
                if r is not None:
                    v = r.get(metric_key)
                    if v is not None and not np.isnan(v):
                        scores.append(v)
                        row += f"  {v:>{col}.4f}"
                    else:
                        row += f"  {'nan':>{col}}"
                else:
                    row += f"  {'N/A':>{col}}"
            avg = np.mean(scores) if scores else float("nan")
            avg_str = f"{avg:.4f}" if not np.isnan(avg) else "nan"
            row += f"  {avg_str:>{col}}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for the 'our' HSI food anomaly architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ablation.py
  python ablation.py --food Almond
  python ablation.py --dry-run
  python ablation.py --retrain no --output-dir ./results/ablation
        """,
    )
    parser.add_argument("--food", nargs="*", default=None, metavar="FOOD",
                        help=f"Food types (default: all). Options: {ALL_FOOD_TYPES}")
    parser.add_argument("--dry-run", "-dr", action="store_true",
                        help="Train for 1 iteration/epoch only")
    parser.add_argument("--retrain", type=str, choices=["yes", "no"], default="yes",
                        help="Retrain models or load saved ones (default: yes)")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for JSON results (default: ./results)")
    args = parser.parse_args()

    food_types = args.food if (args.food and args.food != ["all"]) else ALL_FOOD_TYPES
    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            sys.exit(1)

    if args.dry_run:
        food_types = ["Almond"]

    retrain = args.retrain == "yes"
    dry_run_str = " [DRY RUN]" if args.dry_run else ""

    print(f"\n{'='*80}")
    print(f"  HSI Food Anomaly — Ablation Study{dry_run_str}")
    print(f"  Food types : {', '.join(food_types)}")
    print(f"  Retrain    : {args.retrain}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*80}")

    total_start = time.time()

    by_variant = run_ablation(food_types, args.output_dir, args.dry_run, retrain)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ablation_all.json")
    with open(out_path, "w") as f:
        json.dump(by_variant, f, indent=2, default=convert_numpy)
    print(f"\n✅ Ablation results saved → {out_path}")

    total_elapsed = time.time() - total_start
    print(f"⏱  Total wall time: {total_elapsed:.1f}s")

    # ── Print per-axis summary tables ──────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"{'ABLATION SUMMARY':^80}")
    print(f"{'='*80}")

    for group_name, entries in ABLATION_GROUPS.items():
        print_group_table(group_name, entries, food_types, by_variant)

    print()


if __name__ == "__main__":
    main()

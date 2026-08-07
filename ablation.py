#!/usr/bin/env python3
"""
ablation.py — Run ablation study for the 'our' architecture.

Ablation axes:
  1. HSI Calibration  : mean | mean_per_column (★) | none
  2. Encoder type     : cnn (★) | linear
  3. Loss function    : mse (★) | mae
  4. Scoring method   : nnmb (★) | dsvdd
  5. Partial windows  : partial (★) | non_partial_cnn | non_partial_linear
  6. Loss weights     : alpha × beta grid {0.2..1.0 step 0.2}, default α=0.2 × β=1.0 (★)
  7. Post-processing  : median kernel none | 3x3 (★) | 5x5 | 7x7
  8. Training phases  : phase1 + phase2 (★) | phase2 only (no phase 1)

(★) = baseline/current setting

Usage:
    python ablation.py
    python ablation.py --food Almond
    python ablation.py --dry-run
    python ablation.py --retrain no
    python ablation.py --output-dir ./results/ablation
    python ablation.py --group weight
    python ablation.py --group loss
python ablation.py --group post
    python ablation.py --group phase
    python ablation.py --group phase
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

WEIGHT_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]

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
    "6. Loss Weights": [
        ("weight_a0.2_b1.0", "α=0.2 × β=1.0 ★ (default)"),
    ] + [
        (f"weight_a{a:.1f}_b{b:.1f}", f"α={a:.1f} × β={b:.1f}")
        for a in WEIGHT_GRID for b in WEIGHT_GRID if not (a == 0.2 and b == 1.0)
    ],
    "7. Post-Processing": [
        ("pp_zero", "No median smoothing"),
        ("pp_3",    "Median 3x3 ★"),
        ("pp_5",    "Median 5x5"),
        ("pp_7",    "Median 7x7"),
    ],
    "8. Training Phases": [
        ("phase_both",  "Phase 1 + Phase 2 ★"),
        ("phase_no_p1", "No Phase 1 (Phase 2 only)"),
        ("phase_p2_no_center", "Phase 1 + Phase 2 (no center loss)"),
        ("phase_no_p2", "Phase 1 (no Phase 2)"),

    ],
}

# ── Group aliases (short keys accepted by --group) ───────────────────────────
GROUP_ALIASES = {
    "calib":   "1. HSI Calibration",
    "encoder": "2. Encoder Type",
    "loss":    "3. Loss Function",
    "scoring": "4. Scoring Method",
    "partial": "5. Partial Windows",
    "weight":  "6. Loss Weights",
    "post":    "7. Post-Processing",
    "phase":   "8. Training Phases",
}


def resolve_group(name: str) -> str:
    """Map a --group value (alias or full label) to the canonical group name."""
    if name in ABLATION_GROUPS:
        return name
    if name in GROUP_ALIASES:
        return GROUP_ALIASES[name]
    choices = ", ".join([f"{k} ({v})" for k, v in GROUP_ALIASES.items()])
    raise ValueError(f"Unknown group: {name!r}. Available groups: {choices}")


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_ablation(food_types: list, output_dir: str, dry_run: bool, retrain: bool,
                 variants: list = None) -> dict:
    """Run full ablation over all food types. Returns nested dict: variant → food → result.

    variants: optional list of variant names; if given, only those variants run.
    """
    module = importlib.import_module("scripts.ablation.__main__")
    bench_fn = module.benchmark_food_type

    # Outer key = food type; inner key = variant name → result
    food_results: dict = {}

    for ft in food_types:
        print(f"\n{'#'*80}\n#  FOOD TYPE: {ft}\n{'#'*80}")
        try:
            food_results[ft] = bench_fn(ft, dry_run=dry_run, retrain=retrain,
                                        variants=variants)
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
  python ablation.py --group loss
  python ablation.py --group weight
  python ablation.py --group post
  python ablation.py --group phase
  python ablation.py --group "3. Loss Function" --food Almond
        """,
    )
    parser.add_argument("--food", nargs="*", default=None, metavar="FOOD",
                        help=f"Food types (default: all). Options: {ALL_FOOD_TYPES}")
    parser.add_argument("--dry-run", "-dr", action="store_true",
                        help="Train for 1 iteration/epoch only")
    parser.add_argument("--retrain", type=str, choices=["yes", "no"], default="yes",
                        help="Retrain models or load saved ones (default: yes)")
    parser.add_argument("--group", default=None, metavar="GROUP",
                        help=f"Run only one ablation axis group. Aliases: {', '.join(GROUP_ALIASES)} "
                             f"or full labels: {', '.join(ABLATION_GROUPS)}")
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

    # Resolve ablation group (if any) → variant subset
    variants = None
    group_name = None
    if args.group:
        group_name = resolve_group(args.group)
        variants = [v for v, _ in ABLATION_GROUPS[group_name]]
        print(f"\n{'='*80}")
        print(f"  HSI Food Anomaly — Ablation Study{dry_run_str}")
        print(f"  Group      : {group_name}  ({len(variants)} variants)")
        print(f"  Food types : {', '.join(food_types)}")
        print(f"  Retrain    : {args.retrain}")
        print(f"  Output dir : {args.output_dir}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"  HSI Food Anomaly — Ablation Study{dry_run_str}")
        print(f"  Food types : {', '.join(food_types)}")
        print(f"  Retrain    : {args.retrain}")
        print(f"  Output dir : {args.output_dir}")
        print(f"{'='*80}")

    total_start = time.time()

    by_variant = run_ablation(food_types, args.output_dir, args.dry_run, retrain,
                              variants=variants)

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

    groups_to_print = [(group_name, ABLATION_GROUPS[group_name])] if group_name else list(ABLATION_GROUPS.items())
    for g_name, entries in groups_to_print:
        print_group_table(g_name, entries, food_types, by_variant)

    print()


if __name__ == "__main__":
    main()

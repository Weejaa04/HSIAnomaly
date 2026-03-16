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

import argparse
import importlib
import json
import os
import sys
import time
import traceback

ALL_FOOD_TYPES  = ['Almond', 'Pistachio', 'GarlicStems']
ALL_ARCHS       = ['BockNet', 'our', 'pa2e', 'gthad']
ARCH_MODULE_MAP = {
    'BockNet':'scripts.bocknet.__main__',
    'our': 'scripts.our.__main__',
    'pa2e': 'scripts.pa2e.__main__',
    'gthad': 'scripts.gthad.__main__',
}
ARCH_RESULT_FILE = {
    'BockNet':'bocknet.json',
    'our': 'our.json',
    'pa2e': 'pa2e.json',
    'gthad': 'gthad.json',
}


def run_arch(arch: str, food_types: list, output_dir: str, extra_kwargs: dict) -> dict:
    """Import and call benchmark_food_type for each food type in an architecture."""
    print(f"\n{'#'*80}")
    print(f"# ARCHITECTURE: {arch}")
    print(f"{'#'*80}")

    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    arch_results = {}
    for ft in food_types:
        try:
            # Merge architecture-specific kwargs with common kwargs
            kwargs = {**extra_kwargs.get('common', {}), **extra_kwargs.get(arch, {})}
            result = bench_fn(ft, **kwargs)
            arch_results[ft] = result
        except Exception as e:
            print(f"\n❌ [{arch}] Error on {ft}: {e}")
            traceback.print_exc()

    # Save per-architecture JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, ARCH_RESULT_FILE[arch])
    with open(out_path, 'w') as f:
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
    archs      = list(all_results.keys())

    col = 16
    header = f"{'Food Type':<20}"
    for arch in archs:
        header += f"  {arch+' ROC':>{col}} {arch+' PR':>{col}}"
    div = '─' * len(header)

    print(f"\n{'='*80}\n{'FINAL CROSS-ARCHITECTURE SUMMARY':^80}\n{'='*80}")
    print(header)
    print(div)

    for ft in food_types:
        row = f"{ft:<20}"
        for arch in archs:
            r = all_results[arch].get(ft)
            if r is None:
                row += f"  {'N/A':>{col}} {'N/A':>{col}}"
            elif 'original_model' in r:          # ours / our (fused pair)
                roc = r['original_model']['roc_auc']
                pr  = r['original_model']['pr_auc']
                row += f"  {roc:>{col}.4f} {pr:>{col}.4f}"
            else:                                # GT-HAD (single model)
                row += f"  {r['roc_auc']:>{col}.4f} {r['pr_auc']:>{col}.4f}"
        print(row)

    print(div)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Full benchmark: run all HSI food anomaly architectures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py
  python benchmark.py --food Almond
  python benchmark.py --only GT-HAD
  python benchmark.py --skip GT-HAD --output-dir ./results/run1
  python benchmark.py --dry-run                  # train only 1 epoch/iter per architecture
  python benchmark.py --dry-run --retrain no     # load saved models without retraining
        """,
    )
    parser.add_argument('--food',        nargs='*', default=None,
                        metavar='FOOD',  help='Food types (default: all)')
    parser.add_argument('--only',        nargs='*', default=None,
                        metavar='ARCH',  help=f'Run only these architectures: {ALL_ARCHS}')
    parser.add_argument('--skip',        nargs='*', default=None,
                        metavar='ARCH',  help='Skip these architectures')
    parser.add_argument('--dry-run', '-dr',
                        action='store_true',
                        help='Train for 1 iteration/epoch only to validate the script works')
    parser.add_argument('--retrain',
                        type=str, choices=['yes', 'no'], default='yes',
                        help='Retrain models or load saved ones (default: yes)')
    parser.add_argument('--output-dir',  default='./results',
                        help='Directory for JSON results (default: ./results)')
    args = parser.parse_args()

    # ── food types ──────────────────────────────────────────────────────
    food_types = args.food if (args.food and args.food != ['all']) else ALL_FOOD_TYPES
    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            sys.exit(1)
    
    # For dry-run, only test with Almond (faster validation)
    if args.dry_run:
        food_types = ['Almond']

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
        'common': {
            'dry_run': args.dry_run,
            'retrain': (args.retrain == 'yes'),
        },
    }

    dry_run_str = " [DRY RUN]" if args.dry_run else ""

    print(f"\n{'='*80}")
    print(f"  HSI Food Anomaly Benchmark{dry_run_str}")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"  Food types    : {', '.join(food_types)}")
    print(f"  Retrain       : {args.retrain}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'='*80}")

    # ── run ─────────────────────────────────────────────────────────────
    all_results = {}
    total_start = time.time()

    for arch in archs:
        arch_start = time.time()
        all_results[arch] = run_arch(arch, food_types, args.output_dir, extra_kwargs)
        elapsed = time.time() - arch_start
        print(f"\n⏱  [{arch}] total wall time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall time: {total_elapsed:.1f}s")

    # ── combined JSON ────────────────────────────────────────────────────
    combined_path = os.path.join(args.output_dir, 'benchmark_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Combined results saved → {combined_path}")

    print_summary(all_results)


if __name__ == '__main__':
    main()

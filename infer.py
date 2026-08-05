#!/usr/bin/env python3
"""
infer.py — Load pre-trained clean weights and run inference on HSI data across all architectures.

Usage:
    python infer.py                                         # All arches, all food, clean only
    python infer.py --arch our pa2e                         # Specific architectures
    python infer.py --food Almond                           # Specific food type
    python infer.py --noise 5 10 20                         # Also infer on noisy data (clean weights)
    python infer.py --arch our --food Almond --noise 5      # Narrow scope
    python infer.py --output-dir ./results/infer             # Custom output dir
"""

import argparse
import contextlib
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]
DATASET_DIR = "AnomalyonFood/Dataset"

ALL_ARCHS = ["autoad", "bocknet", "dms2f", "gthad", "otad", "our", "pa2e", "sglnet", "superad"]

ARCH_MODULE_MAP = {
    "autoad": "scripts.autoad.__main__",
    "bocknet": "scripts.bocknet.__main__",
    "dms2f": "scripts.dms2f.__main__",
    "gthad": "scripts.gthad.__main__",
    "otad": "scripts.otad.__main__",
    "our": "scripts.our.__main__",
    "pa2e": "scripts.pa2e.__main__",
    "sglnet": "scripts.sglnet.__main__",
    "superad": "scripts.superad.__main__",
}

ARCH_DISPLAY = {
    "autoad": "Auto-AD",
    "bocknet": "BockNet",
    "dms2f": "DMS2F-HAD",
    "gthad": "GT-HAD",
    "otad": "OT-AD",
    "our": "Ours",
    "pa2e": "PA2E",
    "sglnet": "SGLNet",
    "superad": "SuperAD",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@contextlib.contextmanager
def swap_test_data(food_type, noise_pct):
    """Temporarily swap Test/data.hdr with Test/data_{pct}.hdr, restores on exit."""
    base = os.path.join(DATASET_DIR, food_type, 'Test')
    clean_hdr = os.path.join(base, 'data.hdr')
    clean_raw = os.path.join(base, 'data.raw')
    noisy_hdr = os.path.join(base, f'data_{noise_pct}.hdr')
    noisy_raw = os.path.join(base, f'data_{noise_pct}')

    if not os.path.exists(noisy_hdr):
        print(f"  ⚠️  Noisy data not found: {noisy_hdr}")
        yield
        return

    os.rename(clean_hdr, clean_hdr + '.clean_bak')
    if os.path.exists(clean_raw):
        os.rename(clean_raw, clean_raw + '.clean_bak')
    os.rename(noisy_hdr, clean_hdr)
    if os.path.exists(noisy_raw):
        os.rename(noisy_raw, clean_raw)

    try:
        yield
    finally:
        os.rename(clean_hdr, noisy_hdr)
        if os.path.exists(clean_raw):
            os.rename(clean_raw, noisy_raw)
        os.rename(clean_hdr + '.clean_bak', clean_hdr)
        if os.path.exists(clean_raw + '.clean_bak'):
            os.rename(clean_raw + '.clean_bak', clean_raw)


def run_arch_inference(arch, food_types, noise_levels, output_dir, dry_run=False):
    """Run clean (+ optionally noisy) inference for one architecture across food types."""
    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    label = ARCH_DISPLAY.get(arch, arch)
    print(f"\n{'#' * 70}")
    print(f"# {label}")
    print(f"{'#' * 70}")

    arch_dir = os.path.join(output_dir, arch)
    os.makedirs(arch_dir, exist_ok=True)

    all_results = {}

    for ft in food_types:
        # ── Clean inference ──────────────────────────────────────────────
        print(f"\n  ── {ft} (clean) ──")
        try:
            adir = os.path.join(arch_dir, 'tmp', 'clean')
            os.makedirs(adir, exist_ok=True)
            result = bench_fn(ft, dry_run=dry_run, retrain=False, anomaly_dir=adir)
            all_results[f"{ft}|clean"] = result
            _move_scores(adir, arch, ft, 'clean', arch_dir)
            _save_metrics(arch_dir, ft, 'clean', result)
            _print_row(ft, 'clean', result)
        except Exception as e:
            print(f"  ❌ [{arch}] {ft} clean: {e}")
            import traceback; traceback.print_exc()
            all_results[f"{ft}|clean"] = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Noisy inference ─────────────────────────────────────────────
        for noise_pct in noise_levels:
            label_n = f"noisy_{noise_pct}"
            print(f"\n  ── {ft} (noise {noise_pct}%) ──")
            try:
                adir = os.path.join(arch_dir, 'tmp', label_n)
                os.makedirs(adir, exist_ok=True)
                with swap_test_data(ft, noise_pct):
                    result = bench_fn(ft, dry_run=dry_run, retrain=False, anomaly_dir=adir)
                all_results[f"{ft}|{label_n}"] = result
                _move_scores(adir, arch, ft, label_n, arch_dir)
                _save_metrics(arch_dir, ft, label_n, result)
                _print_row(ft, label_n, result)
            except Exception as e:
                print(f"  ❌ [{arch}] {ft} noise {noise_pct}%: {e}")
                import traceback; traceback.print_exc()
                all_results[f"{ft}|{label_n}"] = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    arch_path = os.path.join(arch_dir, 'results.json')
    with open(arch_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"  💾 {arch} results → {arch_path}")

    # Cleanup tmp dirs
    _rmtree(os.path.join(arch_dir, 'tmp'))

    return all_results


def _move_scores(tmp_dir, arch, food_type, label, arch_dir):
    """Move scores from tmp dir to arch root with label-aware name."""
    src = os.path.join(tmp_dir, f'{arch}_{food_type}_scores.npy')
    if os.path.exists(src):
        dst = os.path.join(arch_dir, f'{food_type}_{label}_scores.npy')
        os.makedirs(arch_dir, exist_ok=True)
        os.replace(src, dst)
    src_l = os.path.join(tmp_dir, f'{food_type}_labels.npy')
    if os.path.exists(src_l):
        dst_l = os.path.join(arch_dir, f'{food_type}_{label}_labels.npy')
        os.replace(src_l, dst_l)


def _save_metrics(arch_dir, food_type, label, result):
    if result is None:
        return
    metrics = {k: v for k, v in result.items()
               if k in ('roc_auc', 'pr_auc', 'infer_time_sec', 'n_params', 'max_vram_gb',
                        'fpr', 'tpr', 'precision', 'recall')}
    with open(os.path.join(arch_dir, f'{food_type}_{label}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=convert_numpy)


def _print_row(food_type, label, result):
    if result is None:
        print(f"  [{food_type}] [{label}] ❌ FAILED")
        return
    roc = result.get('roc_auc', 'N/A')
    pr = result.get('pr_auc', 'N/A')
    t = result.get('infer_time_sec', 'N/A')
    if isinstance(roc, float):
        print(f"  [{food_type}] [{label}] ROC-AUC={roc:.4f}, PR-AUC={pr:.4f} | {t:.2f}s")
    else:
        print(f"  [{food_type}] [{label}] ROC-AUC={roc}, PR-AUC={pr}")


def _rmtree(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(
        description="Load pre-trained clean weights and run inference across all architectures."
    )
    parser.add_argument(
        '--arch', '-a', nargs='*', default=None,
        help=f"Architectures (default: all). Choices: {', '.join(ALL_ARCHS)}"
    )
    parser.add_argument(
        '--food', nargs='*', default=None,
        help=f"Food types (default: all). Choices: {', '.join(ALL_FOOD_TYPES)}"
    )
    parser.add_argument(
        '--noise', nargs='*', type=int, default=None,
        help="Noise percentages to also infer on (e.g., 5 10 20)"
    )
    parser.add_argument(
        '--output-dir', default='./results/infer',
        help="Output directory for results (default: ./results/infer)"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Run 1 iteration only for validation"
    )
    args = parser.parse_args()

    archs = args.arch if args.arch else ALL_ARCHS
    food_types = args.food if args.food else ALL_FOOD_TYPES
    noise_levels = sorted(args.noise) if args.noise else []

    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            sys.exit(1)
    for a in archs:
        if a not in ALL_ARCHS:
            print(f"❌ Unknown architecture: {a}. Available: {', '.join(ALL_ARCHS)}")
            sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Archs      : {', '.join(ARCH_DISPLAY.get(a, a) for a in archs)}")
    print(f"Food types : {', '.join(food_types)}")
    print(f"Noise      : {', '.join(f'{p}%' for p in noise_levels) or 'none'}")
    print(f"Output dir : {args.output_dir}")
    print(f"{'=' * 60}")

    all_results = {}

    for arch in archs:
        arch_results = run_arch_inference(
            arch, food_types, noise_levels,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )
        for k, v in arch_results.items():
            all_results[f"{arch}|{k}"] = v

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_path = os.path.join(args.output_dir, 'all_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(all_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"\n✅ All results → {all_path}")
    print(f"   Per-arch results in {args.output_dir}/<arch>/")


if __name__ == '__main__':
    main()

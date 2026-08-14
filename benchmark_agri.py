#!/usr/bin/env python3
"""
benchmark_agri.py — Run all architectures against the AgriFood anomaly dataset.

Protocol (AgriFood / UseCase_1_(Avoine1)):
    Train+val : UseCase_1_(Avoine1)_Normal_3        (spatial guillotine, scaled:
                  train Y[12.5%:50%], val Y[75%:87.5%]  -> Y[125:500]/Y[750:875] on H=1000)
    Test step1: UseCase_1_(Avoine1)_Anomaly_Easy_L4_7
    Test step2: UseCase_1_(Avoine1)_Anomaly_Easy_L12_8  (1 train, 2 inference steps:
                  step 1 trains + infers, step 2 loads the step-1 weights and only infers)

Usage:
    python benchmark_agri.py
    python benchmark_agri.py --only gthad
    python benchmark_agri.py --skip autoad
    python benchmark_agri.py --dry-run                  # train only 1 epoch/iter per architecture
    python benchmark_agri.py --retrain no               # reuse saved weights
    python benchmark_agri.py --seeds 3
    python benchmark_agri.py --plot                     # replot from saved JSON
"""

import argparse
import contextlib
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOOD = "AgriFood"
BASE_DIR = "AgriFood/Dataset"
STEPS = [("L4_7", None), ("L12_8", "L12_8")]   # (step name, swap source basename or None)

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
]

# scripts_agri instances override data loading for the AgriFood protocol (train cube =
# Train/, test cube = Test/). The remaining architectures already accept base_dir.
ARCH_MODULE_MAP = {
    "bocknet": "scripts_agri.bocknet.__main__",
    "autoad": "scripts_agri.autoad.__main__",
    "otad": "scripts_agri.otad.__main__",
    "our": "scripts_agri.our.__main__",
    "pa2e": "scripts.pa2e.__main__",
    "gthad": "scripts.gthad.__main__",
    "superad": "scripts.superad.__main__",
    "sglnet": "scripts.sglnet.__main__",
    "dms2f": "scripts.dms2f.__main__",
}
ARCH_MODULE_BASE_DIR = {
    "pa2e": True, "gthad": True, "superad": True, "sglnet": True, "dms2f": True,
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
ARCH_RESULT_FILE = {a: f"{a}.json" for a in ALL_ARCHS}

SCALAR_KEYS = [
    "roc_auc", "pr_auc", "f1_tnr95", "acc_tnr95",
    "infer_time_sec", "max_vram_gb", "n_params", "gflops",
    "train_time_sec", "converged_epoch",
]


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def aggregate_runs(runs: list) -> dict:
    """Aggregate per-seed result dicts into mean ± std for each metric.

    Scalar metrics are stored as mean with a parallel ``<key>_std`` entry.
    Curve metrics (fpr/tpr/precision/recall) are interpolated onto a shared
    grid and stored as mean curves with ``<key>_std`` entries.
    """
    agg = {}
    for k in SCALAR_KEYS:
        vals = [r.get(k) for r in runs if r.get(k) is not None]
        if not vals:
            continue
        vals = [float(v) for v in vals]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    grid = np.linspace(0.0, 1.0, 200)
    for x_key, y_key in [("fpr", "tpr"), ("recall", "precision")]:
        ys = []
        for r in runs:
            x, y = r.get(x_key), r.get(y_key)
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
    return agg


def fmt_mean_std(mean, std, nd=4):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "nan"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


@contextlib.contextmanager
def staged_test(swap_src):
    """Temporarily swap Test/data + label with Test/data_{src} + label_{src}; restores on exit."""
    base = os.path.join(BASE_DIR, FOOD, 'Test')
    pairs = [
        ('data.hdr', f'data_{swap_src}.hdr'),
        ('data.bil', f'data_{swap_src}.bil'),
        ('label.npy', f'label_{swap_src}.npy'),
    ]
    staged = [p for p in pairs if os.path.exists(os.path.join(base, p[1]))]
    if swap_src is None or not staged:
        yield
        return

    moved = []
    try:
        for clean, alt in staged:
            bak = alt + '.agri_bak'
            os.rename(os.path.join(base, clean), os.path.join(base, bak))
            os.rename(os.path.join(base, alt), os.path.join(base, clean))
            moved.append((clean, alt, bak))
        yield
    finally:
        for clean, alt, bak in moved:
            os.rename(os.path.join(base, clean), os.path.join(base, alt))
            os.rename(os.path.join(base, bak), os.path.join(base, clean))


def run_arch(arch: str, output_dir: str, dry_run: bool, retrain: bool,
             seeds: int, cooldown: int) -> dict:
    """Run one architecture: 1 train (Normal_3) + 2 inference steps (L4_7, L12_8)."""
    print(f"\n{'#' * 80}\n# ARCHITECTURE: {arch}\n{'#' * 80}")

    module = importlib.import_module(ARCH_MODULE_MAP[arch])
    bench_fn = module.benchmark_food_type

    anomaly_dir = os.path.join(output_dir, 'anomaly-map')
    os.makedirs(anomaly_dir, exist_ok=True)

    step_runs = {name: [] for name, _ in STEPS}
    per_seed = []

    try:
        for s in range(seeds):
            seed_results = []
            for step_i, (step_name, swap_src) in enumerate(STEPS):
                adir = os.path.join(anomaly_dir, f'{FOOD}_{step_name}')
                os.makedirs(adir, exist_ok=True)
                kwargs = {
                    "dry_run": dry_run,
                    "retrain": retrain if step_i == 0 else False,
                    "anomaly_dir": adir,
                    "seed": s,
                }
                if ARCH_MODULE_BASE_DIR.get(arch):
                    kwargs["base_dir"] = BASE_DIR
                with staged_test(swap_src):
                    print(f"  [{arch}] step {step_name} (train={kwargs['retrain']})")
                    result = bench_fn(FOOD, **kwargs)
                seed_results.append(result)
                step_runs[step_name].append(result)
            # Aggregate the 2 inference steps of this seed into one run
            per_seed.append(aggregate_runs(seed_results))
            print(f"  [{arch}] {FOOD} — seed {s} done")
    except Exception as e:
        print(f"\n❌ [{arch}] Error on {FOOD}: {e}")
        traceback.print_exc()

    if not per_seed:
        return {}

    agg = aggregate_runs(per_seed)
    agg["test_steps"] = {name: aggregate_runs(runs) for name, runs in step_runs.items() if runs}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, ARCH_RESULT_FILE[arch])
    with open(out_path, "w") as f:
        json.dump({FOOD: agg}, f, indent=2, default=convert_numpy)
    print(f"\n✅ [{arch}] Results saved → {out_path}")

    if torch_cuda_available() and cooldown > 0:
        import torch
        torch.cuda.empty_cache()
        print(f"  Cooling down GPU for {cooldown}s...")
        time.sleep(cooldown)

    return agg


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _metric_table(title, results, key, nd=4):
    """Print a mean ± std table per architecture (single food column + average)."""
    col = 22
    header = f"{'Architecture':<20}  {FOOD:>{col}}"
    print(f"\n{title}:")
    print(header)
    print("─" * len(header))

    def get_val(r):
        if r is None:
            return None, None
        return r.get(key), r.get(f"{key}_std")

    scored = []
    for arch in ALL_ARCHS:
        r = results.get(arch)
        m, s = get_val(r)
        if m is not None and not np.isnan(m):
            scored.append((m, arch))
    scored.sort(reverse=True)

    for _, arch in scored:
        m, s = get_val(results[arch])
        row = f"{ARCH_DISPLAY_NAME.get(arch, arch):<20}  {fmt_mean_std(m, s, nd):>{col}}"
        print(row)


def _res_cell(results, arch, key, div=1.0, nd=2):
    r = results.get(arch)
    if not r or r.get(key) is None:
        return "nan"
    return f"{float(r[key]) / div:>{nd + 4}.{nd}f} ± {float(r.get(f'{key}_std', 0.0)) / div:.{nd}f}"


def print_summary(results: dict):
    """Print the AgriFood cross-architecture summary tables."""
    print(f"\n{'=' * 80}\n{'AGRIFOOD BENCHMARK SUMMARY (1 train, 2 inference)':^80}\n{'=' * 80}")
    print(f"Train+val : AgriFood/Train  = UseCase_1_(Avoine1)_Normal_3")
    print(f"Test step1: UseCase_1_(Avoine1)_Anomaly_Easy_L4_7")
    print(f"Test step2: UseCase_1_(Avoine1)_Anomaly_Easy_L12_8")
    print("Cells are mean ± std across seeds and the 2 test steps.\n")

    _metric_table("📊 ROC-AUC", results, "roc_auc")
    _metric_table("📊 PR-AUC", results, "pr_auc")
    _metric_table("📊 F1 @ 95% TNR", results, "f1_tnr95")
    _metric_table("📊 Accuracy @ 95% TNR", results, "acc_tnr95")

    # Per-step table
    col = 14
    print(f"\nPer-step scores (mean ± std across seeds):")
    step_header = f"{'Architecture':<20}  {'L4_7 ROC':>{col}}  {'L12_8 ROC':>{col}}  {'L4_7 PR':>{col}}  {'L12_8 PR':>{col}}"
    print(step_header)
    print("─" * len(step_header))
    for arch in ALL_ARCHS:
        r = results.get(arch) or {}
        steps = r.get("test_steps", {})
        name = ARCH_DISPLAY_NAME.get(arch, arch)
        cells = []
        for step_name, _ in STEPS:
            sr = steps.get(step_name) or {}
            roc = sr.get("roc_auc")
            pr = sr.get("pr_auc")
            cells.append(f"{roc:.4f} ± {sr.get('roc_auc_std', 0):.4f}" if roc is not None else "N/A")
            cells.append(f"{pr:.4f} ± {sr.get('pr_auc_std', 0):.4f}" if pr is not None else "N/A")
        print(f"{name:<20}  {cells[0]:>{col}}  {cells[1]:>{col}}  {cells[2]:>{col}}  {cells[3]:>{col}}")

    # Resources
    col = 14
    print(f"\n🖥  Resources (mean ± std across seeds and test steps):")
    res_header = f"{'Architecture':<20}  {'Infer s':>{col}}  {'VRAM GB':>{col}}  {'Params M':>{col}}  {'GFLOPs':>{col}}"
    print(res_header)
    print("─" * len(res_header))
    for arch in ALL_ARCHS:
        name = ARCH_DISPLAY_NAME.get(arch, arch)
        print(f"{name:<20}  {_res_cell(results, arch, 'infer_time_sec'):>{col}}  "
              f"{_res_cell(results, arch, 'max_vram_gb'):>{col}}  "
              f"{_res_cell(results, arch, 'n_params', div=1e6):>{col}}  "
              f"{_res_cell(results, arch, 'gflops'):>{col}}")
    print()


def plot_curves(results: dict, output_dir: str):
    """Plot ROC/PR curves (averaged + per-step) and anomaly heatmaps + box plots."""
    archs = [a for a in ALL_ARCHS if a != 'our'] + (['our'] if 'our' in results else [])
    archs = [a for a in archs if a in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(archs)))

    plot_dir = os.path.join(output_dir, 'plot')
    anomaly_dir = os.path.join(output_dir, 'anomaly-map')
    os.makedirs(plot_dir, exist_ok=True)

    # ── Averaged ROC / PR curves ───────────────────────────────────────
    for metric, x_key, y_key, xlabel, ylabel in [
        ("ROC", "fpr", "tpr", 'False Positive Rate', 'True Positive Rate'),
        ("PR", "recall", "precision", 'Recall', 'Precision'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        if metric == "ROC":
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
        for arch, color in zip(archs, colors):
            r = results[arch]
            if x_key not in r:
                continue
            auc_label = f"AUC={r.get('roc_auc', 0):.4f}" if metric == "ROC" else f"AP={r.get('pr_auc', 0):.4f}"
            ax.plot(r[x_key], r[y_key], color=color, lw=1.5,
                    label=f"{ARCH_DISPLAY_NAME.get(arch, arch)} ({auc_label})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric} Curves (avg over seeds & steps) — {FOOD}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{metric.lower()}_curve_{FOOD}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  {metric} curve saved → {path}")

    # ── Per-step ROC curves + heatmaps + box plots ─────────────────────
    for step_name, _ in STEPS:
        step_dir = os.path.join(anomaly_dir, f'{FOOD}_{step_name}')
        os.makedirs(step_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
        for arch, color in zip(archs, colors):
            sr = (results[arch].get("test_steps") or {}).get(step_name) or {}
            if not sr or "fpr" not in sr:
                continue
            ax.plot(sr["fpr"], sr["tpr"], color=color, lw=1.5,
                    label=f"{ARCH_DISPLAY_NAME.get(arch, arch)} (AUC={sr.get('roc_auc', 0):.4f})")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {FOOD} step {step_name}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"roc_curve_{FOOD}_{step_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ROC curve ({step_name}) saved → {path}")

        # Heatmaps
        for arch in archs:
            score_path = os.path.join(step_dir, f'{arch}_{FOOD}_scores.npy')
            if not os.path.exists(score_path):
                continue
            scores = np.load(score_path)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(scores, cmap='hot', aspect='auto')
            plt.colorbar(im, ax=ax, label='Anomaly Score')
            name = ARCH_DISPLAY_NAME.get(arch, arch)
            ax.set_title(f'{name} — {FOOD} step {step_name} Anomaly Map')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            fig.tight_layout()
            map_path = os.path.join(step_dir, f'{arch}_{FOOD}_{step_name}_map.png')
            fig.savefig(map_path, dpi=150)
            plt.close(fig)
        print(f"  Anomaly maps saved → {step_dir}/")

        # Box plot: normal vs anomaly scores
        fig, ax = plt.subplots(figsize=(10, 6))
        label_path = os.path.join(step_dir, f'{FOOD}_labels.npy')
        positions, labels_list, data_normal, data_anomaly = [], [], [], []
        if os.path.exists(label_path):
            gt = np.load(label_path).astype(bool)
            for i, arch in enumerate(archs):
                score_path = os.path.join(step_dir, f'{arch}_{FOOD}_scores.npy')
                if not os.path.exists(score_path):
                    continue
                scores = np.load(score_path).flatten()
                gt_flat = gt.flatten()[:len(scores)]
                normal_scores = scores[~gt_flat]
                anomaly_scores = scores[gt_flat]
                if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                    s_min, s_max = scores.min(), scores.max()
                    scores_norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(scores)
                    data_normal.append(scores_norm[~gt_flat])
                    data_anomaly.append(scores_norm[gt_flat])
                    positions.append(i)
                    labels_list.append(ARCH_DISPLAY_NAME.get(arch, arch))
            if data_normal:
                bp1 = ax.boxplot(data_normal, positions=[p - 0.2 for p in positions], widths=0.3,
                                 patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='steelblue', alpha=0.7),
                                 medianprops=dict(color='white'))
                bp2 = ax.boxplot(data_anomaly, positions=[p + 0.2 for p in positions], widths=0.3,
                                 patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='crimson', alpha=0.7),
                                 medianprops=dict(color='white'))
                ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Normal', 'Anomaly'], loc='upper right')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels_list, fontsize=8)
                ax.set_xlabel('Architecture')
                ax.set_ylabel('Anomaly Score (min-max scaled)')
                ax.set_title(f'Score Distribution — {FOOD} step {step_name}')
                ax.set_ylim(-0.02, 0.4)
                ax.grid(True, alpha=0.3, axis='y')
                fig.tight_layout()
                box_path = os.path.join(plot_dir, f'boxplot_{FOOD}_{step_name}.png')
                fig.savefig(box_path, dpi=150)
                print(f"  Box plot ({step_name}) saved → {box_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="AgriFood benchmark: train on Normal_3, inference on L4_7 and L12_8.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_agri.py
  python benchmark_agri.py --only gthad
  python benchmark_agri.py --skip autoad
  python benchmark_agri.py --dry-run --seed 1
  python benchmark_agri.py --retrain no
  python benchmark_agri.py --plot
        """,
    )
    parser.add_argument("--only", nargs="*", default=None, metavar="ARCH",
                        help=f"Run only these architectures: {ALL_ARCHS}")
    parser.add_argument("--skip", nargs="*", default=None, metavar="ARCH",
                        help="Skip these architectures")
    parser.add_argument("--dry-run", "-dr", action="store_true",
                        help="Train for 1 iteration/epoch only")
    parser.add_argument("--retrain", type=str, choices=["yes", "no"], default="yes",
                        help="Retrain models or load saved ones (default: yes)")
    parser.add_argument("--seeds", type=int, default=5, metavar="N",
                        help="Number of seeds to average results over (default: 5)")
    parser.add_argument("--output-dir", default="./results_agri",
                        help="Directory for JSON results (default: ./results_agri)")
    parser.add_argument("--cooldown", type=int, default=0, metavar="SECONDS",
                        help="GPU cooldown between architectures (default: 0)")
    parser.add_argument("--plot", action="store_true",
                        help="Skip running, load existing JSON results and regenerate plots only")
    args = parser.parse_args()

    only = [a for a in args.only for a in a.split(",")] if args.only else None
    skip = [a for a in args.skip for a in a.split(",")] if args.skip else None
    archs = [a for a in ALL_ARCHS if (not only or a in only) and (not skip or a not in skip)]
    if not archs:
        print("❌ No architectures selected.")
        sys.exit(1)

    if not os.path.isdir(os.path.join(BASE_DIR, FOOD)):
        print(f"❌ Dataset not found at {BASE_DIR}/{FOOD}. Run  python prepare_agrifood.py  first.")
        sys.exit(1)

    if args.plot:
        print(f"\n{'=' * 80}\n  Plot-only mode — loading results from {args.output_dir}\n{'=' * 80}")
        results = {}
        combined = os.path.join(args.output_dir, "benchmark_agri_all.json")
        if os.path.isfile(combined):
            with open(combined) as f:
                loaded = json.load(f)
            results = {a: (loaded.get(a) or {}).get(FOOD, {}) for a in ALL_ARCHS if loaded.get(a)}
            print(f"  ✅ Loaded {combined}")
        else:
            for arch in archs:
                path = os.path.join(args.output_dir, ARCH_RESULT_FILE[arch])
                if os.path.isfile(path):
                    with open(path) as f:
                        loaded = json.load(f)
                    results[arch] = loaded.get(FOOD, {})
            print(f"  ✅ Stitched {len(results)} architectures")
        print_summary(results)
        plot_curves(results, args.output_dir)
        return

    dry_run_str = " [DRY RUN]" if args.dry_run else ""
    print(f"\n{'=' * 80}")
    print(f"  AgriFood Anomaly Benchmark{dry_run_str}")
    print(f"  Food          : {FOOD}  (dataset: {BASE_DIR})")
    print(f"  Train+val     : UseCase_1_(Avoine1)_Normal_3 (guillotine, scaled split)")
    print(f"  Test steps    : {' + '.join(n for n, _ in STEPS)}  (1 train, 2 inference)")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"  Retrain       : {args.retrain}")
    print(f"  Seeds         : {args.seeds}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'=' * 80}")

    results = {}
    total_start = time.time()
    for arch in archs:
        results[arch] = run_arch(arch, args.output_dir, args.dry_run,
                                 args.retrain == "yes", args.seeds, args.cooldown)
        print(f"\n⏱  [{arch}] elapsed: {time.time() - total_start:.1f}s total")

    os.makedirs(args.output_dir, exist_ok=True)
    combined_path = os.path.join(args.output_dir, "benchmark_agri_all.json")
    with open(combined_path, "w") as f:
        json.dump({a: {FOOD: r} for a, r in results.items()}, f, indent=2, default=convert_numpy)
    print(f"\n✅ Combined results saved → {combined_path}")

    print_summary(results)
    plot_curves(results, args.output_dir)


if __name__ == "__main__":
    main()
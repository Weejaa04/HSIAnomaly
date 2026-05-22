#!/usr/bin/env python3
"""
BockNet: Blind-Block Reconstruction Network with a Guard Window for Hyperspectral Anomaly Detection
Reference: IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no. 5531916.
Implementation adapted from: https://github.com/wangdegang-hdu/IEEE_TGRS_BockNet

Run as:  python -m scripts.bocknet [options]
"""

import argparse
import json
import os
import random
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader

# Force FP32 precision (disable TF32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('highest')

from .model import BockNet
from .data import get_food_data
from .utils import SEED_DICT, UniversalEarlyStopping

# ─────────────────────────────────────────────────────────────────────────────

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. BockNet might be slow on CPU.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_auc(HSI_old, HSI_new, gt):
    """
    HSI_old: (H, W, B)
    HSI_new: (H, W, B)
    gt: (H, W)
    """
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col

    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order="F")
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order="F")
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord=2, axis=1, keepdims=True) ** 2
    detectmap = detectmap / n_band

    # get auc and pr
    label = np.reshape(gt, (n_pixels, 1), order="F")

    roc_auc = roc_auc_score(label, detectmap)
    pr_auc = average_precision_score(label, detectmap)

    # compute curve data
    scores_flat = detectmap.flatten()
    labels_flat = label.flatten().astype(int)
    fpr, tpr, _ = roc_curve(labels_flat, scores_flat)
    precision, recall, _ = precision_recall_curve(labels_flat, scores_flat)

    detectmap = np.reshape(detectmap, (n_row, n_col), order="F")

    return roc_auc, pr_auc, detectmap, fpr.tolist(), tpr.tolist(), precision.tolist(), recall.tolist()


def TensorToHSI(img):
    HSI = img.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    return HSI


# ============================= BENCHMARK ================================


def benchmark_food_type(
    food_type: str,
    base_dir: str = "AnomalyonFood/Dataset",
    lr: float = 1e-4,
    blindspot: int = 15,
    nch_ker: int = 64,
    weight_decay: float = 1e-5,
    lossm: str = "l1",
    dry_run: bool = False,
    retrain: bool = True,
    split_method: str = "spatial",
    **kwargs,
) -> dict:

    print(f"\n{'=' * 70}\nBENCHMARKING: {food_type} (split_method={split_method})\n{'=' * 70}")
    suffix = '_random' if split_method == 'random' else ''
    suffix += kwargs.get('noise_suffix', '')

    # For dry run, use tight constraints to complete quickly while testing full pipeline
    if dry_run:
        patience = 1  # Stop after first epoch without improvement
        min_delta = 1e-8  # Very tight threshold
    else:
        patience = 50
        min_delta = 1e-4

    seed = SEED_DICT.get(food_type, 42)
    set_seed(seed)

    # ===== Load data =====
    img_tensor, gt, H, W, B, train_mask, val_mask = get_food_data(
        food_type, base_dir, device=device, split_method=split_method
    )
    print(f"Image: ({H}, {W}, {B})  |  Anomaly pixels: {int(gt.sum())} / {H * W}")
    print(f"Split method: {split_method}")
    print(f"Train pixels: {int(train_mask.sum())}  |  Val pixels: {int(val_mask.sum())}")

    # ===== Model =====
    net = BockNet(blindspot=blindspot, nch_in=B, nch_out=B, nch_ker=nch_ker).to(device)

    params_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nParameter Count: {params_total:,}\n")

    optimizer = optim.Adam(
        net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay
    )

    if lossm.lower() == "l1":
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)

    # ===== Training =====
    model_dir = f"./weights/bocknet"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{food_type}{suffix}.pt"

    if not retrain and os.path.exists(model_path):
        print(f"\n=== Loading Model ({model_path}) ===")
        net.load_state_dict(torch.load(model_path, map_location=device))
        train_time = 0.0
    else:
        print(
            f"=== Training with Early Stopping (patience={patience}, min_delta={min_delta}) ==="
        )
        start = time.time()

        # Initialize early stopping monitor
        early_stopper = UniversalEarlyStopping(patience=patience, min_delta=min_delta)

        epoch = 0
        while not early_stopper.early_stop:
            epoch += 1

            # Training pass - apply mask to prevent learning from validation region
            net.train()
            optimizer.zero_grad()
            outputs = net(img_tensor)

            # Calculate loss only on Train Block (Y[50:200])
            raw_loss = criterion(outputs, img_tensor)
            masked_train_loss = (raw_loss * train_mask.unsqueeze(0).unsqueeze(0)).mean()

            masked_train_loss.backward()
            optimizer.step()

            # Validation pass - evaluate only on Validation Block (Y[300:350])
            net.eval()
            with torch.inference_mode():
                val_outputs = net(img_tensor)
                val_raw_loss = criterion(val_outputs, img_tensor)
                val_loss = (val_raw_loss * val_mask.unsqueeze(0).unsqueeze(0)).mean()

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:>4}  train_loss={masked_train_loss.item():.6f}  val_loss={val_loss.item():.6f}"
                )

            # Check early stopping based on validation loss
            early_stopper(val_loss.item())
            if dry_run:
                break

        train_time = time.time() - start
        print(f"\nTraining time: {train_time:.2f}s")
        print(f"Epochs trained: {epoch}")

        print(f"Saving model to {model_path}...")
        torch.save(net.state_dict(), model_path)

    # ===== Inference =====
    print("\n=== Inference ===")
    net.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    infer_start = time.time()

    with torch.no_grad():
        img_new = net(img_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    infer_time = time.time() - infer_start
    peak_vram_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    # ===== Metrics =====
    HSI_old = TensorToHSI(img_tensor)
    HSI_new = TensorToHSI(img_new)

    roc_auc, pr_auc, detectmap, fpr, tpr, precision, recall = get_auc(HSI_old, HSI_new, gt)

    print(
        f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  Inference time={infer_time:.4f}s  Peak VRAM={peak_vram_mib:.1f} MiB"
    )

    print(f"\n{'=' * 70}\nResults for {food_type}")
    print(f"{'Metric':<25} {'Value':>20}\n{'-' * 70}")
    print(f"{'ROC-AUC':<25} {roc_auc:>20.4f}")
    print(f"{'PR-AUC':<25} {pr_auc:>20.4f}")
    print(f"{'Inference Time (s)':<25} {infer_time:>20.4f}")
    print(f"{'Training Time (s)':<25} {train_time:>20.4f}")
    print(f"{'Peak VRAM (MiB)':<25} {peak_vram_mib:>20.1f}")
    print(f"{'=' * 70}\n")

    anomaly_dir = kwargs.get('anomaly_dir')
    if anomaly_dir:
        np.save(os.path.join(anomaly_dir, 'bocknet_' + food_type + '_scores.npy'), detectmap)
        label_path = os.path.join(anomaly_dir, f'{food_type}_labels.npy')
        if not os.path.exists(label_path):
            np.save(label_path, gt.astype(np.uint8))

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "detectmap_shape": list(detectmap.shape),
        "infer_time_sec": infer_time,
        "n_params": params_total,
        "max_vram_gb": round(peak_vram_mib / 1024, 4),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


# ============================= CLI ======================================


def main():
    parser = argparse.ArgumentParser(
        description="BockNet Benchmarking for Hyperspectral Food Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.bocknet
  python -m scripts.bocknet --food Almond
  python -m scripts.bocknet --food Almond Pistachio
        """,
    )
    parser.add_argument(
        "--food",
        nargs="*",
        default=None,
        help="Food type(s): Almond, Pistachio, GarlicStems (default: all)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--blindspot", type=int, default=15, help="Blindspot window size (default: 15)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--dry-run",
        "-dr",
        action="store_true",
        help="Train for 1 epoch only to validate the script",
    )
    parser.add_argument(
        "--retrain",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to retrain the model (default: yes)",
    )
    args = parser.parse_args()

    if not args.food or args.food == ["all"]:
        food_types = ALL_FOOD_TYPES
    else:
        food_types = args.food

    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            return

    print(f"\n📊 Food types: {', '.join(food_types)}")
    print(f"   LR={args.lr}  Blindspot={args.blindspot}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for ft in food_types:
        try:
            all_results[ft] = benchmark_food_type(
                ft,
                lr=args.lr,
                blindspot=args.blindspot,
                dry_run=args.dry_run,
                retrain=(args.retrain == "yes"),
            )
        except Exception as e:
            print(f"❌ Error on {ft}: {e}")
            traceback.print_exc()

    results_file = os.path.join(args.output_dir, "bocknet.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    if all_results:
        print(f"\n{'=' * 80}\n{'FINAL SUMMARY':^80}\n{'=' * 80}")
        print(
            f"{'Food Type':<20} {'ROC-AUC':>18} {'PR-AUC':>18} {'Infer Time (s)':>18}"
        )
        print(f"{'-' * 80}")
        for ft, r in all_results.items():
            print(
                f"{ft:<20} {r['roc_auc']:>18.4f} {r['pr_auc']:>18.4f} {r['inference_time']:>18.4f}"
            )
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

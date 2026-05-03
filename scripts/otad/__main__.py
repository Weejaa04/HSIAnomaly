#!/usr/bin/env python3
"""
OT-AD: Optimal Transport-Guided Transformer for Hyperspectral Anomaly Detection
Reference: IEEE Trans. Geosci. Remote Sens., vol. 64, 2026.

Run as:  python -m scripts.otad [options]
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
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from .model import OTADNet, OTADDataset, BlockRestore, hyper_norm
from .utils import get_food_data, SEED_DICT, UniversalEarlyStopping

ALL_FOOD_TYPES = ["Almond", "Pistachio", "GarlicStems"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. OT-AD might be slow on CPU.")


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


def map01(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def get_auc(HSI_old, HSI_new, gt):
    """Compute anomaly detection scores based on reconstruction error."""
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col

    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order="F")
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order="F")
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord=2, axis=1, keepdims=True) ** 2
    detectmap = detectmap / n_band

    detectmap = map01(detectmap)

    label = np.reshape(gt, (n_pixels, 1), order="F")

    roc_auc = roc_auc_score(label, detectmap)
    pr_auc = average_precision_score(label, detectmap)

    detectmap = np.reshape(detectmap, (n_row, n_col), order="F")

    return roc_auc, pr_auc, detectmap


def benchmark_food_type(
    food_type: str,
    base_dir: str = "AnomalyonFood/Dataset",
    lr: float = 1e-3,
    patch_size: int = 3,
    patch_grid: int = 5,
    stride: int = 5,
    batch_size: int = 64,
    num_heads: int = 2,
    dry_run: bool = False,
    retrain: bool = True,
    split_method: str = "spatial",
) -> dict:

    print(f"\n{'=' * 70}\nBENCHMARKING: {food_type} (split_method={split_method})\n{'=' * 70}")

    if dry_run:
        patience = 1
        min_delta = 1e-8
        max_iterations = 1
    else:
        patience = 50
        min_delta = 1e-4
        max_iterations = 150

    seed = SEED_DICT.get(food_type, 42)
    set_seed(seed)

    img_tensor, gt, H, W, B, train_mask, val_mask = get_food_data(food_type, base_dir, device=device, split_method=split_method)
    print(f"Image: ({H}, {W}, {B})  |  Anomaly pixels: {int(gt.sum())} / {H * W}")
    print(f"Split method: {split_method}")
    print(f"Train pixels: {int(train_mask.sum())}  |  Val pixels: {int(val_mask.sum())}")

    net = OTADNet(
        input_channels=B,
        embedding_dim=64,
        patch_size=patch_size,
        patch_grid=patch_grid,
        num_heads=num_heads,
        mlp_ratio=2.0,
        attn_drop_ratio=0.0,
        drop_ratio=0.0,
    ).to(device)

    params_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nParameter Count: {params_total:,}\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss().to(device)

    model_dir = f"./weights/otad"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{food_type}.pt"

    block_size = patch_size * patch_grid
    data_set_train = OTADDataset(img_tensor, block_size=block_size, stride=stride, spatial_mask=train_mask)
    data_set_val = OTADDataset(img_tensor, block_size=block_size, stride=stride, spatial_mask=val_mask)
    train_loader = DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_set_val, batch_size=batch_size, shuffle=False)
    block_restore = BlockRestore(block_size=block_size, stride=stride)

    if not retrain and os.path.exists(model_path):
        print(f"\n=== Loading Model ({model_path}) ===")
        net.load_state_dict(torch.load(model_path, map_location=device))
        train_time = 0.0
    else:
        print(
            f"=== Training with Early Stopping (patience={patience}, min_delta={min_delta}) ==="
        )
        start = time.time()

        early_stopper = UniversalEarlyStopping(patience=patience, min_delta=min_delta)

        iteration = 0
        while not early_stopper.early_stop and iteration < max_iterations:
            iteration += 1

            net.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                net_in = batch["block_input"].to(device)
                net_gt = batch["block_gt"].to(device)
                net_out, w_loss = net(net_in)
                loss = criterion(net_out, net_gt) + w_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            net.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch in test_loader:
                    net_in = batch["block_input"].to(device)
                    net_gt = batch["block_gt"].to(device)
                    net_out, w_loss = net(net_in)
                    loss = criterion(net_out, net_gt) + w_loss
                    val_loss += loss.item()

            val_loss /= len(test_loader)

            if iteration % 10 == 0:
                print(
                    f"  Iter {iteration:>4}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                )

            early_stopper(val_loss)
            if dry_run:
                break

        train_time = time.time() - start
        print(f"\nTraining time: {train_time:.2f}s")
        print(f"Iterations trained: {iteration}")

        print(f"Saving model to {model_path}...")
        torch.save(net.state_dict(), model_path)

    print("\n=== Inference ===")
    net.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    infer_start = time.time()

    res_map = []
    with torch.inference_mode():
        for batch in test_loader:
            test_in = batch["block_input"].to(device)
            test_out = net(test_in)
            res = torch.abs(test_in - test_out[0])
            res_map.append(res)

    res_map = torch.cat(res_map, dim=0)
    res_map = block_restore(res_map, data_set_val.padding, H, W, valid_indices=data_set_val.valid_indices)
    res_map = res_map[0].sum(0).cpu().numpy()
    res_map = hyper_norm(res_map)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    infer_time = time.time() - infer_start
    peak_vram_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    HSI_old = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    HSI_new = np.zeros_like(HSI_old)
    if res_map.shape == (H, W):
        for b in range(B):
            HSI_new[:, :, b] = HSI_old[:, :, b] - res_map * (
                HSI_old[:, :, b].max() - HSI_old[:, :, b].min()
            )

    roc_auc = roc_auc_score(gt.flatten(), res_map.flatten())
    pr_auc = average_precision_score(gt.flatten(), res_map.flatten())

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

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "detectmap_shape": list(res_map.shape),
        "infer_time_sec": infer_time,
        "n_params": params_total,
        "max_vram_gb": round(peak_vram_mib / 1024, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="OT-AD Benchmarking for Hyperspectral Food Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.otad
  python -m scripts.otad --food Almond
  python -m scripts.otad --food Almond Pistachio
        """,
    )
    parser.add_argument(
        "--food",
        nargs="*",
        default=None,
        help="Food type(s): Almond, Pistachio, GarlicStems (default: all)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=3, help="Patch size (default: 3)"
    )
    parser.add_argument(
        "--patch-grid", type=int, default=5, help="Patch grid size (default: 5)"
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Sliding window stride (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
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
        help="Train for 1 iteration only to validate the script",
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
    print(f"   LR={args.lr}  PatchSize={args.patch_size}  PatchGrid={args.patch_grid}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for ft in food_types:
        try:
            all_results[ft] = benchmark_food_type(
                ft,
                lr=args.lr,
                patch_size=args.patch_size,
                patch_grid=args.patch_grid,
                stride=args.stride,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                dry_run=args.dry_run,
                retrain=(args.retrain == "yes"),
            )
        except Exception as e:
            print(f"❌ Error on {ft}: {e}")
            traceback.print_exc()

    results_file = os.path.join(args.output_dir, "otad.json")
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
                f"{ft:<20} {r['roc_auc']:>18.4f} {r['pr_auc']:>18.4f} {r['infer_time_sec']:>18.4f}"
            )
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

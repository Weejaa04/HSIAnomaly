#!/usr/bin/env python3
"""
GT-HAD: Graph Transformer for Hyperspectral Anomaly Detection
Reference: https://github.com/jeline0110/GT-HAD

Run as:  python -m scripts.GT-HAD [options]
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
import torch.optim as Optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from .data  import load_food_data, DatasetHsi
from .net   import Net
from .block import Block_fold, Block_search
from .utils import SEED_DICT, get_params, img2mask


ALL_FOOD_TYPES = ['Almond', 'Pistachio', 'GarlicStems']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================= BENCHMARK ================================

def benchmark_food_type(food_type: str,
                        base_dir: str = 'AnomalyonFood/Dataset',
                        num_iters: int = 150,
                        search_iter: int = 25,
                        batch_size: int = 64,
                        lr: float = 1e-3) -> dict:
    print(f"\n{'='*70}\nBENCHMARKING: {food_type}\n{'='*70}")

    seed = SEED_DICT.get(food_type, 42)
    set_seed(seed)

    # ===== Load data =====
    img_var, gt, H, W, B = load_food_data(food_type, base_dir, device=device)
    print(f'Image: ({H}, {W}, {B})  |  Anomaly pixels: {int(gt.sum())} / {H*W}')

    # ===== Dataset + DataLoader =====
    patch_size   = 3
    patch_stride = 3
    block_size   = patch_size * patch_stride

    data_set     = DatasetHsi(img_var, wsize=block_size, wstride=3)
    block_fold   = Block_fold(wsize=block_size, wstride=3)
    block_search = Block_search(img_var, wsize=block_size, wstride=3)
    data_loader  = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    data_num = len(data_set)

    # ===== Model =====
    net = Net(in_chans=B, embed_dim=64, patch_size=patch_size,
              patch_stride=patch_stride, mlp_ratio=2.0, attn_drop=0.0, drop=0.0).to(device)

    # Move internal mask to correct device
    net.attn_layer.attn.mask = net.attn_layer.attn.mask.to(device)

    params_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'\nParameter Count: {params_total:,}\n')

    mse       = nn.MSELoss().to(device)
    optimizer = Optim.Adam(get_params(net), lr=lr)
    avgpool   = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)).to(device)

    # ===== Training =====
    match_vec     = torch.zeros(data_num, device=device)
    search_matrix = torch.zeros(data_num, B, block_size, block_size, device=device)
    search_index  = torch.arange(0, data_num, device=device)

    print(f'=== Training ({num_iters} iterations) ===')
    start = time.time()

    for iteration in range(1, num_iters + 1):
        net.train()
        search_flag = (iteration % search_iter == 0) and (iteration != num_iters)
        iter_loss = 0.0

        for batch_data in data_loader:
            optimizer.zero_grad()
            net_gt    = batch_data['block_gt'].to(device)
            net_input = batch_data['block_input'].to(device)
            block_idx = batch_data['index'].to(device)

            net_out = net(net_input, block_idx=block_idx, match_vec=match_vec)
            if search_flag:
                search_matrix[block_idx] = net_out.detach()

            loss = mse(net_out, net_gt)
            loss.backward()
            optimizer.step()
            iter_loss += loss.item()

        if search_flag:
            match_vec     = torch.zeros(data_num, device=device)
            search_back   = block_fold(search_matrix.detach(), data_set.padding, H, W)
            match_vec     = block_search(search_back.detach(), match_vec, search_index)

        avg_loss = iter_loss / len(data_loader)
        print(f'  Iter {iteration:>3}/{num_iters}  loss={avg_loss:.6f}')

    train_time = time.time() - start
    print(f'\nTraining time: {train_time:.2f}s')

    # ===== Inference =====
    print('\n=== Inference ===')
    net.eval()
    infer_loader   = DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=False)
    infer_res_list = []
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.time()

    with torch.no_grad():
        for data in infer_loader:
            infer_in  = data['block_input'].to(device)
            infer_idx = data['index'].to(device)
            infer_out = net(infer_in, block_idx=infer_idx, match_vec=match_vec)
            infer_res = torch.abs(infer_in - infer_out) ** 2
            infer_res = avgpool(infer_res)
            infer_res_list.append(infer_res)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    infer_time    = time.time() - infer_start
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
    infer_res_out = torch.cat(infer_res_list, dim=0)
    infer_res_back = block_fold(infer_res_out.detach(), data_set.padding, H, W)
    residual_np   = img2mask(infer_res_back)

    # ===== Metrics =====
    gt_flat  = gt.flatten()
    res_flat = residual_np.flatten()

    roc_auc = roc_auc_score(gt_flat, res_flat)
    pr_auc  = average_precision_score(gt_flat, res_flat)

    print(f'ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  Inference time={infer_time:.4f}s  Peak VRAM={peak_vram_mib:.1f} MiB')

    print(f"\n{'='*70}\nResults for {food_type}")
    print(f"{'Metric':<25} {'Value':>20}\n{'-'*70}")
    print(f"{'ROC-AUC':<25} {roc_auc:>20.4f}")
    print(f"{'PR-AUC':<25} {pr_auc:>20.4f}")
    print(f"{'Inference Time (s)':<25} {infer_time:>20.4f}")
    print(f"{'Training Time (s)':<25} {train_time:>20.4f}")
    print(f"{'Peak VRAM (MiB)':<25} {peak_vram_mib:>20.1f}")
    print(f"{'='*70}\n")

    return {
        'food_type':      food_type,
        'parameters':     params_total,
        'roc_auc':        float(roc_auc),
        'pr_auc':         float(pr_auc),
        'inference_time': infer_time,
        'peak_vram_mib':  round(peak_vram_mib, 2),
        'training_time':  train_time,
    }


# ============================= CLI ======================================

def main():
    parser = argparse.ArgumentParser(
        description='GT-HAD Benchmarking for Hyperspectral Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.GT-HAD
  python -m scripts.GT-HAD --food Almond
  python -m scripts.GT-HAD --food Almond Pistachio
        """,
    )
    parser.add_argument('--food',        nargs='*', default=None,
                        help='Food type(s): Almond, Pistachio, GarlicStems (default: all)')
    parser.add_argument('--num-iters',   type=int, default=150,  help='Training iterations (default: 150)')
    parser.add_argument('--search-iter', type=int, default=25,   help='CMM search interval (default: 25)')
    parser.add_argument('--batch-size',  type=int, default=64,   help='Batch size (default: 64)')
    parser.add_argument('--lr',          type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--output-dir',  type=str, default='./results', help='Output directory (default: ./results)')
    args = parser.parse_args()

    if not args.food or args.food == ['all']:
        food_types = ALL_FOOD_TYPES
    else:
        food_types = args.food

    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            return

    print(f"\n📊 Food types: {', '.join(food_types)}")
    print(f"   Iters={args.num_iters}  SearchEvery={args.search_iter}  Batch={args.batch_size}  LR={args.lr}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for ft in food_types:
        try:
            all_results[ft] = benchmark_food_type(
                ft,
                num_iters=args.num_iters,
                search_iter=args.search_iter,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        except Exception as e:
            print(f"❌ Error on {ft}: {e}")
            traceback.print_exc()

    results_file = os.path.join(args.output_dir, 'gt-had.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    if all_results:
        print(f"\n{'='*80}\n{'FINAL SUMMARY':^80}\n{'='*80}")
        print(f"{'Food Type':<20} {'ROC-AUC':>18} {'PR-AUC':>18} {'Infer Time (s)':>18}")
        print(f"{'-'*80}")
        for ft, r in all_results.items():
            print(f"{ft:<20} {r['roc_auc']:>18.4f} {r['pr_auc']:>18.4f} {r['inference_time']:>18.4f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
PA2E (Conv1D variant) — Partial Autoencoder with Deep k-NN
Run as:  python -m scripts.ours [options]
"""

import argparse
import copy
import json
import os
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data  import load_hsi_data, load_label, calibrate_hsi, HSIPixelDataset
from .model import PA2E, PA2EFT
from .utils import device


ALL_FOOD_TYPES = ['Almond', 'Pistachio', 'GarlicStems']


# ============================= TRAINING =================================

def compute_reconstruction_loss(model, x):
    z, reconstructed_windows = model(x)
    original_windows = model.window_extractor(x)
    return F.mse_loss(reconstructed_windows, original_windows), z


def train_phase1(model, train_loader, num_epochs=20, lr=1e-3):
    print('\n=== Phase 1: Pre-training (Autoencoder + OneCycleLR) ===')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1e4,
    )
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            x = batch.to(device)
            optimizer.zero_grad()
            loss, _ = compute_reconstruction_loss(model, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}')
    return losses


def initialize_center(model, train_loader):
    print('\nInitializing center...')
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Computing center', leave=False):
            z, _ = model.encode(batch.to(device))
            latents.append(z)
    center = torch.cat(latents).mean(dim=0)
    model.center = center
    model.center_initialized = True
    print(f'Center shape: {center.shape}, norm: {center.norm().item():.4f}')
    return center


def train_phase2(model, train_loader, num_epochs=100, lr=1e-4, alpha=0.1):
    print('\n=== Phase 2: Main Training (DSVDD-style + CosineAnnealingLR) ===')
    initialize_center(model, train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 1e-2)
    losses, center_losses, recon_losses = [], [], []
    for epoch in range(num_epochs):
        model.train()
        el = ecl = erl = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            x = batch.to(device)
            optimizer.zero_grad()
            recon_loss, z = compute_reconstruction_loss(model, x)
            center_loss = torch.mean(torch.sum((z - model.center) ** 2, dim=1))
            loss = center_loss + alpha * recon_loss
            loss.backward()
            optimizer.step()
            el += loss.item(); ecl += center_loss.item(); erl += recon_loss.item()
        scheduler.step()
        n = len(train_loader)
        al, acl, arl = el/n, ecl/n, erl/n
        losses.append(al); center_losses.append(acl); recon_losses.append(arl)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {al:.6f}, Center: {acl:.6f}, '
              f'Recon: {arl:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}')
    return losses, center_losses, recon_losses


# ============================= INFERENCE ================================

def build_gpu_memory_bank(model, loader):
    model.eval()
    bank = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Compiling Bank on GPU', leave=False):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            z, _ = model.encode(x.to(device))
            bank.append(z)
    return torch.cat(bank, dim=0)


@torch.no_grad()
def fuse_conv_bn_eval(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    assert not conv.training and not bn.training
    fused = nn.Conv1d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True,
    ).to(conv.weight.device)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = conv.weight.data * scale[:, None, None]
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data = scale * (conv_bias - bn.running_mean) + bn.bias
    return fused


def apply_conv_bn_fusion(model):
    model_fused = copy.deepcopy(model)
    model_fused.eval()
    for encoder_seq in model_fused.partial_encoders.encoders:
        pairs = [
            (i, i + 1)
            for i in range(len(encoder_seq) - 1)
            if isinstance(encoder_seq[i], nn.Conv1d) and isinstance(encoder_seq[i + 1], nn.BatchNorm1d)
        ]
        for ci, bi in pairs:
            encoder_seq[ci] = fuse_conv_bn_eval(encoder_seq[ci], encoder_seq[bi])
            encoder_seq[bi] = nn.Identity()
    return model_fused


# ============================= BENCHMARK ================================

def benchmark_food_type(food_type, base_dir='AnomalyonFood/Dataset',
                        batch_size=512, phase1_epochs=20, phase2_epochs=100):
    print(f"\n{'='*70}\nBENCHMARKING: {food_type}\n{'='*70}")
    base = f'{base_dir}/{food_type}'

    print(f'\nLoading {food_type} training data...')
    train_data = calibrate_hsi(
        load_hsi_data(f'{base}/Train/data.hdr'),
        f'{base}/Train/WHITEREF.hdr', f'{base}/Train/DARKREF.hdr',
    )
    H, W, B = train_data.shape
    X_train, _ = train_test_split(train_data.reshape(-1, B), test_size=0.7, random_state=42)

    print(f'Loading {food_type} test data...')
    test_data = calibrate_hsi(
        load_hsi_data(f'{base}/Test/data.hdr'),
        f'{base}/Test/WHITEREF.hdr', f'{base}/Test/DARKREF.hdr',
    )
    test_labels = load_label(f'{base}/Test/label.npy')
    H_test, W_test, B = test_data.shape
    test_data   = test_data.reshape(-1, B)
    test_labels = test_labels.reshape(-1)

    train_dataset = HSIPixelDataset(X_train)
    test_dataset  = HSIPixelDataset(test_data, test_labels)
    print(f'Train: {X_train.shape}, Test: {test_data.shape}, Bands: {B}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)

    model = PA2E(input_dim=B, window_size=56, stride=14,
                 latent_local_dim=28, latent_global_dim=32).to(device)
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_encoders = sum(p.numel() for p in model.partial_encoders.parameters() if p.requires_grad)
    params_agg_enc = sum(p.numel() for p in model.aggregate_encoder.parameters() if p.requires_grad)
    params_agg_dec = sum(p.numel() for p in model.aggregate_decoder.parameters() if p.requires_grad)
    params_decoders = sum(p.numel() for p in model.partial_decoders.parameters() if p.requires_grad)

    print(f'\nParameter Count Breakdown:')
    print(f'  Partial Encoders:  {params_encoders:,}')
    print(f'  Aggregate Encoder: {params_agg_enc:,}')
    print(f'  Aggregate Decoder: {params_agg_dec:,}')
    print(f'  Partial Decoders:  {params_decoders:,}')
    print(f'  TOTAL:             {params_total:,}\n')

    train_phase1(model, train_loader, num_epochs=phase1_epochs, lr=1e-3)

    model_ft = PA2EFT(model).to(device)
    train_phase2(model_ft, train_loader, num_epochs=phase2_epochs, lr=1e-3, alpha=0.5)

    labels = (test_labels != 2).astype(int)

    def _run_knn(mdl, tag):
        bank = build_gpu_memory_bank(mdl, train_loader)
        print(f'Memory Bank: {bank.shape}')
        mdl.eval()
        all_scores = []
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = datetime.now()
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                z, _ = mdl.encode(x.to(device))
                min_dist, _ = torch.cdist(z, bank, p=2.0).min(dim=1)
                all_scores.append(min_dist.cpu())
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        scores = torch.cat(all_scores).numpy()
        smoothed = median_filter(scores.reshape(H_test, W_test), size=3).flatten()
        elapsed = (datetime.now() - start).total_seconds()
        peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0

        roc  = roc_auc_score(labels, smoothed)
        p, r, _ = precision_recall_curve(labels, smoothed)
        pr   = auc(r, p)
        print(f'⚡ {tag}: {elapsed:.4f}s  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  Peak VRAM={peak_vram_mib:.1f} MiB')
        return roc, pr, elapsed, peak_vram_mib

    print('\n--- Original model ---')
    roc_o, pr_o, t_o, vram_o = _run_knn(model_ft, 'Original')
    print('\n--- Fused model ---')
    roc_f, pr_f, t_f, vram_f = _run_knn(apply_conv_bn_fusion(model_ft).to(device), 'Fused')

    speedup = t_o / t_f
    print(f"\n{'='*70}\nResults for {food_type}")
    print(f"{'Metric':<25} {'Original':>20} {'Fused':>20}")
    print(f"{'-'*70}")
    print(f"{'ROC-AUC':<25} {roc_o:>20.4f} {roc_f:>20.4f}")
    print(f"{'PR-AUC':<25} {pr_o:>20.4f} {pr_f:>20.4f}")
    print(f"{'Time (s)':<25} {t_o:>20.4f} {t_f:>20.4f}")
    print(f"{'Peak VRAM (MiB)':<25} {vram_o:>20.1f} {vram_f:>20.1f}")
    print(f"{'Speedup':<25} {'1.00x':>20} {speedup:>20.2f}x\n{'='*70}\n")

    return {
        'food_type': food_type,
        'parameters': params_total,
        'original_model': {'roc_auc': float(roc_o), 'pr_auc': float(pr_o), 'inference_time': t_o, 'peak_vram_mib': round(vram_o, 2)},
        'fused_model':    {'roc_auc': float(roc_f), 'pr_auc': float(pr_f), 'inference_time': t_f, 'peak_vram_mib': round(vram_f, 2)},
        'speedup': speedup,
    }



# ============================= CLI ======================================

def main():
    parser = argparse.ArgumentParser(
        description='PA2E (Conv1D) Benchmarking for Hyperspectral Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.ours
  python -m scripts.ours --food Almond
  python -m scripts.ours --food Almond Pistachio
        """,
    )
    parser.add_argument('--food',          nargs='*', default=None)
    parser.add_argument('--batch-size',    type=int, default=512)
    parser.add_argument('--phase1-epochs', type=int, default=20)
    parser.add_argument('--phase2-epochs', type=int, default=100)
    parser.add_argument('--output-dir',    type=str, default='./results')
    args = parser.parse_args()

    if not args.food or args.food == ['all']:
        food_types = ALL_FOOD_TYPES
    else:
        food_types = args.food

    for ft in food_types:
        if ft not in ALL_FOOD_TYPES:
            print(f"❌ Unknown food type: {ft}. Available: {', '.join(ALL_FOOD_TYPES)}")
            return

    print(f"\n📊 Food types: {', '.join(food_types)}")
    print(f"   Batch={args.batch_size}  P1={args.phase1_epochs}  P2={args.phase2_epochs}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for ft in food_types:
        try:
            all_results[ft] = benchmark_food_type(
                ft, batch_size=args.batch_size,
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
            )
        except Exception as e:
            print(f"❌ Error on {ft}: {e}")
            traceback.print_exc()

    results_file = os.path.join(args.output_dir, 'ours.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    if all_results:
        print(f"\n{'='*80}\n{'FINAL SUMMARY':^80}\n{'='*80}")
        print(f"{'Food Type':<20} {'Orig PR-AUC':>18} {'Fused PR-AUC':>18} {'Speedup':>18}")
        print(f"{'-'*80}")
        for ft, r in all_results.items():
            print(f"{ft:<20} {r['original_model']['pr_auc']:>18.4f} "
                  f"{r['fused_model']['pr_auc']:>18.4f} {r['speedup']:>18.2f}x")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

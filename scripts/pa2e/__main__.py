#!/usr/bin/env python3
"""
PA2E (FC variant) — Partial Autoencoder with Deep k-NN
Run as:  python -m scripts.pa2e [options]
"""

import argparse
import copy
import datetime
import json
import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
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
    print('\n=== Phase 1: Pre-training (Autoencoder) ===')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg:.6f}')
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


def train_phase2(model, train_loader, num_epochs=20, lr=1e-4, alpha=0.1):
    print('\n=== Phase 2: Main Training (DSVDD-style) ===')
    initialize_center(model, train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, center_losses, recon_losses = [], [], []
    for epoch in range(num_epochs):
        model.train()
        el = ecl = erl = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            x = batch.to(device)
            optimizer.zero_grad()
            recon_loss, z  = compute_reconstruction_loss(model, x)
            center_loss    = torch.mean(torch.sum((z - model.center) ** 2, dim=1))
            loss           = center_loss + alpha * recon_loss
            loss.backward()
            optimizer.step()
            el += loss.item(); ecl += center_loss.item(); erl += recon_loss.item()
        n = len(train_loader)
        al, acl, arl = el/n, ecl/n, erl/n
        losses.append(al); center_losses.append(acl); recon_losses.append(arl)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {al:.6f}, Center: {acl:.6f}, Recon: {arl:.6f}')
    return losses, center_losses, recon_losses


# ============================= LAYER FUSION =============================

def fuse_linear_linear(first: nn.Linear, second: nn.Linear) -> nn.Linear:
    if first.out_features != second.in_features:
        raise ValueError("Linear dimensions incompatible for fusion")
    fused = nn.Linear(first.in_features, second.out_features, bias=True).to(
        device=first.weight.device, dtype=first.weight.dtype)
    with torch.no_grad():
        w1 = first.weight
        b1 = first.bias if first.bias is not None else torch.zeros(first.out_features, device=w1.device, dtype=w1.dtype)
        w2 = second.weight
        b2 = second.bias if second.bias is not None else torch.zeros(second.out_features, device=w2.device, dtype=w2.dtype)
        fused.weight.copy_(w2 @ w1)
        fused.bias.copy_(w2 @ b1 + b2)
    return fused


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    fused = nn.Linear(linear.in_features, linear.out_features, bias=True).to(
        device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        w  = linear.weight
        b  = linear.bias if linear.bias is not None else torch.zeros(linear.out_features, device=w.device, dtype=w.dtype)
        gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
        beta  = bn.bias   if bn.affine else torch.zeros_like(bn.running_mean)
        scale = gamma / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(w * scale.unsqueeze(1))
        fused.bias.copy_((b - bn.running_mean) * scale + beta)
    return fused


def fuse_sequential(seq: nn.Sequential) -> nn.Sequential:
    # Pass 1: fuse Linear+BN pairs
    mods = list(seq.children())
    bn_fused, i = [], 0
    while i < len(mods):
        if (i + 1 < len(mods)
                and isinstance(mods[i], nn.Linear)
                and isinstance(mods[i + 1], nn.BatchNorm1d)):
            bn_fused.append(fuse_linear_bn(mods[i], mods[i + 1]))
            i += 2
        else:
            bn_fused.append(mods[i]); i += 1
    # Pass 2: fuse consecutive Linears
    final, i = [], 0
    while i < len(bn_fused):
        cur = bn_fused[i]
        if isinstance(cur, nn.Linear):
            j = i + 1
            while j < len(bn_fused) and isinstance(bn_fused[j], nn.Linear):
                cur = fuse_linear_linear(cur, bn_fused[j]); j += 1
            final.append(cur); i = j
        else:
            final.append(cur); i += 1
    return nn.Sequential(*final)


def fuse_pa2e_for_inference(model: PA2E) -> PA2E:
    fused = copy.deepcopy(model).eval()
    for idx, enc in enumerate(fused.partial_encoders.encoders):
        fused.partial_encoders.encoders[idx] = fuse_sequential(enc)
    for idx, dec in enumerate(fused.partial_decoders.decoders):
        fused.partial_decoders.decoders[idx] = fuse_sequential(dec)
    return fused


# ============================= BENCHMARK ================================

def benchmark_food_type(food_type, base_dir='AnomalyonFood/Dataset',
                        batch_size=512, phase1_epochs=20, phase2_epochs=20):
    print(f"\n{'='*70}\nBENCHMARKING: {food_type}\n{'='*70}")
    base = f'{base_dir}/{food_type}'

    print(f'\nLoading {food_type} training data...')
    train_data = calibrate_hsi(
        load_hsi_data(f'{base}/Train/data.hdr'),
        f'{base}/Train/WHITEREF.hdr', f'{base}/Train/DARKREF.hdr',
    )
    print(f'Loading {food_type} test data...')
    test_data   = calibrate_hsi(
        load_hsi_data(f'{base}/Test/data.hdr'),
        f'{base}/Test/WHITEREF.hdr', f'{base}/Test/DARKREF.hdr',
    )
    test_labels = load_label(f'{base}/Test/label.npy')
    print(f'Train: {train_data.shape}, Test: {test_data.shape}')

    train_dataset = HSIPixelDataset(train_data)
    test_dataset  = HSIPixelDataset(test_data, test_labels)
    print(f'Training pixels: {len(train_dataset)}, Test pixels: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)

    input_dim = train_dataset.pixels.shape[1]
    model = PA2E(input_dim=input_dim, window_size=56, stride=14,
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
    train_phase2(model_ft, train_loader, num_epochs=phase2_epochs, lr=1e-4, alpha=0.1)

    def _eval(mdl, tag):
        mdl.eval()
        all_scores, all_labels = [], []
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = datetime.datetime.now()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'{tag} inference', leave=False):
                x, lbl = batch
                all_scores.append(mdl.compute_anomaly_score(x.to(device)).cpu())
                all_labels.append(lbl)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed = (datetime.datetime.now() - start).total_seconds()
        peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
        scores = torch.cat(all_scores).numpy()
        labels = (torch.cat(all_labels).numpy() != 2).astype(int)
        roc = roc_auc_score(labels, scores)
        pr  = average_precision_score(labels, scores)
        print(f'{tag}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, time={elapsed:.4f}s, Peak VRAM={peak_vram_mib:.1f} MiB')
        return roc, pr, elapsed, peak_vram_mib

    print('\n--- Evaluating original model ---')
    roc_o, pr_o, t_o, vram_o = _eval(model_ft, 'Original')

    print('\n--- Applying layer fusion ---')
    fused_ft = fuse_pa2e_for_inference(model_ft).to(device)
    print('--- Evaluating fused model ---')
    roc_f, pr_f, t_f, vram_f = _eval(fused_ft, 'Fused')

    speedup = t_o / t_f if t_f > 0 else float('inf')
    print(f"\n{'='*70}\nResults for {food_type}")
    print(f"{'Metric':<25} {'Original':>20} {'Fused':>20}\n{'-'*70}")
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
        description='PA2E (FC) Benchmarking for Hyperspectral Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.pa2e
  python -m scripts.pa2e --food Almond
  python -m scripts.pa2e --food Almond Pistachio
        """,
    )
    parser.add_argument('--food',          nargs='*', default=None)
    parser.add_argument('--batch-size',    type=int, default=512)
    parser.add_argument('--phase1-epochs', type=int, default=20)
    parser.add_argument('--phase2-epochs', type=int, default=20)
    parser.add_argument('--output-dir',    type=str, default='./results')
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

    results_file = os.path.join(args.output_dir, 'pa2e.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    if all_results:
        print(f"\n{'='*80}\n{'FINAL SUMMARY':^80}\n{'='*80}")
        print(f"{'Food Type':<20} {'Orig ROC-AUC':>18} {'Fused ROC-AUC':>18} {'Speedup':>18}")
        print(f"{'-'*80}")
        for ft, r in all_results.items():
            print(f"{ft:<20} {r['original_model']['pr_auc']:>18.4f} "
                  f"{r['fused_model']['pr_auc']:>18.4f} {r['speedup']:>18.2f}x")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

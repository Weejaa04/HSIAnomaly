"""
GT-HAD (Global-local Transformer for Hyperspectral Anomaly Detection) Benchmark

Architecture: Block-based Transformer with Content Matching Module (CMM)
Training: 150 epochs with CMM updates every 25 iterations
Inference: Residual-based anomaly scoring

Usage:
    from scripts.gthad.__main__ import benchmark_food_type
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import median_filter
from tqdm import tqdm
import time
import random

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.gthad.model import Net
from scripts.gthad.data import DatasetHsi
from scripts.gthad.block import BlockFold, BlockSearch
from scripts.gthad.utils import (
    load_hsi_data, load_label, calibrate_hsi,
    get_spatial_train_val_mask, save_weights, load_weights, weights_exist,
    img2mask, UniversalEarlyStopping
)

# ─────────────────────────────────────────────────────────────────────────────
# SEED & DEVICE
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_food_type(food_type: str,
                        base_dir: str = 'AnomalyonFood/Dataset',
                        dry_run: bool = False,
                        retrain: bool = True,
                        **kwargs) -> dict:
    """
    Train and evaluate GT-HAD on a single food type.
    
    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        base_dir: Root path to dataset
        dry_run: If True, run minimal training (1 epoch) for validation
        retrain: If True, train from scratch. If False, load saved weights
        **kwargs: Additional hyperparameters
    
    Returns:
        dict with keys: roc_auc, pr_auc, detectmap_shape, infer_time_sec, n_params
    """
    print(f"\n{'='*70}\nBENCHMARKING: {food_type} (GT-HAD)\n{'='*70}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────
    # Load training data for model training
    train_dir = os.path.join(base_dir, food_type, 'Train')
    data = load_hsi_data(os.path.join(train_dir, 'data.hdr'))
    
    # Calibrate training data
    data = calibrate_hsi(
        data,
        os.path.join(train_dir, 'WHITEREF.hdr'),
        os.path.join(train_dir, 'DARKREF.hdr')
    )
    
    # Load test data for evaluation (has ground truth anomalies)
    test_dir = os.path.join(base_dir, food_type, 'Test')
    data_test = load_hsi_data(os.path.join(test_dir, 'data.hdr'))
    gt = load_label(os.path.join(test_dir, 'label.npy'))
    gt = (gt != 2).astype(np.uint8)  # Convert to binary mask (0=normal, 1=anomaly)
    
    # Calibrate test data
    data_test = calibrate_hsi(
        data_test,
        os.path.join(test_dir, 'WHITEREF.hdr'),
        os.path.join(test_dir, 'DARKREF.hdr')
    )
    
    print(f"Training data shape: {data.shape}")
    print(f"Test data shape: {data_test.shape}")
    print(f"GT shape: {gt.shape}")
    
    # Convert to torch tensor (batch=1, bands, height, width)
    data_torch = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODEL & TRAINING SETUP
    # ─────────────────────────────────────────────────────────────────────────
    band = data.shape[2]
    patch_size = 3
    patch_stride = 3
    block_size = patch_size * patch_stride  # 9x9
    
    # Model
    net = Net(
        in_chans=band,
        embed_dim=64,
        patch_size=patch_size,
        patch_stride=patch_stride,
        mlp_ratio=2.0,
        attn_drop=0.0,
        drop=0.0,
        depth=4
    ).to(device)
    
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of params: {n_params}')
    
    # Check if we should load pre-trained weights
    if not retrain and weights_exist(food_type, arch='gthad'):
        load_weights(net, food_type, device=device, arch='gthad')
        train_time = 0.0
    else:
        # ─────────────────────────────────────────────────────────────────────
        # TRAINING WITH SPATIAL GUILLOTINE & EARLY STOPPING (DATA.md, ES.md)
        # ─────────────────────────────────────────────────────────────────────
        start_time = time.time()
        
        # Get spatial masks (Y[50:200] train, Y[300:350] val)
        H, W = data.shape[0], data.shape[1]
        train_mask, val_mask = get_spatial_train_val_mask(H, W)
        
        print(f'Spatial guillotine protocol:')
        print(f'  Train block: Y[50:200] ({train_mask.sum()} pixels)')
        print(f'  Val block:   Y[300:350] ({val_mask.sum()} pixels)')
        
        # Create MASKED datasets for training iteration (gradient restriction)
        dataset_train = DatasetHsi(data_torch, wsize=block_size, wstride=3, spatial_mask=train_mask)
        dataset_val = DatasetHsi(data_torch, wsize=block_size, wstride=3, spatial_mask=val_mask)
        
        # Create FULL dataset for CMM search and reconstruction (needs all blocks)
        dataset_full = DatasetHsi(data_torch, wsize=block_size, wstride=3)
        
        data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, drop_last=False)
        data_loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False, drop_last=False)
        
        print(f'Train blocks: {len(dataset_train)}, Val blocks: {len(dataset_val)}, Full blocks: {len(dataset_full)}')
        
        # Block operations
        block_fold = BlockFold(wsize=block_size, wstride=3)
        block_search = BlockSearch(data_torch, wsize=block_size, wstride=3)
        
        # Loss and Optimizer
        mse = nn.MSELoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        
        # Early Stopping (ES.md)
        early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)
        
        # Training configuration
        search_iter = 25  # Search every 25 iterations
        
        # Initialize match vector for FULL dataset
        data_num_full = len(dataset_full)
        match_vec = torch.zeros((data_num_full)).to(device)
        search_index = torch.arange(0, data_num_full).type(torch.cuda.LongTensor)
        # search_matrix will be allocated only when needed (during search phases)
        
        # Training loop
        print('\nTraining with early stopping...')
        iteration = 0
        while not early_stopper.early_stop:
            iteration += 1
            # Fixed interval search scheduling
            should_search = (iteration % search_iter == 0)
            
            # Allocate search_matrix only when needed
            if should_search:
                search_matrix = torch.zeros((data_num_full, band, block_size, block_size)).to(device)
            
            # --- TRAINING PASS (masked blocks only) ---
            net.train()
            for idx, batch_data in enumerate(tqdm(data_loader_train, desc=f'Iter {iteration}', leave=False)):
                batch_gt = batch_data['block_gt'].to(device)
                batch_input = batch_data['block_input'].to(device)
                block_idx = batch_data['index'].to(device)
                
                optimizer.zero_grad()
                net_out = net(batch_input, block_idx=block_idx, match_vec=match_vec)
                
                if should_search:
                    search_matrix[block_idx] = net_out
                
                loss = mse(net_out, batch_gt)
                loss.backward()
                optimizer.step()
            
            # Content Matching Module (CMM) on training blocks
            if should_search:
                # Phase 3: Clear all memory to prepare for large CDist operation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                match_vec = torch.zeros((data_num_full)).to(device)
                
                # Use FULL dataset padding for reconstruction (CMM needs full grid)
                search_back = block_fold(search_matrix.detach(), dataset_full.padding, 
                                        data.shape[0], data.shape[1])
                
                # Phase 3: Clear memory before large CMM distance computation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Note: block_search uses full image, so we get matches across train zone
                match_vec = block_search(search_back.detach(), match_vec, search_index)
                print(f'    CMM updated at epoch {iteration}: {int(match_vec.sum().item())} blocks matched')
                
                # Phase 3: Power Management - free fragmented GPU memory after CMM search
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Free search_matrix after use to recover memory for training
                del search_matrix
            
            # --- VALIDATION PASS (ES.md) ---
            net.eval()
            val_loss_total = 0.0
            
            with torch.inference_mode():
                for batch_data in data_loader_val:
                    batch_input = batch_data['block_input'].to(device)
                    block_idx = batch_data['index'].to(device)
                    
                    net_out = net(batch_input, block_idx=block_idx, match_vec=match_vec)
                    loss = mse(net_out, batch_input)
                    val_loss_total += loss.item()
            
            avg_val_loss = val_loss_total / len(data_loader_val) if len(data_loader_val) > 0 else 0.0

            early_stopper(avg_val_loss)
            if early_stopper.early_stop:
                print(f'  Convergence reached at iteration {iteration}. Terminating training.')
            if dry_run:
                break
        
        train_time = time.time() - start_time
        print(f'Training completed in {train_time:.2f} seconds')
        
        # Save weights
        save_weights(net, food_type, arch='gthad')
    
    # Define avg_pool (used in both training and inference paths)
    avg_pool = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
    
    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE (on test data with ground truth anomalies)
    # ─────────────────────────────────────────────────────────────────────────
    print('\nRunning inference on test data...')
    net.eval()
    infer_start = time.time()
    
    # Clear memory aggressively before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert test data to tensor
    data_test_torch = torch.from_numpy(data_test).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    dataset_infer = DatasetHsi(data_test_torch, wsize=block_size, wstride=3)  # No spatial_mask
    # Reduce batch size for inference to minimize VRAM accumulation
    data_loader_infer = DataLoader(dataset_infer, batch_size=32, shuffle=False, drop_last=False)
    block_fold_infer = BlockFold(wsize=block_size, wstride=3)
    
    # Initialize for inference CMM
    data_num_infer = dataset_infer.__len__()
    match_vec_infer = torch.zeros((data_num_infer)).to(device)
    
    infer_res_list = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader_infer, desc='Inference', leave=False):
            batch_input = batch_data['block_input'].to(device)
            block_idx = batch_data['index'].to(device)
            
            net_out = net(batch_input, block_idx=block_idx, match_vec=match_vec_infer)
            infer_res = torch.abs(batch_input - net_out) ** 2
            infer_res = avg_pool(infer_res)
            infer_res_list.append(infer_res)
    
    infer_res_out = torch.cat(infer_res_list, dim=0)
    
    # Move fold operation to CPU to reduce VRAM pressure during averaging mask computation
    infer_res_cpu = infer_res_out.cpu()
    block_fold_infer.cpu()
    infer_res_back = block_fold_infer(infer_res_cpu, dataset_infer.padding,
                                       data_test.shape[0], data_test.shape[1])
    infer_res_back = infer_res_back.to(device)
    block_fold_infer.to(device)
    
    # Convert to anomaly score map
    residual_np = img2mask(infer_res_back)
    infer_time = time.time() - infer_start
    
    # Phase 3: Clear inference memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up large inference lists
    del infer_res_list, infer_res_out, infer_res_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────
    # gt was already converted to binary (0=normal, 1=anomaly) on line 105
    roc_auc = roc_auc_score(gt.flatten(), residual_np.flatten())
    pr_auc = average_precision_score(gt.flatten(), residual_np.flatten())
    
    print(f'\nResults:')
    print(f'  ROC-AUC: {roc_auc:.4f}')
    print(f'  PR-AUC:  {pr_auc:.4f}')
    
    # ─────────────────────────────────────────────────────────────────────────
    # VRAM TRACKING
    # ─────────────────────────────────────────────────────────────────────────
    max_vram_gb = 0.0
    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'detectmap_shape': list(residual_np.shape),
        'infer_time_sec': infer_time,
        'n_params': n_params,
        'max_vram_gb': max_vram_gb,
    }


if __name__ == '__main__':
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
    print('\n' + json.dumps(result, indent=2, default=str))

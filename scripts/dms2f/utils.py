"""Utility functions for DMS2F-HAD module."""

import os
import numpy as np
import spectral


def load_hsi_data(data_path):
    """Load HSI data from ENVI format."""
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)


def load_label(label_path):
    """Load binary anomaly labels."""
    return np.load(label_path)


def load_white_ref(path):
    """Load white reference for calibration."""
    img = spectral.open_image(path)
    return np.array(img.load(), dtype=np.float32)


def load_dark_ref(path):
    """Load dark reference for calibration."""
    img = spectral.open_image(path)
    return np.array(img.load(), dtype=np.float32)


def calibrate_hsi(data, white_ref_path, dark_ref_path):
    """
    Calibrate HSI data (white-dark correction).
    Preserves spatial lighting gradients.
    """
    white_ref = load_white_ref(white_ref_path)
    dark_ref = load_dark_ref(dark_ref_path)

    if len(white_ref.shape) == 3:
        white_profile = white_ref.mean(axis=0)
        dark_profile = dark_ref.mean(axis=0)
    else:
        white_profile = white_ref
        dark_profile = dark_ref

    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def get_spatial_train_val_mask(H=400, W=512):
    """
    Get spatial guillotine masks for train and validation zones.
    Protocol (percentages of H):
        Y[0:12.5%]:        Discarded (sensor startup noise)
        Y[12.5%:50%]:      Train block (37.5% of rows)
        Y[50%:75%]:        Dead zone (buffer)
        Y[75%:87.5%]:      Val block (12.5% of rows)
        Y[87.5%:H]:        Remaining (test zone)

    For H=400 this equals the classic Y[50:200] / Y[300:350]; larger cubes
    (e.g. AgriFood, H=1000) scale to Y[125:500] / Y[750:875].
    """
    train_y_start, train_y_end = round(0.125 * H), round(0.5 * H)
    val_y_start, val_y_end = round(0.75 * H), round(0.875 * H)

    train_mask = np.zeros((H, W), dtype=bool)
    train_mask[train_y_start:train_y_end, :] = True

    val_mask = np.zeros((H, W), dtype=bool)
    val_mask[val_y_start:val_y_end, :] = True

    return train_mask, val_mask


def get_random_train_val_mask(H, W, seed=42):
    """Generate random masks for training and validation.
    Random Sampling: 37.5% train, 12.5% val, remaining unlabeled.
    """
    np.random.seed(seed)
    rand_map = np.random.rand(H, W)

    train_mask = rand_map < 0.375
    val_mask = (rand_map >= 0.375) & (rand_map < 0.500)

    return train_mask, val_mask

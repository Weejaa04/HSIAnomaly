#!/usr/bin/env python3
"""Generate 10% noise variants (salt & pepper, gaussian, stripe, band) for all data.hdr files in the dataset.

Output naming:
    data_10            → salt & pepper (unchanged, backward compatible with noise.py)
    data_10_gaussian   → additive gaussian noise
    data_10_stripe     → vertical stripe noise
    data_10_band       → corrupted spectral bands
Each variant gets a matching *.noise_mask.npy marking the corrupted pixels.
"""

import numpy as np
import spectral.io.envi as envi
from pathlib import Path

DATASET = "AnomalyonFood/Dataset"
NOISE_PCT = 10
SEED = 42
NOISE_TYPES = ["saltpepper", "gaussian", "stripe", "band"]

GAUSSIAN_SIGMA_FRAC = 0.2  # sigma = frac * per-band std
STRIPE_SIGMA_FRAC = 0.2    # column offset sigma = frac * per-band std
STRIPE_MIN_WIDTH = 1
STRIPE_MAX_WIDTH = 3
BAND_SIGMA_FRAC = 1.0      # corrupted band: noise std = frac * band std


def out_stem(noise_type):
    if noise_type == "saltpepper":
        return f"data_{NOISE_PCT}"
    return f"data_{NOISE_PCT}_{noise_type}"


def add_saltpepper(data, rng):
    h, w, bands = data.shape
    total = h * w
    num_noisy = int(total * (NOISE_PCT / 100))
    num_salt = num_noisy // 2

    noisy = data.copy()
    coords = rng.choice(total, num_noisy, replace=False)
    for b in range(bands):
        flat = noisy[:, :, b].ravel()
        flat[coords[:num_salt]] = 1.0
        flat[coords[num_salt:]] = 0.0
        noisy[:, :, b] = flat.reshape(h, w)

    mask = np.zeros(total, dtype=bool)
    mask[coords] = True
    return noisy, mask.reshape(h, w)


def add_gaussian(data, rng):
    h, w, bands = data.shape
    total = h * w
    num_noisy = int(total * (NOISE_PCT / 100))

    sigma = GAUSSIAN_SIGMA_FRAC * data.std(axis=(0, 1))
    coords = rng.choice(total, num_noisy, replace=False)
    noisy = data.copy()
    noise = rng.normal(0.0, 1.0, size=(num_noisy, bands)) * sigma
    noisy.reshape(total, bands)[coords] += noise

    mask = np.zeros(total, dtype=bool)
    mask[coords] = True
    return noisy, mask.reshape(h, w)


def add_stripe(data, rng):
    h, w, bands = data.shape
    total = h * w
    target = int(total * (NOISE_PCT / 100))

    sigma = STRIPE_SIGMA_FRAC * data.std(axis=(0, 1))
    noisy = data.copy()
    mask = np.zeros((h, w), dtype=bool)

    covered = 0
    while covered < target:
        col = int(rng.integers(w))
        width = int(rng.integers(STRIPE_MIN_WIDTH, STRIPE_MAX_WIDTH + 1))
        cols = np.arange(col, min(col + width, w))
        cols = cols[~mask[:, cols].any(axis=0)]
        if cols.size == 0:
            continue
        remaining = target - covered
        if h * cols.size > remaining:
            cols = cols[: (remaining + h - 1) // h]
        if cols.size == 0:
            continue
        for c in cols:
            offset = rng.normal(0.0, 1.0, size=bands) * sigma
            noisy[:, c, :] += offset
            mask[:, c] = True
        covered += h * cols.size

    return noisy, mask


def add_band(data, rng):
    h, w, bands = data.shape
    num_bands = max(1, round(bands * (NOISE_PCT / 100)))

    band_idx = rng.choice(bands, num_bands, replace=False)
    band_mean = data.mean(axis=(0, 1))
    band_std = data.std(axis=(0, 1))

    noisy = data.copy()
    for b in band_idx:
        noisy[:, :, b] = rng.normal(band_mean[b], BAND_SIGMA_FRAC * band_std[b], size=(h, w))

    mask = np.ones((h, w), dtype=bool)
    return noisy, mask


GENERATORS = {
    "saltpepper": add_saltpepper,
    "gaussian": add_gaussian,
    "stripe": add_stripe,
    "band": add_band,
}


def generate(food_types=None, noise_types=None, seed=None):
    if food_types is None:
        food_types = ["Almond", "Pistachio", "GarlicStems"]
    if noise_types is None:
        noise_types = NOISE_TYPES
    if seed is None:
        seed = SEED
    rng = np.random.default_rng(seed)

    for ft in food_types:
        for split in ["Train", "Test"]:
            for noise_type in noise_types:
                stem = out_stem(noise_type)
                hdr_path = Path(DATASET) / ft / split / "data.hdr"
                out_hdr = Path(DATASET) / ft / split / f"{stem}.hdr"
                out_raw = Path(DATASET) / ft / split / stem
                mask_npy = out_hdr.with_suffix('.noise_mask.npy')

                if out_hdr.exists() and out_raw.exists() and mask_npy.exists():
                    print(f"  ✅ {ft}/{split} ({stem}.hdr): already exists")
                    continue

                print(f"  Generating {ft}/{split} {stem}.hdr ({NOISE_PCT}% {noise_type})...")
                img = envi.open(str(hdr_path))
                data = np.array(img.load(), dtype=np.float32)

                noisy, aggregate_mask = GENERATORS[noise_type](data, rng)

                envi.save_image(str(out_hdr), noisy, dtype=np.float32,
                                metadata=img.metadata, ext="", force=True)
                np.save(str(mask_npy), aggregate_mask.astype(np.uint8))
                print(f"  ✅ {ft}/{split} ({stem}.hdr): saved")


if __name__ == "__main__":
    generate(seed=SEED)
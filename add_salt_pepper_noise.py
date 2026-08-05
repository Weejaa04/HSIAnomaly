#!/usr/bin/env python3
"""Generate salt & pepper noise at 5%, 10%, 20% for all data.hdr files in the dataset."""

import numpy as np
import spectral.io.envi as envi
from pathlib import Path

DATASET = "AnomalyonFood/Dataset"
NOISE_LEVELS = [5, 10, 20]
SEED = 42


def generate(food_types=None, levels=None, seed=None):
    if food_types is None:
        food_types = ["Almond", "Pistachio", "GarlicStems"]
    if levels is None:
        levels = NOISE_LEVELS
    if seed is None:
        seed = SEED
    np.random.seed(seed)

    for ft in food_types:
        for split in ["Train", "Test"]:
            for pct in levels:
                hdr_path = Path(DATASET) / ft / split / "data.hdr"
                out_hdr = Path(DATASET) / ft / split / f"data_{pct}.hdr"
                out_raw = Path(DATASET) / ft / split / f"data_{pct}"

                mask_npy = out_hdr.with_suffix('.noise_mask.npy')
                if out_hdr.exists() and out_raw.exists() and mask_npy.exists():
                    print(f"  ✅ {ft}/{split} (data_{pct}.hdr): already exists")
                    continue

                print(f"  Generating {ft}/{split} data_{pct}.hdr ({pct}%)...")
                img = envi.open(str(hdr_path))
                data = np.array(img.load(), dtype=np.float32)

                h, w, bands = data.shape
                total = h * w
                num_noisy = int(total * (pct / 100))
                num_salt = num_noisy // 2

                noisy = data.copy()
                coords = np.random.choice(total, num_noisy, replace=False)
                for b in range(bands):
                    flat = noisy[:, :, b].ravel()
                    flat[coords[:num_salt]] = 1.0
                    flat[coords[num_salt:]] = 0.0
                    noisy[:, :, b] = flat.reshape(h, w)
                noise_mask = np.zeros(h * w, dtype=bool)
                noise_mask[coords] = True
                aggregate_mask = noise_mask.reshape(h, w)

                envi.save_image(str(out_hdr), noisy, dtype=np.float32,
                                metadata=img.metadata, ext="", force=True)
                mask_npy = out_hdr.with_suffix('.noise_mask.npy')
                np.save(str(mask_npy), aggregate_mask.astype(np.uint8))
                print(f"  ✅ {ft}/{split} (data_{pct}.hdr): saved")


if __name__ == "__main__":
    generate(seed=SEED)

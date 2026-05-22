#!/usr/bin/env python3
"""Generate salt & pepper noise at 5%, 10%, 20% for all data.hdr files in the dataset."""

import numpy as np
import spectral.io.envi as envi
from pathlib import Path

DATASET = "AnomalyonFood/Dataset"
NOISE_LEVELS = [5, 10, 20]


def generate(food_types=None, levels=None):
    if food_types is None:
        food_types = ["Almond", "Pistachio", "GarlicStems"]
    if levels is None:
        levels = NOISE_LEVELS

    for ft in food_types:
        for split in ["Train", "Test"]:
            for pct in levels:
                hdr_path = Path(DATASET) / ft / split / "data.hdr"
                out_hdr = Path(DATASET) / ft / split / f"data_{pct}.hdr"
                out_raw = Path(DATASET) / ft / split / f"data_{pct}"

                if out_hdr.exists() and out_raw.exists():
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
                for b in range(bands):
                    flat = noisy[:, :, b].ravel()
                    coords = np.random.choice(total, num_noisy, replace=False)
                    flat[coords[:num_salt]] = 1.0
                    flat[coords[num_salt:]] = 0.0
                    noisy[:, :, b] = flat.reshape(h, w)

                envi.save_image(str(out_hdr), noisy, dtype=np.float32,
                                metadata=img.metadata, ext="", force=True)
                print(f"  ✅ {ft}/{split} (data_{pct}.hdr): saved")


if __name__ == "__main__":
    generate()

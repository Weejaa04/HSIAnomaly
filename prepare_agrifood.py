#!/usr/bin/env python3
"""
prepare_agrifood.py — Build the AgriFood dataset in the loader format used by the
benchmark suites (mirror of AnomalyonFood/Dataset/<food>).

Source: AgriFoodAnomaly/  (UseCase_1_(Avoine1))
    - Train+val  -> UseCase_1_(Avoine1)_Normal_3              (spatial guillotine split)
    - Test step1 -> UseCase_1_(Avoine1)_Anomaly_Easy_L4_7
    - Test step2 -> UseCase_1_(Avoine1)_Anomaly_Easy_L12_8     (staged as data_L12_8.*)

Layout produced:
    AgriFood/Dataset/AgriFood/
        Train/{data.hdr,data.bil,WHITEREF.hdr,WHITEREF.bil,DARKREF.hdr,DARKREF.bil}
        Test/{data.hdr,data.bil,label.npy,
              data_L12_8.hdr,data_L12_8.bil,label_L12_8.npy,
              WHITEREF.hdr,WHITEREF.bil,DARKREF.hdr,DARKREF.bil}

The AgriFood cubes are already radiometrically calibrated reflectance
(scale factor 10000) so a synthetic WHITEREF=65535 / DARKREF=0 makes the
WHITE/DARK correction in the loaders an identity mapping onto [0,1].

Usage:
    python prepare_agrifood.py
"""

import os
import shutil

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "AgriFoodAnomaly")
OUT = os.path.join(ROOT, "AgriFood", "Dataset", "AgriFood")

TRAIN_SPLIT = "train_UseCase_1_(Avoine1)"
VAL_SPLIT = "val_UseCase_1_(Avoine1)"
TEST_SPLIT = "test_UseCase_1_(Avoine1)"

CUBE_NAME = "UseCase_1_(Avoine1)"
TRAIN_CUBE = f"{CUBE_NAME}_Normal_3"
TEST_STEP1 = f"{CUBE_NAME}_Anomaly_Easy_L4_7"
TEST_STEP2 = f"{CUBE_NAME}_Anomaly_Easy_L12_8"

LINES = 1000
SAMPLES = 900
BANDS = 300


def write_envi_header(path, lines, samples, bands, dtype=12, bname=None):
    """Write a minimal ENVI header for a BIL-interleaved file."""
    with open(path, "w") as f:
        f.write("ENVI\n")
        f.write("file type = ENVI BIL Format\n")
        f.write(f"data type = {dtype}\n")
        f.write("interleave = bil\n")
        f.write("byte order = 0\n")
        f.write(f"lines = {lines}\n")
        f.write(f"samples = {samples}\n")
        f.write(f"bands = {bands}\n")
        f.write("header offset = 0\n")
        if bname:
            f.write(f"band names = {{{bname}}}\n")


def add_ref_files(dir_path):
    """Add WHITEREF/DARKREF (identity calibration refs) to a dataset dir."""
    for name, value in [("WHITEREF", 65535), ("DARKREF", 0)]:
        hdr = os.path.join(dir_path, f"{name}.hdr")
        bil = os.path.join(dir_path, f"{name}.bil")
        write_envi_header(hdr, lines=1, samples=SAMPLES, bands=BANDS)
        ref = np.full((1, SAMPLES, BANDS), value, dtype=np.uint16)
        ref.tofile(bil)


def copy_cube(split_name, cube_name, dst_basename):
    """Copy the raw .bil cube + write a loader-friendly header."""
    src_dir = os.path.join(SRC, split_name, "HSI-Hybercube")
    shutil.copyfile(os.path.join(src_dir, f"{cube_name}.bil"),
                    os.path.join(OUT, dst_basename + ".bil"))
    write_envi_header(os.path.join(OUT, dst_basename + ".hdr"),
                      LINES, SAMPLES, BANDS, bname=cube_name)


def mask_to_label(mask_png_path):
    """Convert a binary annotation PNG -> (H, W) label.npy (2=background, 11=anomaly)."""
    mask = np.array(Image.open(mask_png_path).convert("L"))
    return np.where(mask > 0, 11, 2).astype(np.uint8)


def main():
    train_dir = os.path.join(OUT, "Train")
    test_dir = os.path.join(OUT, "Test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Train (+val, spatial guillotine) <- Normal_3
    copy_cube(TRAIN_SPLIT, TRAIN_CUBE, os.path.join(train_dir, "data"))
    add_ref_files(train_dir)

    # Test step 1 <- L4_7 (default data.*)
    copy_cube(TEST_SPLIT, TEST_STEP1, os.path.join(test_dir, "data"))
    test1_mask = os.path.join(SRC, TEST_SPLIT, "Annotation", "PNG", f"{TEST_STEP1}.png")
    np.save(os.path.join(test_dir, "label.npy"), mask_to_label(test1_mask))

    # Test step 2 <- L12_8 (staged as data_L12_8.*, swapped in by benchmark.py)
    copy_cube(TEST_SPLIT, TEST_STEP2, os.path.join(test_dir, "data_L12_8"))
    test2_mask = os.path.join(SRC, TEST_SPLIT, "Annotation", "PNG", f"{TEST_STEP2}.png")
    np.save(os.path.join(test_dir, "label_L12_8.npy"), mask_to_label(test2_mask))

    add_ref_files(test_dir)

    print(f"AgriFood dataset prepared -> {OUT}")
    for dirpath, _, filenames in os.walk(OUT):
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            print(f"  {os.path.relpath(p, ROOT)}  ({os.path.getsize(p):,} B)")


if __name__ == "__main__":
    main()
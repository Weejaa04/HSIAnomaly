#!/usr/bin/env python3
"""
anomalymap.py — Generate anomaly map visualizations from .npy score files.

Takes anomaly score arrays from results/anomaly-map/ and produces heatmap images
with normalized 0-1 colorbar legend, even though each array may have different ranges.

Usage:
    python anomalymap.py                          # Process all scores
    python anomalymap.py --food Almond            # Process specific food type
    python anomalymap.py --arch our pa2e          # Process specific architectures
    python anomalymap.py --output-dir ./viz       # Custom output directory
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def normalize_array(arr):
    """Normalize array to [0, 1] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min)


# Canonical architecture order for visualization
ARCH_ORDER = [
    "gthad",
    "pa2e",
    "bocknet",
    "autoad",
    "superad",
    "otad",
    "sglnet",
    "dms2f",
    "our",
]

def get_architecture_name(arch):
    """Map architecture code to display name."""
    display_names = {
        "bocknet": "BockNet",
        "our": "Ours",
        "pa2e": "PA2E",
        "gthad": "GT-HAD",
        "superad": "SuperAD",
        "sglnet": "SGLNet",
        "autoad": "Auto-AD",
        "otad": "OT-AD",
        "dms2f": "DMS2F-HAD",
    }
    return display_names.get(arch, arch)


def create_single_heatmap(score_array, arch, food, output_dir):
    """Create a single heatmap image for a score array."""
    # Normalize to [0, 1]
    normalized = normalize_array(score_array)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with 'hot' colormap (dark to yellow/white)
    im = ax.imshow(normalized, cmap='hot', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar with normalized 0-1 range
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
    
    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    fig.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, f'{arch}_{food}_anomaly_map.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_file


def create_grid_heatmaps(foods, archs, score_dir, output_dir):
    """Create a grid of heatmaps (architectures x foods, with GT as first row)."""
    scores = {}
    gt_labels = {}

    # Load all score arrays
    for food in foods:
        scores[food] = {}
        for arch in archs:
            score_file = os.path.join(score_dir, f'{arch}_{food}_scores.npy')
            if os.path.exists(score_file):
                scores[food][arch] = np.load(score_file)

        # Load GT label
        gt_file = os.path.join(score_dir, f'{food}_labels.npy')
        if os.path.exists(gt_file):
            gt_labels[food] = np.load(gt_file)

    if not scores:
        print(f"  ❌ No score files found")
        return

    # Rows: GT + architectures, Columns: food types
    display_archs = ['GT'] + ARCH_ORDER
    num_rows = len(display_archs)
    num_cols = len(foods)
    foods_sorted = sorted(foods)

    # Preserve original image aspect ratio 400×512 (width:height = 0.78125)
    aspect_ratio = 400 / 512
    subplot_height = 1.0
    subplot_width = subplot_height * aspect_ratio

    figwidth = num_cols * subplot_width + 1.2  # extra space for vertical colorbar
    figheight = num_rows * subplot_height + 0.5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figwidth, figheight))

    for row_idx, arch_label in enumerate(display_archs):
        for col_idx, food in enumerate(foods_sorted):
            ax = axes[row_idx, col_idx]

            if arch_label == 'GT':
                if food in gt_labels:
                    im = ax.imshow(gt_labels[food], cmap='hot', aspect='auto', vmin=0, vmax=1)
            else:
                if food in scores and arch_label in scores[food]:
                    score_array = scores[food][arch_label]
                    normalized = normalize_array(score_array)
                    im = ax.imshow(normalized, cmap='hot', aspect='auto', vmin=0, vmax=1)

            # Food names at top, architecture names at left
            if row_idx == 0:
                ax.set_title(food, fontsize=10, fontweight='bold')
            if col_idx == 0:
                display_name = 'GroundTruth' if arch_label == 'GT' else get_architecture_name(arch_label)
                ax.set_ylabel(display_name, fontsize=9, fontweight='bold')

            # Remove labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Add vertical colorbar to the right
    fig.subplots_adjust(right=0.90, wspace=0.02, hspace=0.1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(0.5, -0.08, 'Anomaly absent', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=9)
    cbar.ax.text(0.5, 1.08, 'Anomaly present', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=9)

    output_file = os.path.join(output_dir, f'anomaly_maps_grid.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Grid saved → {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate anomaly map visualizations from .npy score files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anomalymap.py
  python anomalymap.py --food Almond
  python anomalymap.py --arch our pa2e
  python anomalymap.py --food Almond Pistachio --output-dir ./viz
        """,
    )
    parser.add_argument(
        "--food",
        nargs="*",
        default=None,
        metavar="FOOD",
        help="Food types to visualize (default: all found)",
    )
    parser.add_argument(
        "--arch",
        nargs="*",
        default=None,
        metavar="ARCH",
        help="Architectures to visualize (default: all found)",
    )
    parser.add_argument(
        "--score-dir",
        default="./results/anomaly-map",
        help="Directory containing .npy score files (default: ./results/anomaly-map)",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/anomaly-maps-viz",
        help="Output directory for visualization images (default: ./results/anomaly-maps-viz)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create grid visualizations (multiple archs per food type)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Create individual heatmaps (default: true if no --grid)",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover food types and architectures
    if not os.path.exists(args.score_dir):
        print(f"❌ Score directory not found: {args.score_dir}")
        return
    
    score_files = glob.glob(os.path.join(args.score_dir, "*_scores.npy"))
    
    # Extract unique food types and architectures
    food_types_found = set()
    archs_found = set()
    
    for f in score_files:
        basename = os.path.basename(f)
        # Format: {arch}_{food}_scores.npy
        parts = basename.replace('_scores.npy', '').rsplit('_', 1)
        if len(parts) == 2:
            arch, food = parts
            food_types_found.add(food)
            archs_found.add(arch)
    
    # Filter by user selection
    food_types = args.food if args.food else sorted(food_types_found)
    archs = args.arch if args.arch else sorted(archs_found)
    
    if not food_types or not archs:
        print(f"❌ No score files found or no matches")
        return
    
    print(f"\n{'=' * 80}")
    print(f"  Anomaly Map Visualization")
    print(f"  Score dir     : {args.score_dir}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Food types    : {', '.join(food_types)}")
    print(f"  Architectures : {', '.join(archs)}")
    print(f"{'=' * 80}\n")
    
    # Create visualizations
    create_grid = args.grid or not args.individual  # Default to grid unless --individual
    create_ind = args.individual or not args.grid    # Default to individual unless --grid only
    
    if create_grid:
        print(f"📊 Creating grid (GT + architectures × food types)...")
        create_grid_heatmaps(food_types, archs, args.score_dir, args.output_dir)
    
    if create_ind:
        for food in food_types:
            print(f"📊 Creating individual maps for {food}...")
            for arch in archs:
                score_file = os.path.join(args.score_dir, f'{arch}_{food}_scores.npy')
                if os.path.exists(score_file):
                    score_array = np.load(score_file)
                    output_file = create_single_heatmap(score_array, arch, food, args.output_dir)
                    print(f"  ✅ {arch}: {output_file}")
    
    print(f"\n✅ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract and plot TensorBoard logs as static images.
Requires: pip install tensorboard matplotlib
"""
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard package not found. Install with: pip install tensorboard")
    exit(1)

def extract_scalars(tb_log_dir, scalar_tags=None):
    """Extract scalar values from TensorBoard logs."""
    event_acc = EventAccumulator(tb_log_dir)
    event_acc.Reload()
    
    available_tags = event_acc.Tags()['scalars']
    
    if scalar_tags is None:
        scalar_tags = available_tags
    
    results = {}
    for tag in scalar_tags:
        if tag in available_tags:
            scalar_events = event_acc.Scalars(tag)
            steps = [s.step for s in scalar_events]
            values = [s.value for s in scalar_events]
            results[tag] = {'steps': steps, 'values': values}
    
    return results, available_tags

def plot_results(results, output_file=None, title="Training Metrics"):
    """Plot extracted scalar values."""
    n_plots = len(results)
    if n_plots == 0:
        print("No data to plot")
        return
    
    # Determine grid size
    cols = 2
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (tag, data) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(data['steps'], data['values'], linewidth=2)
        ax.set_xlabel('Step/Epoch')
        ax.set_ylabel('Value')
        ax.set_title(tag.replace('/', ' - '))
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract and plot TensorBoard logs')
    parser.add_argument('log_dir', type=str, help='TensorBoard log directory')
    parser.add_argument('--output', '-o', type=str, help='Output image file (e.g., plot.png)')
    parser.add_argument('--tags', nargs='+', help='Specific tags to plot (default: all)')
    parser.add_argument('--title', type=str, help='Plot title')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Directory {args.log_dir} does not exist")
        return
    
    print(f"Loading TensorBoard logs from: {args.log_dir}")
    results, available_tags = extract_scalars(args.log_dir, args.tags)
    
    if not results:
        print(f"No scalar data found. Available tags: {available_tags}")
        return
    
    print(f"\nFound {len(results)} metrics:")
    for tag in results.keys():
        print(f"  - {tag} ({len(results[tag]['steps'])} data points)")
    
    title = args.title or f"Training Metrics - {os.path.basename(args.log_dir)}"
    plot_results(results, args.output, title)

if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to generate plots from saved metrics JSON file.
Useful for regenerating plots without retraining the model.

Usage:
    python generate_plots.py <path_to_metrics.json>
    python generate_plots.py logs/TextGNN_jigsaw_*/plots/metrics.json
"""

import sys
import json
import os
from os.path import join, dirname, basename
from plot_metrics import (plot_all_metrics, plot_final_test_metrics, 
                          create_training_summary_plot)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <path_to_metrics.json>")
        print("\nExample:")
        print("  python generate_plots.py logs/TextGNN_jigsaw_0.8_0.1_0.1_2025-12-01T18-27-42/plots/metrics.json")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    
    if not os.path.exists(metrics_file):
        print(f"Error: File not found: {metrics_file}")
        sys.exit(1)
    
    print(f"Loading metrics from: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    history = data['training_history']
    test_results = data['test_results']
    
    # Create output directory
    output_dir = dirname(metrics_file)
    print(f"Saving plots to: {output_dir}")
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_all_metrics(history, output_dir)
    create_training_summary_plot(history, test_results, join(output_dir, 'training_summary.png'))
    plot_final_test_metrics(test_results, join(output_dir, 'test_metrics.png'))
    
    print(f"\n✓ All plots regenerated successfully!")
    print(f"  Location: {output_dir}")


if __name__ == "__main__":
    main()

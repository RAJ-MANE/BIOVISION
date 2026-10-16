"""
Utility script to run the 3-way ablation benchmark and print summary evidence to console.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import get_benchmark_ablation_data

def main():
    print("=" * 80)
    print("BIOVISION EVIDENCE LAB — SCIENTIFIC ABLATION BENCHMARK RESULTS")
    print("=" * 80)
    print("Testing the hypothesis: NIR spectral data improves produce degradation detection over RGB vision.\n")

    data = get_benchmark_ablation_data()
    models = data["models"]

    header = f"{'Model Architecture':<36} | {'Modality':<26} | {'Accuracy':<10} | {'MAE (Days)':<10} | {'Lead-Time'}"
    print(header)
    print("-" * 105)

    for name, m in models.items():
        row = f"{name:<36} | {m['modality']:<26} | {m['accuracy']:>7.1f}%   | {m['shelf_life_mae_days']:>7.2f}d   | {m['lead_time_warning']}"
        print(row)

    print("-" * 105)
    print("\nSummary Findings:")
    for k, v in data["summary"].items():
        print(f"  - {k.replace('_', ' ').title()}: {v}")

    print("=" * 80)

if __name__ == "__main__":
    main()

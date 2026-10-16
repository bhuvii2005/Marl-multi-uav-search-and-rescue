"""
Plot mean_time_to_find_all across updates for multiple runs, for the
comparison required in report Section 3.4 / 3.5.

Usage:
    python plot_results.py runs/belief_main runs/baseline_coverage runs/baseline_static_belief
"""
import csv
import sys
import matplotlib.pyplot as plt


def load(path):
    updates, times, redundant = [], [], []
    with open(f"{path}/log.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            updates.append(int(row["update"]))
            times.append(float(row["mean_time_to_find_all"]))
            redundant.append(float(row["mean_redundant_hits"]))
    return updates, times, redundant


def main():
    run_dirs = sys.argv[1:]
    if not run_dirs:
        print("Usage: python plot_results.py <run_dir> [run_dir ...]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for run_dir in run_dirs:
        u, t, r = load(run_dir)
        label = run_dir.rstrip("/").split("/")[-1]
        axes[0].plot(u, t, label=label)
        axes[1].plot(u, r, label=label)

    axes[0].set_title("Time-to-find-all-victims")
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("Steps")
    axes[0].legend()

    axes[1].set_title("Redundant re-visits")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Count / episode")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    print("Saved comparison.png")


if __name__ == "__main__":
    main()

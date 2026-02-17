import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from config import LOG_DIR


def _cast_value(name, value):
    if name in {"episode", "steps", "success", "illegal"}:
        return int(float(value))
    return float(value)


def load_metrics(csv_path):
    columns = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}
        columns = {name: [] for name in header}
        for row in reader:
            for idx, name in enumerate(header):
                columns[name].append(_cast_value(name, row[idx]))

    for name, values in columns.items():
        columns[name] = np.array(values)
    return columns


def moving_average(values, window=200):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    flat_path = os.path.join(LOG_DIR, "flat_returns.csv")
    opt_path = os.path.join(LOG_DIR, "options_returns.csv")

    if not os.path.exists(flat_path) or not os.path.exists(opt_path):
        raise RuntimeError("Missing logs. Run train_flat.py and train_options.py first.")

    flat = load_metrics(flat_path)
    opt = load_metrics(opt_path)


    metrics = [
        ("return", "Return"),
        ("steps", "Steps per Episode"),
        ("success", "Success Rate"),
        ("illegal", "Illegal Action Count"),
        ("efficiency", "Efficiency (Optimal / Actual Steps)"),
        ("illegal_ratio", "Illegal Action Ratio"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (key, label) in enumerate(metrics):
        ax = axes[idx]
        if key not in flat or key not in opt:
            ax.set_visible(False)
            continue

        flat_ep = flat["episode"]
        opt_ep = opt["episode"]
        flat_vals = flat[key]
        opt_vals = opt[key]

        ax.plot(flat_ep, flat_vals, alpha=0.2, label="Flat (raw)")
        ax.plot(opt_ep, opt_vals, alpha=0.2, label="Options (raw)")

        flat_ma = moving_average(flat_vals)
        opt_ma = moving_average(opt_vals)

        if len(flat_ma) > 0:
            ax.plot(flat_ep[len(flat_ep) - len(flat_ma):], flat_ma, label="Flat (MA)")
        if len(opt_ma) > 0:
            ax.plot(opt_ep[len(opt_ep) - len(opt_ma):], opt_ma, label="Options (MA)")

        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import csv
import json
import logging
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False


def create_exp_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Experiment dir : {path}")


def setup_logger(exp_dir):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(exp_dir) / "log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_args(args, exp_dir):
    with open(Path(exp_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def append_metrics_csv(path, row):
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_trend(train_list, val_list, title, save_dir):
    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(train_list, label="Training", c="#1f77b4")
    plt.plot(val_list, label="Validation", c="#ff7f0e")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title}.png")
    plt.close()


def plot_confusion_matrix(con_mat, save_path, class_names=None, normalize=False, title="Confusion Matrix"):
    cm = np.array(con_mat, dtype=np.float32)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(1.1 * num_classes + 3, 1.0 * num_classes + 2))
    vmax = np.quantile(cm, 0.99) if cm.size else 1.0
    vmax = max(float(vmax), 1e-6)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    fmt = ".2f" if normalize else "d"
    threshold = vmax * 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            text = format(value, fmt) if normalize else str(int(round(value)))
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

"""
Tabular permutation importance + MFCC gradient saliency for the FusionLSTM model.

Usage (from project root):
    uv run python src/feature_importance.py
    uv run python src/feature_importance.py --n-samples 200 --repeats 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from modules.lstm.lstm_model import FusionLSTMModel
from torch_datasets.lstm_dataset import LSTMBeeAudioDataset, pad_collate_fn

# ── Paths & constants ─────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parents[1] / "trained_models" / "lstm" / "lstm_model.pth"
CSV_PATH   = Path(__file__).parent / "data" / "all_data_updated.csv"
AUDIO_DIR  = Path(__file__).parent / "data" / "sound_files" / "sound_files"
N_MFCC     = 40

TABULAR_COLS = [
    "hive temp", "hive humidity", "hive pressure",
    "frames",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "day_sin", "day_cos",
    "day_of_week_sin", "day_of_week_cos",
    "device_2",
    "hive number_3", "hive number_4", "hive number_5",
]

_DROP = [
    "weatherID", "lat", "long", "rain", "queen acceptance",
    "target", "queen status", "time", "gust speed",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model() -> FusionLSTMModel:
    model = FusionLSTMModel(
        lstm_layers=(N_MFCC, 64, 2, 0.4),
        features_input_size=32,
        embeddings_model=(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32)),
        ff_hidden_size=64,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy().drop(columns=[c for c in _DROP if c in df.columns])

    dt = pd.to_datetime(df["date"])
    for feat, raw, period in [
        ("hour",        dt.dt.hour,      24),
        ("minute",      dt.dt.minute,    60),
        ("day",         dt.dt.day,       31),
        ("day_of_week", dt.dt.dayofweek, 7),
    ]:
        df[f"{feat}_sin"] = np.sin(2 * np.pi * raw / period)
        df[f"{feat}_cos"] = np.cos(2 * np.pi * raw / period)
    df = df.drop(columns=["date"])

    df["device_2"] = (df["device"] == 2).astype(float)
    df = df.drop(columns=["device"])

    for n in (3, 4, 5):
        df[f"hive number_{n}"] = (df["hive number"] == n).astype(float)
    df = df.drop(columns=["hive number"])

    _WEATHER = ["weather temp", "weather humidity", "weather pressure",
                "wind speed", "cloud coverage"]
    df = df.drop(columns=[c for c in _WEATHER if c in df.columns])

    y  = df["queen presence"].fillna(0).astype(int)
    df = df.drop(columns=["queen presence"])

    return df[["file name"] + TABULAR_COLS], y


def make_loader(x: pd.DataFrame, y: pd.Series) -> DataLoader:
    dataset = LSTMBeeAudioDataset(
        features=x,
        labels=y,
        audio_dir=AUDIO_DIR,
        n_mfcc=N_MFCC,
        columns_to_drop=["file name"],
    )
    return DataLoader(
        dataset, batch_size=16, shuffle=False,
        num_workers=0, collate_fn=pad_collate_fn,
    )

# ── Permutation importance ────────────────────────────────────────────────────
@torch.no_grad()
def _run_accuracy(model: FusionLSTMModel, loader: DataLoader,
                  perturb_col: int | None = None) -> float:
    correct = total = 0
    for batch_mel, batch_tab, batch_labels, batch_lengths in loader:
        if perturb_col is not None:
            batch_tab = batch_tab.clone()
            perm = torch.randperm(batch_tab.size(0))
            batch_tab[:, perturb_col] = batch_tab[perm, perturb_col]

        logits = model(batch_mel, batch_tab, batch_lengths)
        preds  = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
        correct += (preds == batch_labels).sum().item()
        total   += batch_labels.size(0)
    return correct / total if total else 0.0


def permutation_importance(model: FusionLSTMModel, loader: DataLoader,
                           n_repeats: int = 5) -> np.ndarray:
    baseline = _run_accuracy(model, loader)
    print(f"Baseline accuracy : {baseline:.4f}\n")

    scores = np.zeros((len(TABULAR_COLS), n_repeats))
    for i, col in enumerate(TABULAR_COLS):
        for r in range(n_repeats):
            scores[i, r] = baseline - _run_accuracy(model, loader, perturb_col=i)
        print(f"  [{i:02d}] {col:<32}  drop = {scores[i].mean():+.4f} ± {scores[i].std():.4f}")
    return scores

# ── MFCC gradient saliency ────────────────────────────────────────────────────
def mfcc_saliency(model: FusionLSTMModel, loader: DataLoader) -> np.ndarray:
    grad_sum = np.zeros(N_MFCC)
    n_batches = 0

    for batch_mel, batch_tab, _, batch_lengths in loader:
        mel = batch_mel.detach().requires_grad_(True)
        logits = model(mel, batch_tab, batch_lengths)
        torch.sigmoid(logits).mean().backward()
        model.zero_grad()

        # average |grad| across [batch, time] → (n_mfcc,)
        grad_sum  += mel.grad.abs().mean(dim=(0, 1)).detach().numpy()
        n_batches += 1

    saliency = grad_sum / max(n_batches, 1)
    top5 = np.argsort(saliency)[::-1][:5]
    print("\n  Top-5 MFCC coefficients by saliency:")
    for rank, idx in enumerate(top5, 1):
        print(f"    {rank}. MFCC-{idx:02d}  {saliency[idx]:.6f}")
    return saliency

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(importances: np.ndarray, saliency: np.ndarray) -> None:
    mean_imp = importances.mean(axis=1)
    std_imp  = importances.std(axis=1)
    order    = np.argsort(mean_imp)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("FusionLSTM — Feature Importance", fontsize=14, fontweight="bold")

    # ── Tabular permutation importance (sorted, horizontal) ───────────────────
    y_pos  = np.arange(len(TABULAR_COLS))
    colors = ["#e8970a" if v > 0 else "#888888" for v in mean_imp[order]]
    ax1.barh(y_pos, mean_imp[order], xerr=std_imp[order],
             color=colors, align="center",
             error_kw={"elinewidth": 1.2, "capsize": 3})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([TABULAR_COLS[i] for i in order], fontsize=9)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Accuracy drop when permuted  (↑ more important)")
    ax1.set_title("Tabular features — Permutation importance")
    ax1.invert_yaxis()

    # ── MFCC gradient saliency (by coefficient index) ────────────────────────
    coeff_idx = np.arange(N_MFCC)
    norm_sal  = saliency / saliency.max()          # normalise for readability
    bar_colors = plt.cm.YlOrBr(norm_sal)           # warm gradient = higher
    ax2.bar(coeff_idx, saliency, color=bar_colors)
    ax2.set_xlabel("MFCC coefficient index")
    ax2.set_ylabel("Mean |∂output / ∂coeff|")
    ax2.set_title("MFCC coefficients — Gradient saliency")
    ax2.set_xticks(np.arange(0, N_MFCC, 5))

    plt.tight_layout()

    out = Path(__file__).parent / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit analysis to first N rows of the CSV")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Permutation repeats per tabular feature (default: 5)")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    if args.n_samples:
        df = df.head(args.n_samples)
    print(f"Loaded {len(df)} rows from CSV")

    x, y = preprocess(df)
    print(f"Preprocessing done — {len(x)} rows, {len(TABULAR_COLS)} tabular features")

    loader = make_loader(x, y)
    n_kept = len(loader.dataset)
    print(f"Dataset: {n_kept} samples kept after audio availability check")
    if n_kept == 0:
        print("No audio found — check AUDIO_DIR path.")
        sys.exit(1)

    model = load_model()

    print("\n── Permutation importance ──────────────────────────────────────────")
    importances = permutation_importance(model, loader, n_repeats=args.repeats)

    print("\n── MFCC gradient saliency ──────────────────────────────────────────")
    saliency = mfcc_saliency(model, loader)

    plot(importances, saliency)


if __name__ == "__main__":
    main()
